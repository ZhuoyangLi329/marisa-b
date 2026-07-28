#include "bias_operators.h"
#include "field_kernel_provider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo_v1.h"
#include "marisa_b_native.h"
#include "PowerSpectrum.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;

namespace {

int checks=0;

void require(bool condition,const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual,double expected,double relative,double absolute,const std::string& message) {
    ++checks;
    const double error=std::fabs(actual-expected);
    if (error>absolute && error>relative*std::max(std::fabs(actual),std::fabs(expected))) {
        std::ostringstream details;
        details.precision(17);
        details<<message<<": actual="<<actual
               <<", expected="<<expected<<", error="<<error;
        throw std::runtime_error(details.str());
    }
}

eft::Vec3 random_vector(std::mt19937_64& random) {
    std::uniform_real_distribution<double> distribution(-0.8,0.8);
    eft::Vec3 result;
    do { result={distribution(random),distribution(random),distribution(random)}; }
    while (hv1::norm(result)<0.1);
    return result;
}

eft::Vec3 cubic_rotate(const eft::Vec3& value) {
    // A proper signed-permutation rotation of the Cartesian FFT grid.
    return eft::Vec3{value.z,value.x,value.y};
}

double f2_formula(const eft::Vec3& left,const eft::Vec3& right) {
    const double k1=hv1::norm(left),k2=hv1::norm(right),mu=eft::cosine(left,right);
    return 5.0/7.0+0.5*mu*(k1/k2+k2/k1)+2.0*mu*mu/7.0;
}

double g2_formula(const eft::Vec3& left,const eft::Vec3& right) {
    const double k1=hv1::norm(left),k2=hv1::norm(right),mu=eft::cosine(left,right);
    return 3.0/7.0+0.5*mu*(k1/k2+k2/k1)+4.0*mu*mu/7.0;
}

class SmoothTransfer final:public PowerSpectrum {
public:
    real Evaluate(real k) const override {
        return k>0.0 ?0.37+3.5*k*k :0.0;
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("SmoothTransfer has no cosmology");
    }
};

void test_primitives() {
    std::mt19937_64 random(8112026);
    for (int sample=0;sample<300;++sample) {
        const eft::Vec3 a=random_vector(random),b=random_vector(random),c=random_vector(random);
        const eft::SptKernelPair second=eft::spt_kernels({a,b});
        require_close(second.F,f2_formula(a,b),1.0e-13,1.0e-13,"F2 analytic");
        require_close(second.G,g2_formula(a,b),1.0e-13,1.0e-13,"G2 analytic");
        const double native_f3=marisa_b_native::F3eds(
            hv1::norm(a),hv1::norm(b),hv1::norm(c),
            eft::cosine(b,c),eft::cosine(a,b),eft::cosine(a,c));
        require_close(eft::spt_F({a,b,c}),native_f3,2.0e-12,2.0e-12,"F3 native oracle");
    }
    require_close(eft::kappa({1,0,0},{0,1,0}),-1.0,0.0,1.0e-15,"kappa orthogonal");
    require_close(eft::kappa({1,0,0},{-2,0,0}),0.0,0.0,1.0e-15,"kappa opposite");
}

void test_bias_k1_k2_and_folpsd() {
    std::mt19937_64 random(9402);
    std::uniform_real_distribution<double> bias(-2.0,3.0);
    for (int sample=0;sample<500;++sample) {
        const eft::Vec3 a=random_vector(random),b=random_vector(random);
        eft::BiasValues values={};
        for (double& value:values) value=bias(random);
        const double K1=eft::evaluate_bias_kernel(eft::deterministic_kernel({a}),values);
        require_close(K1,values[0],0.0,1.0e-15,"K1 hand formula");
        const double manual=values[0]*f2_formula(a,b)+0.5*values[1]
                            +values[2]*eft::kappa(a,b);
        const double K2=eft::evaluate_bias_kernel(eft::deterministic_kernel({a,b}),values);
        require_close(K2,manual,1.0e-13,1.0e-13,"K2 hand formula");

        // FolpsD commit 94ee3bf, folps.py::set_bias_scheme and ::Z2,
        // f=0, classpt scheme: b2_folps=b2-4 bG2/3, bs2=2 bG2.
        const double mu=eft::cosine(a,b);
        const double b2_folps=values[1]-4.0*values[2]/3.0;
        const double bs2_folps=2.0*values[2];
        const double folps=values[0]*f2_formula(a,b)+0.5*b2_folps
                           +0.5*bs2_folps*(mu*mu-1.0/3.0);
        require_close(K2,folps,1.0e-13,1.0e-13,"FolpsD tree K2 oracle");

        const hv1::KernelTemplate old=hv1::pre_reconstruction_kernel({a,b});
        const hv1::BiasPoint old_bias{
            values[0],values[1]-4.0*values[2]/3.0,values[2]};
        require_close(K2,old.regular.evaluate(old_bias),1.0e-13,1.0e-13,"v1 K2 convention adapter");
    }
}

void compare_polynomials(
    const eft::SparsePolynomial& left,
    const eft::SparsePolynomial& right,
    double tolerance,
    const std::string& message) {
    for (const auto& term:left.terms()) {
        require_close(term.second,right.coefficient(term.first),tolerance,tolerance,message);
    }
    /*
     * Algebraically absent terms can survive a different summation order at
     * round-off level.  Check the union of both sparse supports instead of
     * demanding bitwise-identical pruning.
     */
    for (const auto& term:right.terms()) {
        require_close(left.coefficient(term.first),term.second,tolerance,tolerance,message);
    }
}

void test_symmetry_and_slow_reference() {
    std::mt19937_64 random(3114);
    for (int order=1;order<=4;++order) {
        for (int sample=0;sample<30;++sample) {
            std::vector<eft::Vec3> momenta;
            for (int index=0;index<order;++index) momenta.push_back(random_vector(random));
            const eft::SparsePolynomial production=eft::deterministic_kernel(momenta);
            const eft::SparsePolynomial slow=eft::deterministic_kernel_slow(momenta);
            compare_polynomials(production,slow,1.0e-11,"slow/optimized K"+std::to_string(order));
            std::vector<int> indices(order);
            std::iota(indices.begin(),indices.end(),0);
            do {
                std::vector<eft::Vec3> permuted;
                for (int index:indices) permuted.push_back(momenta[index]);
                compare_polynomials(
                    production,eft::deterministic_kernel(permuted),1.0e-12,
                    "K"+std::to_string(order)+" permutation symmetry");
            } while (std::next_permutation(indices.begin(),indices.end()));
        }
    }
}

void test_reconstructed_provider() {
    std::mt19937_64 random(731922);
    const eft::EftBiasKernelProvider eft_base;
    const eft::V1CompatibilityKernelProvider v1_base;

    hv1::ReconstructionConfig disabled;
    disabled.enabled=false;
    const eft::ReconstructedFieldKernelProvider disabled_provider(
        eft_base,disabled);

    hv1::ReconstructionConfig config;
    config.enabled=true;
    config.smoothing_radius=15.0;
    config.bias_recon=2.7340475186190334;
    config.cell_size=8.0;
    const eft::ReconstructedFieldKernelProvider production(
        eft_base,config);
    const eft::ReconstructedFieldKernelProvider v1(v1_base,config);
    const eft::MatterKernelProvider matter_base;
    const eft::ReconstructedFieldKernelProvider matter(
        matter_base,config);

    hv1::ReconstructionConfig infinite=config;
    infinite.smoothing_radius=1.0e12;
    const eft::ReconstructedFieldKernelProvider no_shift(
        eft_base,infinite);
    hv1::ReconstructionConfig infinite_bias=config;
    infinite_bias.bias_recon=1.0e300;
    const eft::ReconstructedFieldKernelProvider no_shift_from_bias(
        eft_base,infinite_bias);

    eft::BiasValues matter_bias={};
    matter_bias[0]=1.0;

    for (int order=1;order<=4;++order) {
        for (int sample=0;sample<12;++sample) {
            std::vector<eft::Vec3> momenta;
            if (order==4) {
                /*
                 * The legacy halo-v1 K4 oracle is deliberately implemented
                 * only on the B411 topology, i.e. with an opposite loop
                 * pair.  The production EFT-v2 kernel is generic, but the
                 * cross-generation regression must stay inside the legacy
                 * oracle's documented domain.
                 */
                const eft::Vec3 a=random_vector(random);
                const eft::Vec3 b=random_vector(random);
                const eft::Vec3 q=random_vector(random);
                momenta={a,b,q,hv1::negate(q)};
            } else {
                for (int index=0;index<order;++index) {
                    momenta.push_back(random_vector(random));
                }
            }
            const eft::SparsePolynomial pre=eft_base.deterministic(momenta);
            compare_polynomials(
                disabled_provider.deterministic(momenta),pre,1.0e-13,
                "disabled reconstructed provider K"+std::to_string(order));
            compare_polynomials(
                no_shift.deterministic(momenta),pre,1.0e-13,
                "R-infinity reconstructed provider K"+std::to_string(order));
            compare_polynomials(
                no_shift_from_bias.deterministic(momenta),pre,1.0e-13,
                "brec-infinity reconstructed provider K"
                +std::to_string(order));

            const eft::SparsePolynomial old=eft::adapt_v1_polynomial(
                hv1::reconstructed_kernel(momenta,config).regular);
            compare_polynomials(
                v1.deterministic(momenta),old,2.0e-12,
                "generic/v1 reconstruction partition K"
                +std::to_string(order));
            /*
             * Orders one through three are bit-level matter oracles.  The
             * legacy halo-v1/native F4edsb backend predates EFT-v2's audited
             * continuous opposite-pair F4 limit, so K4 is retained only as
             * a loose cross-generation diagnostic; the exact radial-limit
             * oracle below is authoritative for K4.
             */
            require_close(
                matter.deterministic(momenta).evaluate({}),
                hv1::reconstructed_kernel(
                    momenta,config).regular.evaluate(
                        hv1::BiasPoint{1.0,0.0,0.0}),
                order==4?1.0e-2:3.0e-11,
                order==4?1.0e-5:3.0e-11,
                "matter/native reconstruction oracle K"
                +std::to_string(order));
            require_close(
                eft::evaluate_bias_kernel(
                    production.deterministic(momenta),
                    matter_bias),
                matter.deterministic(momenta).evaluate({}),
                3.0e-11,3.0e-11,
                "EFT matter submanifold K"
                +std::to_string(order));

            const eft::SparsePolynomial reference=
                production.deterministic(momenta);
            std::vector<int> indices(static_cast<std::size_t>(order));
            std::iota(indices.begin(),indices.end(),0);
            do {
                std::vector<eft::Vec3> permuted;
                for (int index:indices) {
                    permuted.push_back(
                        momenta[static_cast<std::size_t>(index)]);
                }
                compare_polynomials(
                    production.deterministic(permuted),reference,2.0e-11,
                    "reconstructed EFT K"+std::to_string(order)
                    +" permutation symmetry");
            } while (std::next_permutation(
                         indices.begin(),indices.end()));

            std::vector<eft::Vec3> cubic_momenta;
            cubic_momenta.reserve(momenta.size());
            for (const eft::Vec3& value:momenta) {
                cubic_momenta.push_back(cubic_rotate(value));
            }
            compare_polynomials(
                production.deterministic(cubic_momenta),
                reference,3.0e-11,
                "reconstructed EFT cubic-grid rotation K"
                +std::to_string(order));
        }
    }

    const eft::Vec3 one{0.023,-0.017,0.041};
    compare_polynomials(
        production.deterministic({one}),
        eft_base.deterministic({one}),1.0e-14,
        "reconstruction leaves K1 unchanged");
    require(
        production.name()=="reconstructed(eft_v2_cobra_bias)",
        "reconstructed provider provenance name");

    /*
     * Exact hard-pair topology checks.  At qR >> 1 every partition carrying
     * a hard shift is exponentially suppressed.  K3 therefore approaches
     * its pre-reconstruction kernel, while K4 retains precisely the two
     * soft-shift times hard-tadpole partitions.
     */
    const eft::Vec3 k1{0.031,-0.017,0.052};
    const eft::Vec3 k2{-0.044,0.026,0.019};
    const eft::Vec3 q{3.0,-4.0,2.0};
    const eft::Vec3 minus_q=hv1::negate(q);
    compare_polynomials(
        production.deterministic({k1,q,minus_q}),
        eft_base.deterministic({k1,q,minus_q}),
        2.0e-11,
        "hard-pair reconstructed K3 approaches pre K3");

    const eft::Vec3 output=hv1::add(k1,k2);
    const double shift_k1=hv1::reconstruction_shift_factor(
        output,k1,config);
    const double shift_k2=hv1::reconstruction_shift_factor(
        output,k2,config);
    const eft::SparsePolynomial expected_k4=
        eft_base.deterministic({k1,k2,q,minus_q})
        +0.25*(shift_k1+shift_k2)*(
            eft_base.deterministic({k1,q,minus_q})
                *eft_base.deterministic({k2})
            +eft_base.deterministic({k2,q,minus_q})
                *eft_base.deterministic({k1}));
    compare_polynomials(
        production.deterministic({k1,k2,q,minus_q}),
        expected_k4,5.0e-11,
        "hard-pair reconstructed K4 forest identity");
}

void test_local_png_reconstruction_denominator_variation() {
    const SmoothTransfer transfer;
    hv1::ReconstructionConfig fixed;
    fixed.enabled=true;
    fixed.smoothing_radius=15.0;
    fixed.bias_recon=2.7340475186190334;
    fixed.cell_size=8.0;
    const double kmin=0.02;

    const eft::Vec3 output{0.071,-0.033,0.052};
    const eft::Vec3 above{0.029,0.017,-0.011};
    const eft::Vec3 below{0.006,0.001,-0.002};
    const auto shift=
        hv1::reconstruction_shift_factor_local_png_variation(
            output,above,fixed,transfer,kmin);
    const double expected=
        -shift.value/(fixed.bias_recon*transfer(hv1::norm(above)));
    require_close(
        shift.local_png_denominator_direction,
        expected,2.0e-15,2.0e-15,
        "local-PNG shift analytic quotient-rule direction");

    const double epsilon=1.0e-5;
    hv1::ReconstructionConfig plus=fixed;
    plus.local_png_bias_transfer=&transfer;
    plus.local_png_bias_amplitude=epsilon;
    plus.local_png_bias_kmin=kmin;
    hv1::ReconstructionConfig minus=plus;
    minus.local_png_bias_amplitude=-epsilon;
    const double finite_difference=(
        hv1::reconstruction_shift_factor(output,above,plus)
        -hv1::reconstruction_shift_factor(output,above,minus))
        /(2.0*epsilon);
    require_close(
        shift.local_png_denominator_direction,
        finite_difference,2.0e-9,2.0e-11,
        "local-PNG shift analytic/finite-difference closure");
    require_close(
        hv1::reconstruction_bias_denominator(above,plus),
        fixed.bias_recon+epsilon/transfer(hv1::norm(above)),
        2.0e-15,2.0e-15,
        "local-PNG reconstruction denominator definition");

    const auto excluded=
        hv1::reconstruction_shift_factor_local_png_variation(
            output,below,fixed,transfer,kmin);
    require_close(
        excluded.local_png_denominator_direction,
        0.0,0.0,0.0,
        "finite-box cutoff removes absent displacement mode");
    require_close(
        hv1::reconstruction_bias_denominator(below,plus),
        fixed.bias_recon,0.0,0.0,
        "finite-box cutoff retains fixed denominator");

    const eft::EftBiasKernelProvider base;
    std::mt19937_64 random(28072026);
    for (int order=1;order<=4;++order) {
        for (int sample=0;sample<8;++sample) {
            std::vector<eft::Vec3> momenta;
            for (int index=0;index<order;++index) {
                momenta.push_back(random_vector(random));
            }
            const auto analytic=
                eft::
                reconstructed_field_kernel_local_png_denominator_variation(
                    base,fixed,transfer,kmin,momenta);
            const eft::ReconstructedFieldKernelProvider fixed_provider(
                base,fixed);
            compare_polynomials(
                analytic.value,
                fixed_provider.deterministic(momenta),
                3.0e-13,
                "adaptive field-kernel fixed-map value K"
                +std::to_string(order));

            const auto central_difference=
                [&base,&fixed,&transfer,kmin,&momenta](
                    double step) {
                    hv1::ReconstructionConfig positive=fixed;
                    positive.local_png_bias_transfer=&transfer;
                    positive.local_png_bias_amplitude=step;
                    positive.local_png_bias_kmin=kmin;
                    hv1::ReconstructionConfig negative=positive;
                    negative.local_png_bias_amplitude=-step;
                    const eft::ReconstructedFieldKernelProvider
                        positive_provider(base,positive);
                    const eft::ReconstructedFieldKernelProvider
                        negative_provider(base,negative);
                    return (
                        positive_provider.deterministic(momenta)
                        -negative_provider.deterministic(momenta))
                        *(0.5/step);
                };
            const eft::SparsePolynomial coarse=
                central_difference(2.0e-5);
            const eft::SparsePolynomial fine=
                central_difference(1.0e-5);
            const eft::SparsePolynomial richardson=
                (4.0*fine-coarse)*(1.0/3.0);
            compare_polynomials(
                analytic.direction,richardson,2.0e-8,
                "adaptive field-kernel product-rule closure K"
                +std::to_string(order));

            std::vector<eft::Vec3> rotated;
            for (const eft::Vec3& momentum:momenta) {
                rotated.push_back(cubic_rotate(momentum));
            }
            const auto rotated_analytic=
                eft::
                reconstructed_field_kernel_local_png_denominator_variation(
                    base,fixed,transfer,kmin,rotated);
            compare_polynomials(
                analytic.direction,
                rotated_analytic.direction,
                5.0e-11,
                "adaptive field-kernel cubic/CIC rotation K"
                +std::to_string(order));
        }
    }

    hv1::ReconstructionConfig no_shift=fixed;
    no_shift.smoothing_radius=1.0e12;
    const std::vector<eft::Vec3> probe{
        {0.031,-0.017,0.052},
        {-0.044,0.026,0.019},
        {0.11,0.07,0.19}};
    const auto no_shift_variation=
        eft::reconstructed_field_kernel_local_png_denominator_variation(
            base,no_shift,transfer,kmin,probe);
    require(
        no_shift_variation.direction.empty(),
        "R-infinity removes adaptive reconstruction response");

    hv1::ReconstructionConfig disabled=fixed;
    disabled.enabled=false;
    const auto disabled_variation=
        eft::reconstructed_field_kernel_local_png_denominator_variation(
            base,disabled,transfer,kmin,probe);
    require(
        disabled_variation.direction.empty(),
        "disabled reconstruction removes adaptive response");
}

void test_limits_and_matter_oracle() {
    std::mt19937_64 random(411);
    {
        const eft::Vec3 q{0.06,-0.03,0.015};
        const eft::Vec3 minus_q=hv1::negate(q);
        for (const std::vector<eft::Vec3>& momenta:
             std::vector<std::vector<eft::Vec3>>{
                 {q,q,minus_q},{q,minus_q,q},{minus_q,q,q}}) {
            const eft::SptKernelPair exact=
                eft::spt_kernels(momenta);
            require_close(
                exact.F,-1.0/6.0,0.0,2.0e-15,
                "F3 exact q=k collinear opposite-pair limit");
            require_close(
                exact.G,-1.0/6.0,0.0,2.0e-15,
                "G3 exact q=k collinear opposite-pair limit");
            const eft::SparsePolynomial bias=
                eft::deterministic_kernel(momenta);
            for (const auto& term:bias.terms()) {
                require(
                    std::isfinite(term.second),
                    "q=k collinear opposite-pair bias kernel finite");
            }
        }
    }
    for (int sample=0;sample<100;++sample) {
        const eft::Vec3 k=random_vector(random),p=random_vector(random),q=random_vector(random);
        const eft::Vec3 minus_q=hv1::negate(q);
        const eft::SptKernelPair exact_k3=eft::spt_kernels({k,q,minus_q});
        const double epsilon=1.0e-4;
        const eft::Vec3 qminus_low{
            -(1.0-epsilon)*q.x,-(1.0-epsilon)*q.y,-(1.0-epsilon)*q.z};
        const eft::Vec3 qminus_high{
            -(1.0+epsilon)*q.x,-(1.0+epsilon)*q.y,-(1.0+epsilon)*q.z};
        const eft::SptKernelPair low=eft::spt_kernels({k,q,qminus_low});
        const eft::SptKernelPair high=eft::spt_kernels({k,q,qminus_high});
        require_close(exact_k3.F,0.5*(low.F+high.F),2.0e-6,2.0e-8,
                      "analytic F3 opposite-pair continuous limit");
        require_close(exact_k3.G,0.5*(low.G+high.G),2.0e-6,2.0e-8,
                      "analytic G3 opposite-pair continuous limit");
        for (const std::vector<eft::Vec3>& momenta:
             std::vector<std::vector<eft::Vec3>>{{k,q,minus_q},{k,p,q,minus_q}}) {
            const eft::SparsePolynomial kernel=eft::deterministic_kernel(momenta);
            for (const auto& term:kernel.terms()) {
                require(std::isfinite(term.second),"opposite-pair bias kernel finite");
            }
        }
        const double ours=eft::spt_F({k,p,q,minus_q});
        const double fourth_epsilon=std::min(
            0.02,
            0.02*std::min(hv1::norm(k),hv1::norm(p))/hv1::norm(q));
        const auto radial_average=[&k,&p,&q](double epsilon_value) {
            const eft::Vec3 low{
                -(1.0-epsilon_value)*q.x,
                -(1.0-epsilon_value)*q.y,
                -(1.0-epsilon_value)*q.z};
            const eft::Vec3 high{
                -(1.0+epsilon_value)*q.x,
                -(1.0+epsilon_value)*q.y,
                -(1.0+epsilon_value)*q.z};
            return 0.5*(
                eft::spt_F({k,p,q,low})
                +eft::spt_F({k,p,q,high}));
        };
        const double radial_coarse=radial_average(fourth_epsilon);
        const double radial_fine=radial_average(0.5*fourth_epsilon);
        const double radial_finest=radial_average(0.25*fourth_epsilon);
        const double radial_first_coarse=
            (4.0*radial_fine-radial_coarse)/3.0;
        const double radial_first_fine=
            (4.0*radial_finest-radial_fine)/3.0;
        const double radial_continuation=
            (16.0*radial_first_fine-radial_first_coarse)/15.0;
        require_close(
            ours,radial_continuation,3.0e-8,3.0e-11,
            "F4 exact opposite-pair independent radial continuation");
    }
    const hv1::CanonicalTriangle triangle=
        hv1::canonicalize_triangle({0.045,0.067,0.083});
    const std::vector<eft::Vec3> exact_fourth={
        triangle.k1,triangle.k2,{0.11,0.07,0.19},
        {-0.11,-0.07,-0.19}};
    const eft::SptKernelPair exact=eft::spt_kernels(exact_fourth);
    require_close(
        exact.F,-0.0096443820921974354,5.0e-12,2.0e-15,
        "F4 angular-regulator extrapolation reference");
    require_close(
        exact.G,-0.00849968703629008435,5.0e-12,2.0e-15,
        "G4 angular-regulator extrapolation reference");
    std::array<int,4> fourth_order={{0,1,2,3}};
    do {
        std::vector<eft::Vec3> permuted;
        for (int index:fourth_order) {
            permuted.push_back(
                exact_fourth[static_cast<std::size_t>(index)]);
        }
        const eft::SptKernelPair value=eft::spt_kernels(permuted);
        require_close(
            value.F,exact.F,2.0e-12,2.0e-15,
            "F4 opposite-pair permutation symmetry");
        require_close(
            value.G,exact.G,2.0e-12,2.0e-15,
            "G4 opposite-pair permutation symmetry");
    } while (std::next_permutation(
        fourth_order.begin(),fourth_order.end()));
    const eft::Vec3 k{0.13,-0.04,0.08};
    const std::array<eft::Vec3,5> probes={{
        {1.0e-10,0,0}, {0.2,0,0}, {0.4,0,0}, {-0.2,0,0}, {0,0.3,0}}};
    for (const auto& q:probes) {
        for (const auto& momenta:std::vector<std::vector<eft::Vec3>>{{k,q},{k,q,hv1::negate(q)}}) {
            const auto value=eft::deterministic_kernel(momenta);
            for (const auto& term:value.terms()) require(std::isfinite(term.second),"soft/collinear finite");
        }
    }
}

void test_tree_external_permutations() {
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle({0.047,0.081,0.103});
    const std::array<eft::Vec3,3> vectors={{triangle.k1,triangle.k2,triangle.k3}};
    const std::array<double,3> power={{8500,5100,3300}};
    eft::BiasValues bias={};
    bias[0]=2.73; bias[1]=-0.7; bias[2]=-0.4;
    const double reference=eft::tree_bispectrum(vectors,power,bias);
    std::array<int,3> order={{0,1,2}};
    do {
        std::array<eft::Vec3,3> permuted_vectors;
        std::array<double,3> permuted_power;
        for (int index=0;index<3;++index) {
            permuted_vectors[index]=vectors[order[index]];
            permuted_power[index]=power[order[index]];
        }
        require_close(eft::tree_bispectrum(permuted_vectors,permuted_power,bias),reference,
                      1.0e-13,1.0e-8,"tree external-leg permutation");
    } while (std::next_permutation(order.begin(),order.end()));
}

}  // namespace

int main() {
    try {
        test_primitives();
        test_bias_k1_k2_and_folpsd();
        test_symmetry_and_slow_reference();
        test_reconstructed_provider();
        test_local_png_reconstruction_denominator_variation();
        test_limits_and_matter_oracle();
        test_tree_external_permutations();
        std::cout<<"EFT-v2 bias-operator checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 bias-operator test failed: "<<error.what()<<"\n";
        return 1;
    }
}
