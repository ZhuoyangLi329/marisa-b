#include "counterterms.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "halo_v1.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;

namespace {

int checks=0;

void require(bool condition,const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,double expected,double relative,double absolute,
    const std::string& message) {
    ++checks;
    const double error=std::fabs(actual-expected);
    if (error>absolute && error>relative*std::max(std::fabs(actual),std::fabs(expected))) {
        std::ostringstream details;
        details.precision(17);
        details<<message<<": actual="<<actual<<", expected="<<expected;
        throw std::runtime_error(details.str());
    }
}

void require_polynomial_close(
    const eft::SparsePolynomial& actual,const eft::SparsePolynomial& expected,
    const std::string& message,
    double relative=2.0e-13,double absolute=1.0e-9) {
    for (const auto& term:expected.terms()) {
        require_close(
            actual.coefficient(term.first),term.second,
            relative,absolute,message);
    }
    for (const auto& term:actual.terms()) {
        require_close(
            term.second,expected.coefficient(term.first),
            relative,absolute,message);
    }
}

class PerturbedCountertermProvider final:
    public eft::FieldKernelProvider {
public:
    PerturbedCountertermProvider(
        const eft::FieldKernelProvider& base,
        std::size_t direction,double epsilon,double k_nl)
        :base_(base),direction_(direction),
         epsilon_(epsilon),k_nl_(k_nl) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        double coefficient=0.0;
        if (momenta.size()==1) {
            coefficient=eft::counterterm_K1(
                momenta[0],k_nl_)[direction_];
        } else if (momenta.size()==2) {
            coefficient=eft::counterterm_K2(
                momenta[0],momenta[1],k_nl_)[direction_];
        }
        return base_.deterministic(momenta)
               +epsilon_*eft::SparsePolynomial::constant(
                   coefficient);
    }

    std::string_view name() const noexcept override {
        return "perturbed_counterterm_test";
    }

private:
    const eft::FieldKernelProvider& base_;
    std::size_t direction_;
    double epsilon_;
    double k_nl_;
};

eft::SparsePolynomial tree_from_provider(
    const eft::FieldKernelProvider& provider,
    const std::array<eft::Vec3,3>& triangle,
    const std::array<double,3>& power) {
    eft::SparsePolynomial result;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            result+=2.0*power[left]*power[right]
                    *provider.deterministic({triangle[left]})
                    *provider.deterministic({triangle[right]})
                    *provider.deterministic(
                        {triangle[left],triangle[right]});
        }
    }
    return result;
}

void test_kernel_formulas() {
    const eft::Vec3 first{0.031,-0.047,0.019};
    const eft::Vec3 second{-0.012,0.028,0.053};
    constexpr double k_nl=0.30;
    const double inverse_scale=1.0/(k_nl*k_nl);
    const eft::Vec3 total=hv1::add(first,second);
    const double k_square=hv1::dot(total,total);
    const double dot=hv1::dot(first,second);
    const double tidal=eft::kappa(first,second);
    const auto K1=eft::counterterm_K1(first,k_nl);
    require_close(
        K1[0],
        -hv1::dot(first,first)*inverse_scale,0.0,1.0e-15,"K1 counterterm");
    const auto K2=eft::counterterm_K2(first,second,k_nl);
    require_close(
        K2[0],
        -k_square*inverse_scale*eft::spt_F({first,second}),0.0,1.0e-14,
        "K2 nabla2 delta");
    require_close(
        K2[1],
        -k_square*inverse_scale,0.0,1.0e-15,"K2 nabla2 delta2");
    require_close(
        K2[2],
        -k_square*tidal*inverse_scale,0.0,1.0e-15,"K2 nabla2 G2");
    require_close(
        K2[3],
        -dot*inverse_scale,0.0,1.0e-15,"K2 grad delta squared");
    require_close(
        K2[4],
        -dot*(1.0+tidal)*inverse_scale,0.0,1.0e-15,"K2 grad tidal squared");
}

void test_bispectrum_and_permutations() {
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle({0.047,0.071,0.089});
    const std::array<eft::Vec3,3> triangle={{canonical.k1,canonical.k2,canonical.k3}};
    const std::array<double,3> power={{1200.0,800.0,610.0}};
    const auto reference=eft::counterterm_bispectrum(triangle,power);
    for (std::size_t index=0;index<eft::kCountertermCount;++index) {
        require(reference.total[index].serialize()==
                    (reference.BctrI[index]+reference.BctrII[index]).serialize(),
                "counterterm total recomposition");
    }
    std::array<int,3> order={{0,1,2}};
    do {
        std::array<eft::Vec3,3> permuted_triangle;
        std::array<double,3> permuted_power;
        for (int index=0;index<3;++index) {
            permuted_triangle[index]=triangle[order[index]];
            permuted_power[index]=power[order[index]];
        }
        const auto value=eft::counterterm_bispectrum(permuted_triangle,permuted_power);
        for (std::size_t index=0;index<eft::kCountertermCount;++index) {
            require_polynomial_close(
                value.BctrI[index],reference.BctrI[index],"BctrI external permutation");
            require_polynomial_close(
                value.BctrII[index],reference.BctrII[index],"BctrII external permutation");
            require_polynomial_close(
                value.total[index],reference.total[index],"counterterm total external permutation");
        }
    } while (std::next_permutation(order.begin(),order.end()));

    std::array<double,eft::kParameterCount> parameters={};
    parameters[static_cast<std::size_t>(eft::ParameterId::B1)]=2.3;
    parameters[static_cast<std::size_t>(eft::ParameterId::B2)]=0.4;
    parameters[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.2;
    for (eft::ParameterId id:eft::counterterm_parameter_ids()) {
        parameters[static_cast<std::size_t>(id)]=1.0;
        const double all=reference.evaluate(parameters);
        parameters[static_cast<std::size_t>(id)]=0.0;
        require(std::isfinite(all) && std::fabs(all)>0.0,
                "each counterterm produces a finite nonzero shape");
    }
}

void test_reconstructed_counterterms() {
    const hv1::CanonicalTriangle canonical=
        hv1::canonicalize_triangle({0.047,0.071,0.089});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const std::array<double,3> power={{1200.0,800.0,610.0}};
    const eft::EftBiasKernelProvider base;

    hv1::ReconstructionConfig disabled;
    disabled.enabled=false;
    const auto pre=eft::counterterm_bispectrum(
        triangle,power);
    const auto disabled_post=
        eft::reconstructed_counterterm_bispectrum(
            base,disabled,triangle,power);
    for (std::size_t index=0;
         index<eft::kCountertermCount;++index) {
        require_polynomial_close(
            disabled_post.BctrI[index],pre.BctrI[index],
            "disabled reconstructed BctrI");
        require_polynomial_close(
            disabled_post.BctrII[index],pre.BctrII[index],
            "disabled reconstructed BctrII");
        require_polynomial_close(
            disabled_post.total[index],pre.total[index],
            "disabled reconstructed counterterm");
    }

    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    const auto post=
        eft::reconstructed_counterterm_bispectrum(
            base,reconstruction,triangle,power);

    constexpr double epsilon=1.0e-6;
    for (std::size_t index=0;
         index<eft::kCountertermCount;++index) {
        const PerturbedCountertermProvider plus_base(
            base,index,epsilon,0.30);
        const PerturbedCountertermProvider minus_base(
            base,index,-epsilon,0.30);
        const eft::ReconstructedFieldKernelProvider plus(
            plus_base,reconstruction);
        const eft::ReconstructedFieldKernelProvider minus(
            minus_base,reconstruction);
        const eft::SparsePolynomial derivative=
            (0.5/epsilon)*(
                tree_from_provider(plus,triangle,power)
                -tree_from_provider(minus,triangle,power));
        /*
         * The literal field derivative carries the Wick factor two on
         * both kinds of insertion.  Bakx et al. Eq. (36) defines BctrII
         * with unit coefficient for each of its six ordered permutations,
         * hence dBtree/dc = BctrI + 2 BctrII.
         */
        require_polynomial_close(
            derivative,
            post.BctrI[index]+2.0*post.BctrII[index],
            "reconstructed counterterm field finite difference",
            2.0e-9,1.0e-4);
    }

    std::array<int,3> order={{0,1,2}};
    do {
        std::array<eft::Vec3,3> permuted_triangle;
        std::array<double,3> permuted_power;
        for (int index=0;index<3;++index) {
            permuted_triangle[index]=triangle[order[index]];
            permuted_power[index]=power[order[index]];
        }
        const auto permuted=
            eft::reconstructed_counterterm_bispectrum(
                base,reconstruction,
                permuted_triangle,permuted_power);
        for (std::size_t index=0;
             index<eft::kCountertermCount;++index) {
            require_polynomial_close(
                permuted.total[index],post.total[index],
                "reconstructed counterterm external permutation");
        }
    } while (std::next_permutation(
                 order.begin(),order.end()));

    hv1::ReconstructionConfig no_shift=reconstruction;
    no_shift.smoothing_radius=1.0e12;
    const auto infinite=
        eft::reconstructed_counterterm_bispectrum(
            base,no_shift,triangle,power);
    for (std::size_t index=0;
         index<eft::kCountertermCount;++index) {
        require_polynomial_close(
            infinite.total[index],pre.total[index],
            "R-infinity reconstructed counterterm");
    }
}

}  // namespace

int main() {
    try {
        test_kernel_formulas();
        test_bispectrum_and_permutations();
        test_reconstructed_counterterms();
        std::cout<<"EFT-v2 counterterm checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 counterterm test failed: "<<error.what()<<"\n";
        return 1;
    }
}
