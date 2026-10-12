#include "direct_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "halo_v1.h"
#include "marisa_b_native.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace native=marisa_b_native;

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
        details<<message<<": actual="<<actual<<", expected="<<expected<<", error="<<error;
        throw std::runtime_error(details.str());
    }
}

class FlatPower final:public PowerSpectrum {
public:
    real Evaluate(real) const override { return 1.0; }
    const Cosmology& GetCosmology() const override { throw std::logic_error("FlatPower has no cosmology"); }
};

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table");
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0,p=0.0;
            if (row>>k>>p) { log_k_.push_back(std::log(k)); log_p_.push_back(std::log(p)); }
        }
        if (log_k_.size()<2) throw std::runtime_error("power table too short");
    }
    real Evaluate(real k) const override {
        const double x=std::log(std::max<double>(k,std::exp(log_k_.front())));
        std::size_t right=1;
        if (x>=log_k_.back()) right=log_k_.size()-1;
        else if (x>log_k_.front()) right=std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin();
        const std::size_t left=right-1;
        const double t=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(log_p_[left]+t*(log_p_[right]-log_p_[left]));
    }
    const Cosmology& GetCosmology() const override { throw std::logic_error("TabulatedPower has no cosmology"); }
private:
    std::vector<double> log_k_,log_p_;
};

class UnitKernelProvider final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(const std::vector<eft::Vec3>&) const override {
        return eft::SparsePolynomial::constant(1.0);
    }
    std::string_view name() const noexcept override { return "unit_test_kernel"; }
};

class AffineKernelProvider final:public eft::FieldKernelProvider {
public:
    AffineKernelProvider(
        const eft::FieldKernelProvider& base,
        const eft::FieldKernelProvider& direction,
        double amplitude)
        :base_(base),direction_(direction),amplitude_(amplitude) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        return base_.deterministic(momenta)
               +amplitude_*direction_.deterministic(momenta);
    }
    std::string_view name() const noexcept override {
        return "affine_test_kernel";
    }

private:
    const eft::FieldKernelProvider& base_;
    const eft::FieldKernelProvider& direction_;
    double amplitude_;
};

class InverseSquarePower final:public PowerSpectrum {
public:
    real Evaluate(real k) const override { return 1.0/(k*k); }
    const Cosmology& GetCosmology() const override { throw std::logic_error("InverseSquarePower has no cosmology"); }
};

class AsymptoticKernelProvider final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(const std::vector<eft::Vec3>& momenta) const override {
        if (momenta.size()>=3) {
            for (std::size_t left=0;left<momenta.size();++left) {
                for (std::size_t right=left+1;right<momenta.size();++right) {
                    const double scale=std::max({hv1::norm(momenta[left]),hv1::norm(momenta[right]),1.0});
                    if (hv1::norm(hv1::add(momenta[left],momenta[right]))<1.0e-13*scale) {
                        const double q=hv1::norm(momenta[left]);
                        return eft::SparsePolynomial::constant(1.0+0.04/(q*q));
                    }
                }
            }
        }
        return eft::SparsePolynomial::constant(1.0);
    }
    std::string_view name() const noexcept override { return "asymptotic_test_kernel"; }
};

double constant(const eft::SparsePolynomial& value) {
    return value.coefficient(eft::MonomialKey());
}

void require_polynomial_close(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    double relative,
    double absolute,
    const std::string& message) {
    const eft::SparsePolynomial difference=actual-expected;
    double error=0.0;
    double scale=0.0;
    for (const auto& term:difference.terms()) {
        error=std::max(error,std::fabs(term.second));
    }
    for (const auto& term:actual.terms()) {
        scale=std::max(scale,std::fabs(term.second));
    }
    for (const auto& term:expected.terms()) {
        scale=std::max(scale,std::fabs(term.second));
    }
    require(
        error<=absolute ||error<=relative*scale,
        message+": max coefficient error="+std::to_string(error));
}

void test_dr_tree_roles() {
    const FlatPower power;
    const eft::EftBiasKernelProvider base;
    const eft::MatterKernelProvider direction;
    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=10.0;
    reconstruction.bias_recon=1.9326526728794324;
    reconstruction.cell_size=8.0;

    const hv1::CanonicalTriangle triangle=
        hv1::canonicalize_triangle({0.043,0.067,0.086});
    const std::array<eft::Vec3,3> vectors={{
        triangle.k1,triangle.k2,triangle.k3}};

    const auto labelled=[](double k1,double k2) {
        return [=](const std::vector<eft::Vec3>& momenta) {
            return eft::FieldKernelVariation{
                eft::SparsePolynomial::constant(
                    momenta.size()==1?k1:k2),{}};
        };
    };
    const eft::FieldKernelVariation labelled_tree=
        eft::ordered_tree_bispectrum_variation(
            power,vectors,
            std::array<eft::FieldKernelVariationEvaluator,3>{
                labelled(2.0,5.0),
                labelled(3.0,7.0),
                labelled(11.0,13.0)});
    require_close(
        constant(labelled_tree.value),794.0,0.0,1.0e-12,
        "ordered mixed-tree assembler preserves external labels");
    const std::vector<std::vector<eft::Vec3>> probes={
        {vectors[0]},
        {vectors[0],vectors[1]}};
    for (const auto& momenta:probes) {
        const eft::FieldKernelVariation tied=
            eft::reconstructed_field_kernel_variation_with_shift(
                base,direction,base,direction,
                reconstruction,momenta);
        const eft::FieldKernelVariation random=
            eft::shifted_random_field_kernel_variation(
                base,direction,reconstruction,momenta);
        const eft::FieldKernelVariation displaced=
            eft::displaced_data_field_kernel_variation(
                base,direction,base,direction,
                reconstruction,momenta);
        require_polynomial_close(
            displaced.value-random.value,tied.value,
            3.0e-13,3.0e-13,
            "D-R=F Gaussian kernel closure");
        require_polynomial_close(
            displaced.direction-random.direction,tied.direction,
            3.0e-13,3.0e-13,
            "D-R=F response kernel closure");
    }

    const eft::ShiftedRandomFieldKernelProvider random_provider(
        base,reconstruction);
    const eft::DisplacedDataFieldKernelProvider displaced_provider(
        base,base,reconstruction);
    const eft::ReconstructedFieldKernelProvider tied_provider(
        base,reconstruction);
    for (const auto& momenta:probes) {
        require_polynomial_close(
            displaced_provider.deterministic(momenta)
                -random_provider.deterministic(momenta),
            tied_provider.deterministic(momenta),
            3.0e-13,3.0e-13,
            "D/R provider Gaussian closure");
    }

    const eft::FieldKernelVariationEvaluator displaced=[&](
        const std::vector<eft::Vec3>& momenta) {
        return eft::displaced_data_field_kernel_variation(
            base,direction,base,direction,
            reconstruction,momenta);
    };
    const eft::FieldKernelVariationEvaluator random=[&](
        const std::vector<eft::Vec3>& momenta) {
        return eft::shifted_random_field_kernel_variation(
            base,direction,reconstruction,momenta);
    };
    const eft::FieldKernelVariationEvaluator tied=[&](
        const std::vector<eft::Vec3>& momenta) {
        return eft::reconstructed_field_kernel_variation_with_shift(
            base,direction,base,direction,
            reconstruction,momenta);
    };
    const eft::SignedDrTreeVariations roles=
        eft::signed_dr_tree_variations(
            power,vectors,displaced,random);
    const eft::FieldKernelVariation tied_tree=
        eft::ordered_tree_bispectrum_variation(
            power,vectors,
            std::array<eft::FieldKernelVariationEvaluator,3>{
                tied,tied,tied});
    const eft::DiagramAssembler historical_tied(
        power,vectors,tied_provider);
    require_polynomial_close(
        tied_tree.value,historical_tied.tree(),
        3.0e-13,3.0e-13,
        "ordered mixed-tree tied limit reproduces production assembler");
    require_polynomial_close(
        roles.total.value,tied_tree.value,
        8.0e-13,8.0e-13,
        "signed DDD/DDR/DRR/RRR Gaussian closure");
    require_polynomial_close(
        roles.total.direction,tied_tree.direction,
        8.0e-13,8.0e-13,
        "signed DDD/DDR/DRR/RRR response closure");

    /*
     * The physical local-PNG K1/K2 direction used by the shell driver must
     * reproduce the independent halo-v1 native tree oracle, including the
     * p=1.12 bphi and deliberately preserved p=1 bphidelta policies.
     */
    const eft::LocalPngHaloTreeDirectionProvider png_direction(
        power,1.12,1.0);
    const eft::FieldKernelVariationEvaluator png_tied=[&](
        const std::vector<eft::Vec3>& momenta) {
        return eft::reconstructed_field_kernel_variation_with_shift(
            base,png_direction,base,png_direction,
            reconstruction,momenta);
    };
    const eft::FieldKernelVariation png_tree=
        eft::ordered_tree_bispectrum_variation(
            power,vectors,
            std::array<eft::FieldKernelVariationEvaluator,3>{
                png_tied,png_tied,png_tied});
    std::array<double,eft::kParameterCount> parameter_values{};
    const double b1=1.93;
    const double b2_eft=-0.61;
    const double gamma2=-0.43;
    parameter_values[static_cast<std::size_t>(eft::ParameterId::B1)]=b1;
    parameter_values[static_cast<std::size_t>(eft::ParameterId::B2)]=b2_eft;
    parameter_values[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=gamma2;
    native::HaloBiasV1Params bias;
    bias.b1=b1;
    bias.b2=b2_eft-4.0*gamma2/3.0;
    bias.bK2=gamma2;
    constexpr double delta_c=1.686;
    bias.bphi=2.0*delta_c*(b1-1.12);
    const double bphi_p1=2.0*delta_c*(b1-1.0);
    bias.bphidelta=bphi_p1
        +2.0*(delta_c*(bias.b2-8.0*(b1-1.0)/21.0)-b1+1.0);
    native::NativeConfig native_config;
    native_config.smoothing_radius=reconstruction.smoothing_radius;
    native_config.bias_recon=reconstruction.bias_recon;
    native_config.recon_cellsize=reconstruction.cell_size;
    native_config.recon_cic_window_power=4;
    native::ClosedTriangleVectors native_vectors{};
    for (std::size_t leg=0;leg<3;++leg) {
        native_vectors[leg]={{
            vectors[leg].x,vectors[leg].y,vectors[leg].z}};
    }
    const native::ComponentResult native_tree=
        native::compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,power,native_vectors,native_config,bias);
    require_close(
        png_tree.direction.evaluate(parameter_values),
        native_tree.dBdfNL_local_tree,
        3.0e-12,3.0e-10,
        "D/R physical local-PNG tied tree/native oracle");

    /*
     * Delta=2 provider 使用独立数值 bPsi/bPsiDelta/bPsiS，并把 exact B111
     * 留给 primordial contraction。这里比较 EFT-v2 reconstruction product
     * rule 与 native halo-tree oracle 扣除 B111 后的全部 deterministic response。
     */
    const double bPsi=-0.016;
    const double bPsiDelta=0.031;
    const double bPsiS=-0.024;
    const double k_pivot=0.05;
    const eft::Delta2PngHaloTreeDirectionProvider delta2_direction(
        power,bPsi,bPsiDelta,bPsiS,k_pivot);
    const eft::FieldKernelVariationEvaluator delta2_tied=[&](
        const std::vector<eft::Vec3>& momenta) {
        return eft::reconstructed_field_kernel_variation_with_shift(
            base,delta2_direction,base,delta2_direction,
            reconstruction,momenta);
    };
    const eft::FieldKernelVariation delta2_tree=
        eft::ordered_tree_bispectrum_variation(
            power,vectors,
            std::array<eft::FieldKernelVariationEvaluator,3>{
                delta2_tied,delta2_tied,delta2_tied});
    native::HaloDelta2PNGBiasParams delta2_bias;
    delta2_bias.b1=b1;
    delta2_bias.b2=bias.b2;
    delta2_bias.bK2=bias.bK2;
    delta2_bias.bPsi=bPsi;
    delta2_bias.bPsiDelta=bPsiDelta;
    delta2_bias.bPsiS=bPsiS;
    delta2_bias.k_pivot=k_pivot;
    const native::ComponentResult delta2_native=
        native::compute_post_recon_halo_delta2_png_tree_dfNL_vectors(
            native::PrimordialBispectrumShape::Equilateral,
            power,power,native_vectors,native_config,delta2_bias);
    require_close(
        delta2_tree.direction.evaluate(parameter_values),
        delta2_native.dBdfNL_halo_png_tree
            -delta2_native.dBdfNL_halo_png_primordial,
        3.0e-12,3.0e-10,
        "D/R Delta=2 tied tree/native non-primordial oracle");

    const std::array<eft::Delta2PngTreeDirectionSector,4> sectors={{
        eft::Delta2PngTreeDirectionSector::ExternalBPsi,
        eft::Delta2PngTreeDirectionSector::BPsiAdvection,
        eft::Delta2PngTreeDirectionSector::BPsiDelta,
        eft::Delta2PngTreeDirectionSector::BPsiS}};
    for (const auto& momenta:probes) {
        eft::SparsePolynomial sector_sum;
        for (const auto sector:sectors) {
            const eft::Delta2PngHaloTreeDirectionProvider component(
                power,bPsi,bPsiDelta,bPsiS,k_pivot,sector);
            sector_sum+=component.deterministic(momenta);
        }
        require_polynomial_close(
            sector_sum,delta2_direction.deterministic(momenta),
            2.0e-15,2.0e-15,
            "Delta=2 provider sector sum");
    }

    constexpr double step=1.0e-5;
    const AffineKernelProvider plus(base,direction,step);
    const AffineKernelProvider minus(base,direction,-step);
    const auto finite_roles=[&](
        const eft::FieldKernelProvider& provider) {
        const eft::DisplacedDataFieldKernelProvider D(
            provider,provider,reconstruction);
        const eft::ShiftedRandomFieldKernelProvider R(
            provider,reconstruction);
        const eft::FieldKernelVariationEvaluator d=[&](
            const std::vector<eft::Vec3>& momenta) {
            return eft::FieldKernelVariation{
                D.deterministic(momenta),{}};
        };
        const eft::FieldKernelVariationEvaluator r=[&](
            const std::vector<eft::Vec3>& momenta) {
            return eft::FieldKernelVariation{
                R.deterministic(momenta),{}};
        };
        return eft::signed_dr_tree_variations(
            power,vectors,d,r).total.value;
    };
    const eft::SparsePolynomial finite_difference=
        (0.5/step)*(finite_roles(plus)-finite_roles(minus));
    require_polynomial_close(
        finite_difference,roles.total.direction,
        3.0e-8,3.0e-8,
        "D/R tree analytic response central-difference closure");

    hv1::ReconstructionConfig disabled=reconstruction;
    disabled.enabled=false;
    const eft::ShiftedRandomFieldKernelProvider random_disabled(
        base,disabled);
    const eft::DisplacedDataFieldKernelProvider displaced_disabled(
        base,base,disabled);
    require(random_disabled.deterministic({vectors[0]}).empty(),
            "shift-off R1 is zero");
    require_polynomial_close(
        displaced_disabled.deterministic({vectors[0],vectors[1]}),
        base.deterministic({vectors[0],vectors[1]}),
        0.0,0.0,"shift-off D equals pre field");
}

void test_wick_coefficients() {
    const FlatPower power;
    const UnitKernelProvider provider;
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle({0.05,0.07,0.09});
    const std::array<eft::Vec3,3> vectors={{triangle.k1,triangle.k2,triangle.k3}};
    const eft::DiagramAssembler assembler(power,vectors,provider);
    const eft::Vec3 q{0.013,-0.021,0.037};
    require_close(constant(assembler.tree()),6.0,0.0,1.0e-14,"tree Wick coefficient");
    require_close(constant(assembler.integrand(eft::Diagram::B222,q)),8.0,0.0,1.0e-14,"B222 Wick coefficient");
    require_close(constant(assembler.integrand(eft::Diagram::B321I,q)),36.0,0.0,1.0e-14,"B321I Wick coefficient");
    require_close(constant(assembler.integrand(eft::Diagram::B321II,q)),36.0,0.0,1.0e-14,"B321II Wick coefficient");
    require_close(constant(assembler.integrand(eft::Diagram::B411,q)),36.0,0.0,1.0e-14,"B411 Wick coefficient");
    require(assembler.soft_centers(eft::Diagram::B222).size()==3,"B222 soft center count");
    require(assembler.soft_centers(eft::Diagram::B321I).size()==4,"B321I soft center count");
    require(assembler.soft_centers(eft::Diagram::B321II).size()==1,"B321II soft center count");
}

void test_ir_mapping_integral() {
    const FlatPower power;
    const UnitKernelProvider provider;
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle({0.04,0.06,0.08});
    const std::array<eft::Vec3,3> vectors={{triangle.k1,triangle.k2,triangle.k3}};
    eft::DirectIntegrationConfig bare;
    bare.qmin=1.0e-5; bare.qmax=0.24;
    bare.n_radial=8; bare.n_mu=12; bare.n_phi=12; bare.ir_safe=false;
    eft::DirectIntegrationConfig mapped=bare;
    mapped.n_radial=32; mapped.n_mu=64; mapped.n_phi=64; mapped.ir_safe=true;
    const eft::DiagramTemplates direct=eft::evaluate_direct(power,vectors,provider,bare);
    eft::DirectIntegrationConfig reflected=bare;
    reflected.exploit_phi_reflection=true;
    const eft::DiagramTemplates reflected_direct=
        eft::evaluate_direct(power,vectors,provider,reflected);
    eft::DirectIntegrationConfig fibonacci=bare;
    fibonacci.angular_rule=eft::DirectAngularRule::AntipodalFibonacci;
    const eft::DiagramTemplates fibonacci_direct=
        eft::evaluate_direct(power,vectors,provider,fibonacci);
    eft::DirectIntegrationConfig convolution=bare;
    convolution.diagram_mask=eft::kDirectB222|eft::kDirectB321I;
    const eft::DiagramTemplates convolution_direct=
        eft::evaluate_direct(
            power,vectors,provider,convolution);
    eft::DirectIntegrationConfig tadpoles=bare;
    tadpoles.diagram_mask=
        eft::kDirectB321II|eft::kDirectB411;
    const eft::DiagramTemplates tadpole_direct=
        eft::evaluate_direct(
            power,vectors,provider,tadpoles);
    const eft::DiagramTemplates safe=eft::evaluate_direct(power,vectors,provider,mapped);
    const double volume=4.0*std::acos(-1.0)*(
        std::pow(bare.qmax,3)-std::pow(bare.qmin,3))/3.0
        /std::pow(2.0*std::acos(-1.0),3);
    require_close(constant(direct.B222),8.0*volume,2.0e-6,1.0e-12,"bare constant B222 integral");
    require_close(constant(direct.B321I),36.0*volume,2.0e-6,1.0e-12,"bare constant B321I integral");
    require_close(constant(reflected_direct.B222),constant(direct.B222),
                  2.0e-13,1.0e-13,"phi reflection reduction preserves B222 measure");
    require_close(constant(reflected_direct.B321I),constant(direct.B321I),
                  2.0e-13,1.0e-13,"phi reflection reduction preserves B321I measure");
    require_close(constant(fibonacci_direct.B222),constant(direct.B222),
                  2.0e-13,1.0e-13,
                  "Fibonacci angular rule preserves constant B222 measure");
    require_close(constant(fibonacci_direct.B321I),constant(direct.B321I),
                  2.0e-13,1.0e-13,
                  "Fibonacci angular rule preserves constant B321I measure");
    require_close(
        constant(convolution_direct.one_loop
                 +tadpole_direct.one_loop),
        constant(direct.one_loop),2.0e-13,1.0e-13,
        "direct diagram masks exactly partition the loop");
    require_close(
        constant(convolution_direct.B321II
                 +convolution_direct.B411),
        0.0,0.0,0.0,
        "convolution mask excludes both tadpoles");
    require_close(
        constant(tadpole_direct.B222
                 +tadpole_direct.B321I),
        0.0,0.0,0.0,
        "tadpole mask excludes both convolutions");
    require_close(constant(safe.B222),constant(direct.B222),1.5e-2,2.0e-6,"IR map preserves B222 integral");
    require_close(constant(safe.B321I),constant(direct.B321I),1.5e-2,2.0e-6,"IR map preserves B321I integral");
    require_close(constant(safe.B321II),constant(direct.B321II),2.0e-6,1.0e-12,"IR map preserves origin-only B321II");
    require_close(constant(safe.B411),constant(direct.B411),2.0e-6,1.0e-12,"IR map preserves origin-only B411");
    eft::DirectIntegrationConfig empty=bare;
    empty.diagram_mask=0;
    bool empty_mask_threw=false;
    try {
        static_cast<void>(
            eft::evaluate_direct(
                power,vectors,provider,empty));
    } catch (const std::invalid_argument&) {
        empty_mask_threw=true;
    }
    require(
        empty_mask_threw,
        "direct evaluator rejects an empty diagram mask");
}

void test_uv_subtraction() {
    const FlatPower flat;
    const UnitKernelProvider unit;
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle({0.04,0.06,0.08});
    const std::array<eft::Vec3,3> vectors={{triangle.k1,triangle.k2,triangle.k3}};
    const eft::DiagramAssembler assembler(flat,vectors,unit);
    const eft::UvSubtraction subtraction(assembler,eft::UvSubtractionConfig{});
    const eft::Vec3 direction{0.36,-0.48,0.8};
    const auto coefficients=subtraction.asymptotic_coefficients(direction);
    require_close(constant(coefficients.B321II),36.0,1.0e-13,1.0e-13,
                  "B321II unit-kernel UV coefficient");
    require_close(constant(coefficients.B411),36.0,1.0e-13,1.0e-13,
                  "B411 unit-kernel UV coefficient");
    const eft::Vec3 loop{0.13,-0.07,0.21};
    require_close(
        constant(assembler.integrand(eft::Diagram::B321II,loop)
                 -subtraction.integrands(loop).B321II),0.0,0.0,1.0e-12,
        "unit B321II fully removed as scaleless tadpole");
    require_close(
        constant(assembler.integrand(eft::Diagram::B411,loop)
                 -subtraction.integrands(loop).B411),0.0,0.0,1.0e-12,
        "unit B411 fully removed as scaleless tadpole");

    eft::DirectIntegrationConfig mapped_config;
    mapped_config.qmin=1.0e-4; mapped_config.qmax=0.24;
    mapped_config.n_radial=8; mapped_config.n_mu=10; mapped_config.n_phi=10;
    mapped_config.ir_safe=true; mapped_config.uv_subtract=true;
    const auto mapped_value=eft::evaluate_direct(flat,vectors,unit,mapped_config);
    require_close(constant(mapped_value.B321II),0.0,0.0,1.0e-12,
                  "IR-mapped B321II subtraction has identical support");
    require_close(constant(mapped_value.B411),0.0,0.0,1.0e-12,
                  "IR-mapped B411 subtraction has identical support");

    const InverseSquarePower power;
    const AsymptoticKernelProvider provider;
    std::array<double,3> bare321={},ren321={},bare411={},ren411={};
    const std::array<double,3> qmax={{1.0,2.0,4.0}};
    for (std::size_t index=0;index<qmax.size();++index) {
        eft::DirectIntegrationConfig config;
        config.qmin=0.05; config.qmax=qmax[index];
        config.n_radial=16; config.n_mu=6; config.n_phi=6;
        config.ir_safe=false; config.uv_subtract=true;
        const auto value=eft::evaluate_direct(power,vectors,provider,config);
        bare321[index]=constant(value.B321II_bare);
        ren321[index]=constant(value.B321II);
        bare411[index]=constant(value.B411_bare);
        ren411[index]=constant(value.B411);
        require_close(
            constant(value.B321II+value.B321II_uv_subtraction),bare321[index],
            1.0e-12,1.0e-12,"B321II bare=subtracted+renormalized");
        require_close(
            constant(value.B411+value.B411_uv_subtraction),bare411[index],
            1.0e-12,1.0e-12,"B411 bare=subtracted+renormalized");
    }
    const double bare_running=std::fabs(bare321[2]-bare321[1]);
    const double ren_running=std::fabs(ren321[2]-ren321[1]);
    require(bare_running>20.0*ren_running,"B321II subtraction removes leading qmax running");
    require(std::fabs(bare411[2]-bare411[1])>20.0*std::fabs(ren411[2]-ren411[1]),
            "B411 subtraction removes leading qmax running");
}

void test_dm_regression(const std::string& power_path) {
    const TabulatedPower power(power_path);
    // The exact EFT-v2 matter provider intentionally uses the independently
    // validated opposite-pair K4 continuation.  Historical DM recovery is
    // instead the responsibility of the explicit frozen-v1 compatibility
    // provider.
    const eft::V1CompatibilityKernelProvider provider;
    const hv1::Triangle sides{0.045,0.067,0.083};
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle(sides);
    const std::array<eft::Vec3,3> vectors={{triangle.k1,triangle.k2,triangle.k3}};
    eft::DirectIntegrationConfig config;
    config.qmin=1.0e-3; config.qmax=0.35;
    config.n_radial=10; config.n_mu=20; config.n_phi=12; config.ir_safe=false;
    const eft::DiagramTemplates current=eft::evaluate_direct(power,vectors,provider,config);
    hv1::IntegrationConfig old_config;
    old_config.qmin=config.qmin; old_config.qmax=config.qmax;
    old_config.n_radial=config.n_radial; old_config.n_mu=config.n_mu;
    old_config.n_phi=config.n_phi; old_config.radial_coordinate=hv1::RadialCoordinate::Logarithmic;
    const hv1::ComponentValues old=hv1::compute_direct(power,sides,old_config,{1.0,0.0,0.0});
    std::array<double,eft::kParameterCount> matter={};
    matter[static_cast<std::size_t>(eft::ParameterId::B1)]=1.0;
    require_close(current.tree.evaluate(matter),old.tree,
                  2.0e-12,1.0e-5*std::max(1.0,std::fabs(old.tree)),
                  "frozen DM tree regression");
    require_close(current.B222.evaluate(matter),old.B222,
                  2.0e-11,1.0e-5*std::max(1.0,std::fabs(old.B222)),
                  "frozen DM B222 regression");
    require_close(current.B321I.evaluate(matter),old.B321I,
                  2.0e-11,1.0e-5*std::max(1.0,std::fabs(old.B321I)),
                  "frozen DM B321I regression");
    require_close(current.B411.evaluate(matter),old.B411,
                  2.0e-11,1.0e-5*std::max(1.0,std::fabs(old.B411)),
                  "frozen DM B411 regression");
    require_close(current.B321II.evaluate(matter),old.B321II,
                  2.0e-3,1.0e-5*std::max(1.0,std::fabs(old.B321II)),
                  "frozen DM B321II/P13 regression");
}

void test_provider_and_external_permutations() {
    const FlatPower power;
    const eft::EftBiasKernelProvider provider;
    const hv1::CanonicalTriangle triangle=hv1::canonicalize_triangle({0.051,0.073,0.097});
    const std::array<eft::Vec3,3> base={{triangle.k1,triangle.k2,triangle.k3}};
    const eft::Vec3 q{0.031,-0.022,0.017};
    const eft::Vec3 reflected_q{q.x,-q.y,q.z};
    const eft::DiagramAssembler reference(power,base,provider);
    std::array<int,3> order={{0,1,2}};
    do {
        std::array<eft::Vec3,3> permuted;
        for (int index=0;index<3;++index) permuted[index]=base[order[index]];
        const eft::DiagramAssembler current(power,permuted,provider);
        require(current.tree().serialize()==reference.tree().serialize(),"tree canonical permutation");
        for (eft::Diagram diagram:{eft::Diagram::B222,eft::Diagram::B321I,eft::Diagram::B321II,eft::Diagram::B411}) {
            const auto left=current.integrand(diagram,q);
            const auto right=reference.integrand(diagram,q);
            for (const auto& term:left.terms()) {
                require_close(term.second,right.coefficient(term.first),1.0e-10,1.0e-10,
                              "diagram external permutation");
            }
            const auto mirror=reference.integrand(diagram,reflected_q);
            for (const auto& term:right.terms()) {
                require_close(term.second,mirror.coefficient(term.first),
                              2.0e-12,2.0e-12,
                              "real-space diagram phi-reflection symmetry");
            }
        }
    } while (std::next_permutation(order.begin(),order.end()));

    const eft::V1CompatibilityKernelProvider compatibility;
    const std::vector<eft::Vec3> momenta={base[0],q,base[1]};
    const auto converted=compatibility.deterministic(momenta);
    const auto old=hv1::pre_reconstruction_kernel(momenta).regular;
    const hv1::BiasPoint old_bias{2.1,-0.7,0.23};
    std::array<double,eft::kParameterCount> values={};
    values[0]=old_bias.b1; values[1]=old_bias.b2+4.0*old_bias.bK2/3.0; values[2]=old_bias.bK2;
    require_close(converted.evaluate(values),old.evaluate(old_bias),1.0e-12,1.0e-12,
                  "v1 provider compatibility");
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("usage: test_diagrams POWER_TABLE");
        test_wick_coefficients();
        test_dr_tree_roles();
        test_ir_mapping_integral();
        test_uv_subtraction();
        test_dm_regression(argv[1]);
        test_provider_and_external_permutations();
        std::cout<<"EFT-v2 diagram checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 diagram test failed: "<<error.what()<<"\n";
        return 1;
    }
}
