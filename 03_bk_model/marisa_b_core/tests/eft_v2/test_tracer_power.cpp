#include "tracer_power.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "PowerSpectrum.h"
#include "direct_evaluator.h"

namespace eft=marisa_b_eft_v2;
namespace shell=marisa_b_shell_v1;

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
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    double relative,double absolute,
    const std::string& message) {
    for (const auto& term:expected.terms()) {
        require_close(
            actual.coefficient(term.first),term.second,
            relative,absolute,
            message+" "+term.first.canonical_string());
    }
    for (const auto& term:actual.terms()) {
        require_close(
            term.second,expected.coefficient(term.first),
            relative,absolute,
            message+" reverse "+term.first.canonical_string());
    }
}

class GaussianPower final:public PowerSpectrum {
public:
    explicit GaussianPower(double scale=1.0):scale_(scale) {}
    real Evaluate(real k) const override { return scale_*1000.0*std::exp(-k*k); }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("GaussianPower has no cosmology");
    }
private:
    double scale_;
};

std::array<double,eft::kParameterCount> matter() {
    std::array<double,eft::kParameterCount> values={};
    values[static_cast<std::size_t>(eft::ParameterId::B1)]=1.0;
    return values;
}

void test_components_and_ir_power_counting() {
    const GaussianPower loop;
    const GaussianPower tree(2.0);
    const eft::EftBiasKernelProvider provider;
    eft::TracerPowerIntegrationConfig config;
    config.qmin=1.0e-3;
    config.qmax=0.8;
    config.n_radial=5;
    config.n_mu=10;
    const double k=0.071;
    const auto result=eft::tracer_power_templates(loop,tree,k,provider,config);
    const auto values=matter();
    require_close(result.tree.evaluate(values),tree(k),2.0e-14,1.0e-10,
                  "tree tracer power uses NLO-matched tree spectrum");
    require_close(
        result.one_loop.evaluate(values),
        result.P22.evaluate(values)+result.P13.evaluate(values),
        2.0e-14,1.0e-10,"power loop recomposition");
    require_close(
        result.P13.evaluate(values),
        (result.P13_bare-result.P13_bias_subtraction
         +result.P13_uv_tail_restoration).evaluate(values),
        2.0e-13,1.0e-7,
        "power P13 bare-minus-subtraction-plus-tail decomposition");
    require_close(
        result.spt.evaluate(values),
        result.tree.evaluate(values)+result.one_loop.evaluate(values),
        2.0e-14,1.0e-10,"power SPT recomposition");
    require_close(result.P22_stochastic_subtraction.evaluate(values),0.0,0.0,1.0e-10,
                  "matter P22 has no constant stochastic subtraction");
    require_close(
        result.counterterm_bnabla2_delta.evaluate(values),
        -2.0*k*k/(0.30*0.30)*loop(k),2.0e-14,1.0e-10,
        "power counterterm uses loop-order spectrum");
    require_close(result.stochastic_pshot,1.0,0.0,0.0,"Pshot raw shape");
    require_close(result.stochastic_a0,k*k/(0.30*0.30),1.0e-15,1.0e-15,
                  "a0 raw shape");
    require(result.integration_nodes>0,"power integration node count");
    const auto p13=eft::renormalized_tracer_p13_template(
        loop,k,provider,config);
    require_polynomial_close(
        p13.value,result.P13,2.0e-14,1.0e-10,
        "specialized P13 equals joint power P13");
    require_polynomial_close(
        p13.bare,result.P13_bare,2.0e-14,1.0e-10,
        "specialized bare P13 equals joint power P13");
    require_polynomial_close(
        p13.bias_subtraction,result.P13_bias_subtraction,
        2.0e-14,1.0e-10,
        "specialized P13 subtraction equals joint power P13");
    require_polynomial_close(
        p13.uv_tail_restoration,result.P13_uv_tail_restoration,
        2.0e-14,1.0e-10,
        "specialized P13 UV tail equals joint power P13");
    require(
        std::fabs(
            p13.uv_tail_restoration.evaluate(values))>1.0,
        "specialized P13 UV-tail test has discriminatory power");
    require(
        p13.integration_nodes==result.integration_nodes,
        "specialized P13 uses the joint power quadrature nodes");

    auto biased=values;
    biased[static_cast<std::size_t>(eft::ParameterId::B2)]=1.0;
    require(result.P22_stochastic_subtraction.evaluate(biased)>0.0,
            "quadratic bias produces the expected positive zero-lag subtraction");
}

void test_fft_lattice_shell_average() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    eft::TracerPowerIntegrationConfig integration;
    integration.qmin=1.0e-3;
    integration.qmax=0.8;
    integration.n_radial=3;
    integration.n_mu=6;
    shell::ShellQuadratureConfig radial;
    radial.n_radial=2;
    radial.radial_measure=shell::ShellRadialMeasure::FftLattice;
    radial.fft_box_size=1000.0;
    radial.fft_mesh_size=256;
    const auto nodes=shell::make_radial_shell_nodes(0.035,0.045,radial);
    double weight_sum=0.0;
    for (const auto& node:nodes) weight_sum+=node.weight;
    require_close(weight_sum,1.0,0.0,2.0e-14,"FFT-lattice radial weights normalize");
    const auto result=eft::tracer_power_shell_templates(
        power,power,0.035,0.045,provider,integration,radial);
    require_close(result.stochastic_pshot,1.0,0.0,2.0e-14,
                  "shell-averaged constant Pshot shape");
    require(result.k>0.035 && result.k<0.045,"shell effective k lies in bin");
    require(result.shell_nodes==nodes.size(),
            "power shell reports its actual radial-node count");

    const float fft_fundamental=static_cast<float>(
        2.0*std::acos(-1.0)/static_cast<float>(1000.0));
    const double fundamental=
        static_cast<double>(fft_fundamental);
    shell::ShellQuadratureConfig narrow=radial;
    narrow.n_radial=4;
    bool strict_failed=false;
    try {
        (void)shell::make_radial_shell_nodes(
            fundamental,2.0*fundamental,narrow);
    } catch (const std::invalid_argument&) {
        strict_failed=true;
    }
    require(
        strict_failed,
        "strict FFT-lattice rule rejects order above unique support");
    narrow.cap_fft_lattice_radial_order_to_support=true;
    const auto capped_rule=shell::make_radial_shell_rule(
        fundamental,2.0*fundamental,narrow);
    const auto& capped=capped_rule.nodes;
    require(capped.size()==3,
            "capped FFT-lattice rule uses all three unique radii");
    require(
        capped_rule.requested_radial_order==4
        &&capped_rule.actual_radial_order==3,
        "capped FFT-lattice rule reports requested and actual orders");
    require(
        capped_rule.lattice_unique_radius_count==3
        &&capped_rule.lattice_mode_count==26,
        "capped FFT-lattice rule reports exact support counts");
    double capped_weight_sum=0.0;
    double capped_effective_k=0.0;
    for (const auto& node:capped) {
        capped_weight_sum+=node.weight;
        capped_effective_k+=node.weight*node.k;
    }
    require_close(
        capped_weight_sum,1.0,0.0,2.0e-14,
        "capped FFT-lattice radial weights normalize");
    const float squared=fft_fundamental*fft_fundamental;
    const double exact_effective_k=
        (6.0*fft_fundamental
         +12.0*std::sqrt(2.0F*squared)
         +8.0*std::sqrt(3.0F*squared))
        /26.0;
    require_close(
        capped_effective_k,exact_effective_k,
        0.0,2.0e-15,
        "capped first P0 shell reproduces direct mode average");
    const auto capped_result=eft::tracer_power_shell_templates(
        power,power,fundamental,2.0*fundamental,
        provider,integration,narrow);
    require(
        capped_result.shell_nodes==3
        &&capped_result.shell_unique_radius_count==3
        &&capped_result.shell_mode_count==26,
        "power projector records capped first-shell geometry");

    const std::array<std::uint64_t,47>
        authoritative_mode_counts{{
            26,66,158,234,410,470,738,866,1170,1358,
            1626,1970,2366,2538,3074,3306,3926,4242,
            4826,5270,5754,6338,7014,7370,8330,8754,
            9710,9858,11162,11358,12770,13138,14218,
            14894,15858,16706,17550,18626,19682,20382,
            21794,22602,24050,24926,25962,26810,28494}};
    std::array<std::int64_t,47>
        expected_boundary_adjustments{};
    expected_boundary_adjustments[31]=32;
    expected_boundary_adjustments[32]=-32;
    expected_boundary_adjustments[43]=80;
    expected_boundary_adjustments[44]=-80;
    shell::ShellQuadratureConfig constrained=radial;
    constrained.n_radial=4;
    constrained.cap_fft_lattice_radial_order_to_support=true;
    constrained.fft_lattice_binning=
        shell::ShellFftLatticeBinning::
            EstimatorModeCountConstrainedIntegerRadius;
    const double exact_fundamental=
        2.0*std::acos(-1.0)/1000.0;
    for (std::size_t index=0;
         index<authoritative_mode_counts.size();
         ++index) {
        constrained.fft_lattice_expected_mode_count=
            authoritative_mode_counts[index];
        const double lower=
            exact_fundamental
            +exact_fundamental*static_cast<double>(index);
        const double upper=
            exact_fundamental
            +exact_fundamental
                 *static_cast<double>(index+1);
        const auto rule=shell::make_radial_shell_rule(
            lower,upper,constrained);
        require(
            rule.lattice_mode_count
                ==authoritative_mode_counts[index],
            "estimator-constrained FFT rule reproduces "
            "authoritative P0 mode count");
        require(
            rule.lattice_boundary_mode_adjustment
                ==expected_boundary_adjustments[index],
            "estimator-constrained FFT rule reports "
            "the exact boundary allocation");
        require(
            rule.actual_radial_order
                ==std::min<std::size_t>(
                    4,rule.lattice_unique_radius_count),
            "estimator-constrained FFT rule adapts only "
            "to discrete support");
        double effective_k=0.0;
        for (const auto& node:rule.nodes) {
            effective_k+=node.weight*node.k;
        }
        require(
            effective_k>=lower && effective_k<upper,
            "estimator-constrained FFT effective k lies "
            "inside its requested bin");
    }
    constrained.fft_lattice_expected_mode_count=33;
    bool impossible_mode_count_failed=false;
    try {
        (void)shell::make_radial_shell_rule(
            exact_fundamental,
            2.0*exact_fundamental,
            constrained);
    } catch (const std::invalid_argument&) {
        impossible_mode_count_failed=true;
    }
    require(
        impossible_mode_count_failed,
        "estimator-constrained FFT rule rejects a mode "
        "count not explainable by boundary allocation");
}

void test_factorized_b321ii_matches_direct() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    const marisa_b_halo_v1::CanonicalTriangle canonical=
        marisa_b_halo_v1::canonicalize_triangle({0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    eft::TracerPowerIntegrationConfig p13_config;
    p13_config.qmin=1.0e-3;
    p13_config.qmax=0.8;
    p13_config.n_radial=12;
    p13_config.n_mu=24;
    p13_config.restore_p13_uv_tail=false;
    const auto factorized=eft::factorized_b321ii_templates(
        power,triangle,provider,p13_config);
    std::array<eft::RenormalizedTracerP13Template,3> legs;
    for (std::size_t index=0;index<legs.size();++index) {
        legs[index]=eft::renormalized_tracer_p13_template(
            power,marisa_b_halo_v1::norm(triangle[index]),
            provider,p13_config);
    }
    const auto assembled=eft::assemble_factorized_b321ii_templates(
        power,triangle,provider,legs);
    require_polynomial_close(
        assembled.value,factorized.value,2.0e-14,1.0e-10,
        "cached P13 legs reproduce factorized B321II");
    require_polynomial_close(
        factorized.value,
        factorized.bare-factorized.bias_subtraction
            +factorized.uv_tail_restoration,
        2.0e-13,1.0e-7,
        "factorized B321II retains bare-minus-subtraction-plus-tail decomposition");

    eft::DirectIntegrationConfig direct_config;
    direct_config.qmin=p13_config.qmin;
    direct_config.qmax=p13_config.qmax;
    direct_config.n_radial=12;
    direct_config.n_mu=24;
    direct_config.n_phi=12;
    direct_config.ir_safe=true;
    direct_config.uv_subtract=true;
    direct_config.exploit_phi_reflection=true;
    direct_config.uv=p13_config.uv;
    const auto direct=eft::evaluate_direct(
        power,triangle,provider,direct_config);
    require_close(
        factorized.value.evaluate(matter()),
        direct.B321II.evaluate(matter()),
        2.0e-4,2.0e-4,
        "factorized P13 B321II matches independent 3D direct integral");
    require(factorized.integration_nodes>0,
            "factorized P13 B321II records integration nodes");
}

}  // namespace

int main() {
    try {
        test_components_and_ir_power_counting();
        test_fft_lattice_shell_average();
        test_factorized_b321ii_matches_direct();
        std::cout<<"EFT-v2 tracer-power checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 tracer-power test failed: "<<error.what()<<"\n";
        return 1;
    }
}
