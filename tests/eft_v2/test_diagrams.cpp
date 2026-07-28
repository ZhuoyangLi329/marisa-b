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
