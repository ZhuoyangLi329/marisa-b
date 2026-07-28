#include "ir_resummation.h"

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

class EhNoWigglePower final:public PowerSpectrum {
public:
    explicit EhNoWigglePower(eft::IrResummationConfig config):config_(config) {}
    real Evaluate(real k) const override {
        return 2.5e7*eft::eisenstein_hu_nowiggle_shape(k,config_);
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("EhNoWigglePower has no cosmology");
    }
private:
    eft::IrResummationConfig config_;
};

class WigglyPower final:public PowerSpectrum {
public:
    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        return 500.0*std::pow(k,0.96)*std::exp(-k/0.7)
               *(1.0+0.06*std::sin(110.0*k)*std::exp(-k*k/0.16));
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("WigglyPower has no cosmology");
    }
};

void test_split_limits() {
    eft::IrResummationConfig config;
    config.grid_size=512;
    config.sigma2_override=3.0;
    const EhNoWigglePower no_wiggle_source(config);
    const eft::IrResummation no_wiggle(no_wiggle_source,config);
    for (int index=0;index<80;++index) {
        const double k=std::exp(std::log(2.0e-5)
            +(index+0.5)/80.0*std::log(50.0/2.0e-5));
        require_close(no_wiggle.no_wiggle_power()(k),no_wiggle_source(k),2.0e-11,1.0e-10,
                      "Gaussian EH-ratio split preserves an EH no-wiggle shape");
        require_close(no_wiggle.loop_power()(k),no_wiggle_source(k),2.0e-11,1.0e-10,
                      "Pw=0 loop limit");
        require_close(no_wiggle.tree_power()(k),no_wiggle_source(k),2.0e-11,1.0e-10,
                      "Pw=0 tree limit");
    }

    const WigglyPower wiggly;
    config.sigma2_override=0.0;
    const eft::IrResummation zero_sigma(wiggly,config);
    for (int index=0;index<80;++index) {
        const double k=std::exp(std::log(2.0e-5)
            +(index+0.5)/80.0*std::log(50.0/2.0e-5));
        require_close(zero_sigma.loop_power()(k),wiggly(k),2.0e-14,1.0e-14,
                      "Sigma=0 loop fixed-order limit");
        require_close(zero_sigma.tree_power()(k),wiggly(k),2.0e-14,1.0e-14,
                      "Sigma=0 tree fixed-order limit");
    }
}

void test_nlo_matching_and_sigma() {
    const WigglyPower power;
    eft::IrResummationConfig base;
    base.grid_size=512;
    base.sigma2_override=1.7;
    const eft::IrResummation resummed(power,base);
    const double k=0.13;
    const double smooth=resummed.no_wiggle_power()(k);
    const double x=k*k*resummed.sigma2();
    require_close(
        resummed.loop_power()(k),smooth+std::exp(-x)*(power(k)-smooth),
        1.0e-14,1.0e-14,"loop IR spectrum formula");
    require_close(
        resummed.tree_power()(k),smooth+(1.0+x)*std::exp(-x)*(power(k)-smooth),
        1.0e-14,1.0e-14,"NLO tree matching formula");

    base.sigma2_override=0.04;
    const eft::IrResummation small(power,base);
    base.sigma2_override=0.02;
    const eft::IrResummation half(power,base);
    const double first=std::fabs(small.tree_power()(k)-power(k));
    const double second=std::fabs(half.tree_power()(k)-power(k));
    require_close(first/second,4.0,3.0e-3,0.0,
                  "NLO matching removes the linear Sigma2 term");

    base.sigma2_override=std::numeric_limits<double>::quiet_NaN();
    const eft::IrResummation physical(power,base);
    require(physical.sigma2()>0.0 && std::isfinite(physical.sigma2()),
            "Sigma2 integral is positive and finite");
    const std::string metadata=physical.metadata_json();
    require(metadata.find("\"schema\":\"marisa-b-eft-v2-ir-v1\"")!=std::string::npos,
            "IR metadata schema");
    require(metadata.find("\"sigma_power\":\"P11\"")!=std::string::npos,
            "IR metadata sigma prescription");
}

void test_bispectrum_fixed_order_limit() {
    const WigglyPower power;
    eft::IrResummationConfig ir_config;
    ir_config.grid_size=256;
    ir_config.sigma2_override=0.0;
    const eft::IrResummation ir(power,ir_config);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle({0.043,0.061,0.079});
    const std::array<eft::Vec3,3> triangle={{canonical.k1,canonical.k2,canonical.k3}};
    const eft::MatterKernelProvider provider;
    eft::DirectIntegrationConfig integration;
    integration.qmin=1.0e-3; integration.qmax=0.30;
    integration.n_radial=4; integration.n_mu=8; integration.n_phi=6;
    integration.ir_safe=true; integration.uv_subtract=true;
    const auto fixed=eft::evaluate_direct(power,triangle,provider,integration);
    const auto resumed=eft::evaluate_ir_resummed(ir,triangle,provider,integration);
    std::array<double,eft::kParameterCount> matter={};
    matter[static_cast<std::size_t>(eft::ParameterId::B1)]=1.0;
    require_close(resumed.tree.evaluate(matter),fixed.tree.evaluate(matter),2.0e-13,1.0e-9,
                  "IR Sigma=0 bispectrum tree");
    require_close(resumed.total.evaluate(matter),fixed.total.evaluate(matter),2.0e-12,1.0e-7,
                  "IR Sigma=0 bispectrum total");
}

}  // namespace

int main() {
    try {
        test_split_limits();
        test_nlo_matching_and_sigma();
        test_bispectrum_fixed_order_limit();
        std::cout<<"EFT-v2 IR-resummation checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 IR-resummation test failed: "<<error.what()<<"\n";
        return 1;
    }
}
