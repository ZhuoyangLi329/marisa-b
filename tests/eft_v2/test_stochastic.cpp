#include "stochastic.h"

#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "PowerSpectrum.h"
#include "halo_v1.h"
#include "poisson_reconstruction.h"
#include "tracer_power.h"

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
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    double relative,double absolute,
    const std::string& message) {
    for (const auto& term:actual.terms()) {
        require_close(
            term.second,
            expected.coefficient(term.first),
            relative,absolute,
            message+" "+term.first.canonical_string());
    }
    for (const auto& term:expected.terms()) {
        require_close(
            actual.coefficient(term.first),
            term.second,relative,absolute,
            message+" reverse "
                +term.first.canonical_string());
    }
}

class GaussianPower final:public PowerSpectrum {
public:
    real Evaluate(real k) const override { return 1000.0*std::exp(-k*k); }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("GaussianPower has no cosmology");
    }
};

class ScaledGaussianPower final:public PowerSpectrum {
public:
    explicit ScaledGaussianPower(double scale):scale_(scale) {}
    real Evaluate(real k) const override { return scale_*1000.0*std::exp(-k*k); }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("ScaledGaussianPower has no cosmology");
    }
private:
    double scale_;
};

class MarkedOracleBase final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        return eft::SparsePolynomial::constant(
            2.0+0.3*static_cast<double>(momenta.size()));
    }
    std::string_view name() const noexcept override {
        return "marked_oracle_base";
    }
};

class MarkedOracleDirection final:
    public eft::MarkedFieldKernelProvider {
public:
    eft::SparsePolynomial marked(
        const std::vector<eft::Vec3>& matter_momenta,
        const eft::Vec3&) const override {
        return eft::SparsePolynomial::constant(
            3.0+0.7*
                static_cast<double>(matter_momenta.size()));
    }
    std::string_view name() const noexcept override {
        return "marked_oracle_direction";
    }
};

class ShiftOracleBase final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        return eft::SparsePolynomial::constant(
            4.0+0.5*static_cast<double>(momenta.size()));
    }
    std::string_view name() const noexcept override {
        return "shift_oracle_base";
    }
};

class MarkedShiftOracleDirection final:
    public eft::MarkedFieldKernelProvider {
public:
    eft::SparsePolynomial marked(
        const std::vector<eft::Vec3>& matter_momenta,
        const eft::Vec3&) const override {
        return eft::SparsePolynomial::constant(
            5.0+0.9*
                static_cast<double>(matter_momenta.size()));
    }
    std::string_view name() const noexcept override {
        return "marked_shift_oracle_direction";
    }
};

class ZeroOracleBase final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>&) const override {
        return {};
    }
    std::string_view name() const noexcept override {
        return "zero_oracle_base";
    }
};

class FiniteK1Oracle final:public eft::FieldKernelProvider {
public:
    FiniteK1Oracle(
        const eft::FieldKernelProvider& base,
        double amplitude):
        base_(base),amplitude_(amplitude) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result=
            base_.deterministic(momenta);
        if (momenta.size()==1) {
            result+=eft::SparsePolynomial::constant(amplitude_);
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "finite_K1_oracle";
    }

private:
    const eft::FieldKernelProvider& base_;
    double amplitude_;
};

std::array<double,eft::kParameterCount> bias_point() {
    std::array<double,eft::kParameterCount> values={};
    values[static_cast<std::size_t>(eft::ParameterId::B1)]=2.3;
    values[static_cast<std::size_t>(eft::ParameterId::B2)]=0.4;
    values[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.2;
    values[static_cast<std::size_t>(eft::ParameterId::Gamma21)]=0.1;
    return values;
}

void test_registry() {
    const auto& ids=eft::stochastic_parameter_ids();
    require(ids.size()==eft::kStochasticCount,"stochastic registry size");
    for (std::size_t index=0;index<ids.size();++index) {
        require(eft::parameter_info(ids[index]).sector==eft::ParameterSector::Stochastic,
                "stochastic registry sector");
        if (index>0) {
            require(static_cast<int>(ids[index])==static_cast<int>(ids[index-1])+1,
                    "stochastic registry stable ordering");
        }
    }
}

void test_finite_k1_leading_bshot_response() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider base;
    const FiniteK1Oracle zero(base,0.0);
    const FiniteK1Oracle plus(base,1.0);
    const FiniteK1Oracle minus(base,-1.0);
    const std::array<eft::Vec3,3> triangle{{
        {0.042,0.011,-0.006},
        {-0.017,0.039,0.008},
        {-0.025,-0.050,-0.002}}};
    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;

    const auto generating_coefficients=[&](bool stochastic_shift) {
        const auto value0=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,zero,reconstruction,
                stochastic_shift);
        const auto value_plus=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,plus,reconstruction,
                stochastic_shift);
        const auto value_minus=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,minus,reconstruction,
                stochastic_shift);
        return std::array<eft::SparsePolynomial,3>{{
            value0,
            0.5*(value_plus-value_minus),
            0.5*(value_plus+value_minus)-value0}};
    };
    const auto shared_eq265_coefficients=[&](
        bool stochastic_shift) {
        const auto value0=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,zero,reconstruction,
                stochastic_shift);
        const auto value_plus=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,plus,reconstruction,
                stochastic_shift);
        const auto value_minus=
            eft::
            reconstructed_leading_bshot_residual_generating_shape(
                power,triangle,minus,reconstruction,
                stochastic_shift);
        return std::array<eft::SparsePolynomial,3>{{
            value0,
            0.25*(value_plus-value_minus),
            0.5*(value_plus+value_minus)-value0}};
    };
    const auto density=shared_eq265_coefficients(false);
    const auto tied=shared_eq265_coefficients(true);
    for (std::size_t order=0;order<density.size();++order) {
        require_polynomial_close(
            tied[order],density[order],
            2.0e-13,2.0e-10,
            "leading Bshot finite-PNG tied/density closure");
    }

    double power_sum=0.0;
    for (const eft::Vec3& leg:triangle) {
        power_sum+=power(
            hv1::norm(leg));
    }
    const auto b1=eft::SparsePolynomial::variable(
        eft::ParameterId::B1);
    const std::array<eft::SparsePolynomial,3> expected{{
        power_sum*b1*b1,
        power_sum*b1,
        eft::SparsePolynomial::constant(power_sum)}};
    for (std::size_t order=0;order<density.size();++order) {
        require_polynomial_close(
            density[order],expected[order],
            2.0e-13,2.0e-10,
            "shared Eq. (2.65) Bshot G/L/Q coefficient");
    }

    for (const bool stochastic_shift:{false,true}) {
        const auto finite=generating_coefficients(
            stochastic_shift);
        for (const double amplitude:{-0.63,0.37,1.41}) {
            const FiniteK1Oracle direct(base,amplitude);
            const auto direct_value=
                eft::
                reconstructed_leading_bshot_residual_generating_shape(
                    power,triangle,direct,reconstruction,
                    stochastic_shift);
            require_polynomial_close(
                direct_value,
                finite[0]+amplitude*finite[1]
                    +amplitude*amplitude*finite[2],
                2.0e-13,2.0e-10,
                "finite-K1 direct coefficient recombination");
        }
    }

    auto pre_limit=reconstruction;
    pre_limit.smoothing_radius=1.0e6;
    const auto density_pre=
        eft::
        reconstructed_leading_bshot_residual_generating_shape(
            power,triangle,plus,pre_limit,false);
    const auto tied_pre=
        eft::
        reconstructed_leading_bshot_residual_generating_shape(
            power,triangle,plus,pre_limit,true);
    require_polynomial_close(
        tied_pre,density_pre,
        2.0e-13,2.0e-10,
        "finite-K1 leading Bshot R-infinity map closure");
}

void test_marked_reconstruction_combinatorics() {
    const MarkedOracleBase base;
    const MarkedOracleDirection marked;
    const ShiftOracleBase shift_base;
    const MarkedShiftOracleDirection shift_marked;
    const ZeroOracleBase zero;
    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    const eft::Vec3 m0{0.031,0.012,-0.008};
    const eft::Vec3 m1{-0.009,0.027,0.015};
    const eft::Vec3 epsilon{0.014,-0.006,0.021};
    const auto shift=[&](
        const eft::Vec3& output,
        const eft::Vec3& block) {
        return hv1::reconstruction_shift_factor(
            output,block,reconstruction);
    };
    const auto add=[](
        const eft::Vec3& left,
        const eft::Vec3& right) {
        return hv1::add(left,right);
    };

    const double E0=3.0;
    const double E1=3.7;
    const double E2=4.4;
    const double A1=2.3;
    const double A2=2.6;
    require_close(
        eft::reconstructed_marked_field_kernel(
            base,marked,base,marked,reconstruction,
            {},epsilon).coefficient(eft::MonomialKey{}),
        E0,0.0,1.0e-14,
        "marked reconstruction pure epsilon");

    const eft::Vec3 out1=add(m0,epsilon);
    const double expected1=
        E1+E0*A1*(
            shift(out1,m0)+shift(out1,epsilon));
    require_close(
        eft::reconstructed_marked_field_kernel(
            base,marked,base,marked,reconstruction,
            {m0},epsilon).coefficient(eft::MonomialKey{}),
        expected1,2.0e-14,2.0e-14,
        "marked reconstruction one-matter ordered oracle");

    const eft::Vec3 out2=add(add(m0,m1),epsilon);
    const double R0=shift(out2,m0);
    const double R1=shift(out2,m1);
    const double Re=shift(out2,epsilon);
    const double Re0=shift(out2,add(epsilon,m0));
    const double Re1=shift(out2,add(epsilon,m1));
    const double R01=shift(out2,add(m0,m1));
    const double expected2=
        E2
        +0.5*(
            E1*A1*R1+A1*E1*Re0
            +E1*A1*R0+A1*E1*Re1)
        +(E0*A2*R01+A2*E0*Re)
        +0.5*(
            E0*A1*A1*R0*R1
            +A1*E0*A1*Re*R1
            +A1*E0*A1*Re*R0);
    require_close(
        eft::reconstructed_marked_field_kernel(
            base,marked,base,marked,reconstruction,
            {m0,m1},epsilon).coefficient(
                eft::MonomialKey{}),
        expected2,3.0e-14,3.0e-14,
        "marked reconstruction two-matter independent oracle");

    auto disabled=reconstruction;
    disabled.enabled=false;
    require_close(
        eft::reconstructed_marked_field_kernel(
            base,marked,base,marked,disabled,
            {m0,m1},epsilon).coefficient(
                eft::MonomialKey{}),
        E2,0.0,1.0e-14,
        "marked reconstruction disabled pre limit");

    const eft::SparsePolynomial multimarked_r0=
        eft::reconstructed_multimarked_field_kernel(
            base,marked,shift_base,shift_marked,
            reconstruction,{m0,m1},{});
    const eft::SparsePolynomial product_rule_r0=
        eft::reconstructed_field_kernel_variation_with_shift(
            base,zero,shift_base,zero,
            reconstruction,{m0,m1}).value;
    require_polynomial_close(
        multimarked_r0,product_rule_r0,
        3.0e-14,3.0e-14,
        "multimarked r=0 retains distinct shift provider");

    const eft::SparsePolynomial multimarked_r1=
        eft::reconstructed_multimarked_field_kernel(
            base,marked,shift_base,shift_marked,
            reconstruction,{m0,m1},{epsilon});
    const eft::SparsePolynomial singly_marked=
        eft::reconstructed_marked_field_kernel(
            base,marked,shift_base,shift_marked,
            reconstruction,{m0,m1},epsilon);
    require_polynomial_close(
        multimarked_r1,singly_marked,
        3.0e-14,3.0e-14,
        "multimarked r=1 exactly matches marked helper");

    const eft::Vec3 epsilon1{-0.011,0.019,0.004};
    const eft::Vec3 out_eps=add(epsilon,epsilon1);
    const double expected_r2=
        0.5*3.0*5.0*(
            shift(out_eps,epsilon)
            +shift(out_eps,epsilon1));
    const auto resolved_r2=
        eft::reconstructed_multimarked_field_terms(
            base,marked,shift_base,shift_marked,
            reconstruction,{},{
                epsilon,epsilon1});
    require(resolved_r2.size()==2,
            "multimarked r=2 has two density-shift assignments");
    for (const auto& term:resolved_r2) {
        require(term.stochastic_roles.size()==2,
                "multimarked r=2 preserves both role labels");
        int density_roles=0;
        for (const auto role:term.stochastic_roles) {
            density_roles+=
                role==eft::StochasticBlockRole::Density;
        }
        require(density_roles==1,
                "multimarked r=2 has exactly one density role");
    }
    require_close(
        eft::reconstructed_multimarked_field_kernel(
            base,marked,shift_base,shift_marked,
            reconstruction,{},{
                epsilon,epsilon1}).coefficient(
                    eft::MonomialKey{}),
        expected_r2,3.0e-14,3.0e-14,
        "multimarked r=2 independent ordered oracle");

    require(
        eft::reconstructed_multimarked_field_kernel(
            base,marked,shift_base,shift_marked,
            disabled,{},{
                epsilon,epsilon1}).empty(),
        "multimarked r=2 vanishes before reconstruction");
}

void test_shapes_and_estimator_mapping() {
    const GaussianPower power;
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle({0.04,0.06,0.08});
    const std::array<eft::Vec3,3> triangle={{canonical.k1,canonical.k2,canonical.k3}};
    eft::StochasticIntegrationConfig integration;
    integration.qmin=1.0e-3;
    integration.qmax=1.0;
    integration.n_radial=5;
    integration.n_mu=10;
    const auto templates=eft::stochastic_bispectrum(power,triangle,0.30,integration);
    const ScaledGaussianPower doubled_tree(2.0);
    const auto split_power_templates=eft::stochastic_bispectrum(
        power,triangle,0.30,integration,nullptr,&doubled_tree);
    auto values=bias_point();
    const std::array<double,3> k={{0.04,0.06,0.08}};
    const std::array<double,3> P={{power(k[0]),power(k[1]),power(k[2])}};
    require_close(templates.pure[2].evaluate(values),1.0,0.0,0.0,
                  "pure residual constant raw shape");
    require_close(
        templates.pure[3].evaluate(values),
        (k[0]*k[0]+k[1]*k[1]+k[2]*k[2])/(0.30*0.30),
        1.0e-14,1.0e-14,"pure derivative raw shape");
    require_close(
        templates.mixed_tree[4].evaluate(values),
        values[static_cast<std::size_t>(eft::ParameterId::B1)]
        *values[static_cast<std::size_t>(eft::ParameterId::B1)]
        *(P[0]+P[1]+P[2]),1.0e-14,1.0e-10,"mixed tree raw shape");
    require_close(
        split_power_templates.mixed_tree[4].evaluate(values),
        2.0*templates.mixed_tree[4].evaluate(values),1.0e-14,1.0e-10,
        "leading mixed stochastic uses the NLO-matched tree spectrum");
    require_close(
        split_power_templates.derivative_mixed[8].evaluate(values),
        templates.derivative_mixed[8].evaluate(values),1.0e-14,1.0e-10,
        "derivative mixed stochastic retains the loop-order spectrum");
    require_close(
        templates.bshot_bnabla2_cross.evaluate(values),
        -values[static_cast<std::size_t>(eft::ParameterId::B1)]
        *(P[0]*k[0]*k[0]+P[1]*k[1]*k[1]+P[2]*k[2]*k[2])/(0.30*0.30),
        1.0e-14,1.0e-10,"COBRA Eq. (62) Bshot-times-bnabla normalization");
    for (std::size_t index=0;index<eft::kStochasticCount;++index) {
        require(
            templates.raw[index].serialize()==
                (templates.pure[index]+templates.mixed_tree[index]
                 +templates.mixed_one_loop[index]
                 +templates.derivative_mixed[index]).serialize(),
            "stochastic component recomposition");
    }
    require_close(templates.evaluate(values,2.0e-4),0.0,0.0,0.0,
                  "estimator-subtracted Poisson point is zero");

    values[static_cast<std::size_t>(eft::ParameterId::AshotResidual)]=0.7;
    const double pure_at_n=templates.evaluate(values,2.0e-4);
    const double pure_at_2n=templates.evaluate(values,4.0e-4);
    require_close(pure_at_n/pure_at_2n,4.0,1.0e-14,1.0e-14,
                  "pure stochastic nbar^-2 scaling");
    values[static_cast<std::size_t>(eft::ParameterId::AshotResidual)]=0.0;
    values[static_cast<std::size_t>(eft::ParameterId::BshotResidual)]=0.7;
    const double mixed_at_n=templates.evaluate(values,2.0e-4);
    const double mixed_at_2n=templates.evaluate(values,4.0e-4);
    require_close(mixed_at_n/mixed_at_2n,2.0,1.0e-14,1.0e-14,
                  "mixed stochastic nbar^-1 scaling");

    const double alpha3=1.4;
    const double alpha4=-0.6;
    const double v1=alpha3*templates.mixed_tree[4].evaluate(values)/(2.0e-4)
                    +alpha4/(2.0e-4*2.0e-4);
    const double v2=alpha3*templates.mixed_tree[4].evaluate(values)/(2.0e-4)
                    +alpha4*templates.pure[2].evaluate(values)
                     /(2.0e-4*2.0e-4);
    require_close(v2,v1,1.0e-14,1.0e-8,"v1 alpha3/alpha4 compatibility map");

    const auto f1=templates.derivative_mixed[8];
    const auto f2=templates.derivative_mixed[9];
    const auto f3=templates.derivative_mixed[10];
    const auto f4=templates.derivative_mixed[11];
    require_close(
        (f1+f2-0.5*f3-0.5*f4).evaluate(values),0.0,0.0,1.0e-10,
        "real-space derivative stochastic exact degeneracy");

    const std::array<double,3> tracer_power={{12.0,23.0,31.0}};
    require_close(
        eft::analytic_poisson_bispectrum(tracer_power,2.0e-4),
        (12.0+23.0+31.0)/(2.0e-4)+1.0/(2.0e-4*2.0e-4),
        1.0e-15,1.0e-8,"analytic Poisson add-back");

    for (std::size_t index=2;index<eft::kStochasticCount;++index) {
        require(!templates.raw[index].empty(),"every B-sector stochastic shape is present");
        require(std::isfinite(templates.raw[index].evaluate(values)),
                "every B-sector stochastic shape is finite");
    }
}

void test_mixed_cross_power_renormalization() {
    const GaussianPower power;
    const double k=0.07;
    eft::StochasticIntegrationConfig stochastic;
    stochastic.qmin=1.0e-3;
    stochastic.qmax=0.8;
    stochastic.n_radial=8;
    stochastic.n_mu=16;
    stochastic.mu_ren=0.30;
    stochastic.asymptotic_factor=64.0;
    const auto cross=eft::mixed_stochastic_cross_power(
        power,k,stochastic);
    const auto values=bias_point();
    for (std::size_t index=0;index<4;++index) {
        require_close(
            cross.P22[index].evaluate(values),
            (cross.P22_bare[index]
             -cross.P22_zero_lag_subtraction[index]).evaluate(values),
            2.0e-13,1.0e-7,
            "mixed stochastic P22 bare-minus-zero-lag decomposition");
        require_close(
            cross.P13[index].evaluate(values),
            (cross.P13_bare[index]
             -cross.P13_bias_subtraction[index]
             +cross.P13_uv_tail_restoration[index]).evaluate(values),
            2.0e-13,1.0e-7,
            "mixed stochastic P13 bare-minus-bias-plus-tail decomposition");
        require_close(
            cross[index].evaluate(values),
            (cross.P22[index]+cross.P13[index]).evaluate(values),
            2.0e-13,1.0e-7,
            "mixed stochastic renormalized P22-plus-P13 decomposition");
    }
    require(cross.integration_nodes>0,
            "mixed stochastic records integration work");

    const double b1=values[static_cast<std::size_t>(eft::ParameterId::B1)];
    const double b2=values[static_cast<std::size_t>(eft::ParameterId::B2)];
    const double gamma2=
        values[static_cast<std::size_t>(eft::ParameterId::Gamma2)];
    const double gamma21=
        values[static_cast<std::size_t>(eft::ParameterId::Gamma21)];
    const double bGamma3=-(7.0/4.0)*gamma21-gamma2;
    const auto combine=[&](const std::array<eft::SparsePolynomial,4>& pieces) {
        return b1*pieces[0]+b2*pieces[1]
               +gamma2*pieces[2]+bGamma3*pieces[3];
    };

    eft::TracerPowerIntegrationConfig tracer_config;
    tracer_config.qmin=stochastic.qmin;
    tracer_config.qmax=stochastic.qmax;
    tracer_config.n_radial=stochastic.n_radial;
    tracer_config.n_mu=stochastic.n_mu;
    tracer_config.uv.mu_ren=stochastic.mu_ren;
    tracer_config.uv.asymptotic_factor=stochastic.asymptotic_factor;
    const eft::EftBiasKernelProvider provider;
    const auto tracer=eft::tracer_power_templates(
        power,power,k,provider,tracer_config);
    require_close(
        combine(cross.P22_bare).evaluate(values),
        tracer.P22_bare.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces tracer P22 bare");
    require_close(
        combine(cross.P22_zero_lag_subtraction).evaluate(values),
        tracer.P22_stochastic_subtraction.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces tracer P22 zero lag");
    require_close(
        combine(cross.P13_bare).evaluate(values),
        tracer.P13_bare.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces tracer P13 bare");
    require_close(
        combine(cross.P13_bias_subtraction).evaluate(values),
        tracer.P13_bias_subtraction.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces tracer P13 bias subtraction");
    require_close(
        combine(cross.P13_uv_tail_restoration).evaluate(values),
        tracer.P13_uv_tail_restoration.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces tracer P13 UV tail");
    require_close(
        combine(cross.value).evaluate(values),
        tracer.one_loop.evaluate(values),
        3.0e-13,1.0e-6,
        "mixed stochastic auto-limit reproduces renormalized tracer loop");
    require(
        std::fabs(
            combine(cross.P22_zero_lag_subtraction).evaluate(values))>1.0,
        "mixed stochastic P22 zero-lag test has discriminatory power");
    require(
        std::fabs(
            combine(cross.P13_uv_tail_restoration).evaluate(values))>1.0,
        "mixed stochastic P13 tail-restoration test has discriminatory power");
}

void test_reconstructed_stochastic_map() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider base;
    const double k=0.07;
    const eft::Vec3 external{0.0,0.0,k};
    eft::StochasticIntegrationConfig integration;
    integration.qmin=1.0e-3;
    integration.qmax=0.60;
    integration.n_radial=3;
    integration.n_mu=6;
    integration.n_phi=8;
    integration.restore_p13_uv_tail=false;

    hv1::ReconstructionConfig no_shift;
    no_shift.enabled=true;
    no_shift.smoothing_radius=15.0;
    no_shift.bias_recon=1.0e300;
    no_shift.cell_size=8.0;
    const auto pre=eft::mixed_stochastic_cross_power(
        power,k,integration);
    const auto limiting=
        eft::reconstructed_mixed_stochastic_cross_power(
            power,external,base,no_shift,integration);
    for (std::size_t index=0;index<4;++index) {
        require_polynomial_close(
            limiting.P22_bare[index],
            pre.P22_bare[index],
            2.0e-11,2.0e-7,
            "R-infinity reconstructed stochastic P22 bare");
        require_polynomial_close(
            limiting.P22_zero_lag_subtraction[index],
            pre.P22_zero_lag_subtraction[index],
            2.0e-11,2.0e-7,
            "R-infinity reconstructed stochastic P22 subtraction");
        require_polynomial_close(
            limiting.P22[index],pre.P22[index],
            2.0e-11,2.0e-7,
            "R-infinity reconstructed stochastic P22");
        require_polynomial_close(
            limiting.P13[index],pre.P13[index],
            2.0e-11,2.0e-7,
            "R-infinity reconstructed stochastic P13");
        require_polynomial_close(
            limiting.value[index],pre.value[index],
            2.0e-11,2.0e-7,
            "R-infinity reconstructed stochastic total");
    }

    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    integration.restore_p13_uv_tail=true;
    integration.uv_tail_kmax=20.0;
    integration.uv_tail_quadrature_order=64;
    const eft::Vec3 oriented_external{
        k/std::sqrt(6.0),
        2.0*k/std::sqrt(6.0),
        k/std::sqrt(6.0)};
    const auto post=
        eft::reconstructed_mixed_stochastic_cross_power(
            power,oriented_external,base,
            reconstruction,integration);
    const auto values=bias_point();
    for (std::size_t index=0;index<4;++index) {
        require_polynomial_close(
            post.P22[index],
            post.P22_bare[index]
                -post.P22_zero_lag_subtraction[index],
            3.0e-11,2.0e-6,
            "post stochastic P22 recomposition");
        require_polynomial_close(
            post.P13[index],
            post.P13_bare[index]
                -post.P13_bias_subtraction[index]
                +post.P13_uv_tail_restoration[index],
            3.0e-5,2.0e-5,
            "post stochastic P13 recomposition");
        require_polynomial_close(
            post.value[index],
            post.P22[index]+post.P13[index],
            3.0e-11,2.0e-6,
            "post stochastic loop recomposition");
        require(
            std::isfinite(post.value[index].evaluate(values)),
            "post stochastic value is finite");
    }
    require(
        post.integration_nodes>0,
        "post stochastic records integration nodes");
    require(
        !post.P13_uv_tail_restoration[0].empty(),
        "post stochastic has numerical q^-2 tail");

    const hv1::CanonicalTriangle canonical=
        hv1::canonicalize_triangle({0.04,0.06,0.08});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const auto decomposition=
        eft::reconstructed_stochastic_bispectrum_decomposition(
            power,triangle,base,reconstruction,
            0.30,integration);
    const auto& templates=decomposition.tied;
    const eft::SparsePolynomial expected_leading=
        (power(0.04)+power(0.06)+power(0.08))
        *eft::SparsePolynomial::variable(
            eft::ParameterId::B1)
        *eft::SparsePolynomial::variable(
            eft::ParameterId::B1);
    require_polynomial_close(
        decomposition.tied.mixed_tree[4],
        expected_leading,
        3.0e-13,3.0e-9,
        "marked post Bshot residual Poisson-subtraction closure");
    require_polynomial_close(
        decomposition.density_only.mixed_tree[4],
        expected_leading,
        3.0e-13,3.0e-9,
        "marked density-only Bshot residual closure");
    require(
        decomposition.noisy_shift.mixed_tree[4].empty(),
        "marked K0 flow cancels from Bshot residual noisy-shift branch");
    double noisy_shift_norm=0.0;
    for (std::size_t index=0;
         index<eft::kStochasticCount;
         ++index) {
        require_polynomial_close(
            decomposition.tied.raw[index],
            decomposition.density_only.raw[index]
                +decomposition.noisy_shift.raw[index],
            3.0e-11,2.0e-6,
            "post stochastic tied density-plus-shift");
        noisy_shift_norm+=std::fabs(
            decomposition.noisy_shift.raw[index]
                .evaluate(values));
    }
    require(
        noisy_shift_norm>1.0e-5,
        "post stochastic noisy-shift branch has discriminatory power");
    const std::array<eft::Vec3,3> permuted={{
        triangle[1],triangle[2],triangle[0]}};
    const auto permuted_decomposition=
        eft::reconstructed_stochastic_bispectrum_decomposition(
            power,permuted,base,reconstruction,
            0.30,integration);
    const auto& permuted_templates=
        permuted_decomposition.tied;
    for (std::size_t index=0;
         index<eft::kStochasticCount;
         ++index) {
        require_polynomial_close(
            templates.raw[index],
            permuted_templates.raw[index],
            3.0e-11,2.0e-6,
            "post stochastic external permutation");
    }

    bool rejected_disabled=false;
    try {
        auto disabled=reconstruction;
        disabled.enabled=false;
        (void)eft::reconstructed_mixed_stochastic_cross_power(
            power,external,base,disabled,integration);
    } catch (const std::invalid_argument&) {
        rejected_disabled=true;
    }
    require(
        rejected_disabled,
        "post stochastic rejects disabled reconstruction");
}

void test_fixed_poisson_estimator_matching() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider base;
    const hv1::CanonicalTriangle canonical=
        hv1::canonicalize_triangle(
            {0.043,0.061,0.079});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    eft::StochasticIntegrationConfig integration;
    integration.qmin=2.0e-3;
    integration.qmax=0.25;
    integration.n_radial=2;
    integration.n_mu=2;
    integration.n_phi=2;
    integration.restore_p13_uv_tail=false;

    const auto tied=
        eft::reconstructed_fixed_poisson_bispectrum(
            power,triangle,base,reconstruction,
            integration,
            eft::ReconstructedStochasticMap::
                TiedDensityAndShift,
            eft::PoissonIntensityMode::UnitK0,
            true);
    const eft::SparsePolynomial b1sq=
        eft::SparsePolynomial::variable(
            eft::ParameterId::B1)
        *eft::SparsePolynomial::variable(
            eft::ParameterId::B1);
    eft::SparsePolynomial expected_leading;
    for (int matter_leg=0;
         matter_leg<3;
         ++matter_leg) {
        for (int density_noise_leg=0;
             density_noise_leg<3;
             ++density_noise_leg) {
            if (matter_leg==density_noise_leg) continue;
            const int mixed_leg=
                3-matter_leg-density_noise_leg;
            const double shift=
                hv1::reconstruction_shift_factor(
                    triangle[
                        static_cast<std::size_t>(
                            mixed_leg)],
                    eft::Vec3{
                        -triangle[
                            static_cast<std::size_t>(
                                density_noise_leg)].x,
                        -triangle[
                            static_cast<std::size_t>(
                                density_noise_leg)].y,
                        -triangle[
                            static_cast<std::size_t>(
                                density_noise_leg)].z},
                    reconstruction);
            expected_leading+=
                power(hv1::norm(
                    triangle[
                        static_cast<std::size_t>(
                            matter_leg)]))
                *shift*b1sq;
        }
    }
    require_polynomial_close(
        tied.tree.by_inverse_number_density[1],
        expected_leading,
        5.0e-13,5.0e-10,
        "fixed Poisson leading estimator-filtered D-S oracle");
    require(
        tied.tree.by_inverse_number_density[2].empty()
        &&tied.tree.by_inverse_number_density[3].empty(),
        "fixed Poisson tree pure external cumulants are subtracted");
    require(
        tied.tree.generated_topologies==13
        &&tied.tree.estimator_allowed_topologies==6,
        "fixed Poisson tree topology golden counts");
    require(
        tied.one_loop.generated_topologies==368
        &&tied.one_loop.estimator_allowed_topologies==311,
        "fixed Poisson one-loop topology golden counts");
    for (std::size_t inverse_nbar=1;
         inverse_nbar<4;
         ++inverse_nbar) {
        require(
            !tied.one_loop
                .by_inverse_number_density[inverse_nbar]
                .empty(),
            "fixed Poisson one-loop populates every inverse-nbar slot");
    }

    const auto density_only=
        eft::reconstructed_fixed_poisson_bispectrum(
            power,triangle,base,reconstruction,
            integration,
            eft::ReconstructedStochasticMap::
                DensityOnly,
            eft::PoissonIntensityMode::UnitK0,
            true);
    for (std::size_t inverse_nbar=1;
         inverse_nbar<4;
         ++inverse_nbar) {
        require(
            density_only.tree
                .by_inverse_number_density[inverse_nbar]
                .empty(),
            "density-only fixed Poisson tree is exactly estimator-subtracted");
        require(
            density_only.one_loop
                .by_inverse_number_density[inverse_nbar]
                .empty(),
            "density-only fixed Poisson one-loop is exactly estimator-subtracted");
    }

    auto infinite_smoothing=reconstruction;
    infinite_smoothing.smoothing_radius=1.0e6;
    const auto pre_limit=
        eft::reconstructed_fixed_poisson_bispectrum(
            power,triangle,base,infinite_smoothing,
            integration,
            eft::ReconstructedStochasticMap::
                TiedDensityAndShift,
            eft::PoissonIntensityMode::UnitK0,
            true);
    for (const auto& polynomial:
         pre_limit.total_by_inverse_number_density()) {
        require(
            polynomial.empty(),
            "fixed Poisson R-to-infinity pre limit vanishes");
    }

    auto conditional_integration=integration;
    conditional_integration.n_radial=1;
    conditional_integration.n_mu=1;
    conditional_integration.n_phi=1;
    const auto conditional=
        eft::reconstructed_fixed_poisson_bispectrum(
            power,triangle,base,reconstruction,
            conditional_integration,
            eft::ReconstructedStochasticMap::
                TiedDensityAndShift,
            eft::PoissonIntensityMode::
                ConditionalTracer,
            true);
    require_polynomial_close(
        conditional.tree.by_inverse_number_density[1],
        tied.tree.by_inverse_number_density[1],
        5.0e-13,5.0e-10,
        "conditional Poisson does not alter leading P-over-n oracle");
    require(
        conditional.tree.generated_topologies==16
        &&conditional.tree.estimator_allowed_topologies==6,
        "conditional Poisson tree topology golden counts");
    require(
        conditional.one_loop.generated_topologies==597
        &&conditional.one_loop.estimator_allowed_topologies==482,
        "conditional Poisson one-loop topology golden counts: "
        +std::to_string(
            conditional.one_loop.generated_topologies)
        +"/"
        +std::to_string(
            conditional.one_loop.estimator_allowed_topologies));
    require(
        conditional.one_loop.generated_topologies
            >tied.one_loop.generated_topologies,
        "conditional Poisson adds intensity-decorated one-loop topologies");
    const auto conditional_density_only=
        eft::reconstructed_fixed_poisson_bispectrum(
            power,triangle,base,reconstruction,
            conditional_integration,
            eft::ReconstructedStochasticMap::
                DensityOnly,
            eft::PoissonIntensityMode::
                ConditionalTracer,
            true);
    for (const auto& polynomial:
         conditional_density_only
             .total_by_inverse_number_density()) {
        require(
            polynomial.empty(),
            "conditional density-only Poisson mean is estimator-subtracted");
    }
}

}  // namespace

int main() {
    try {
        test_registry();
        test_finite_k1_leading_bshot_response();
        test_marked_reconstruction_combinatorics();
        test_shapes_and_estimator_mapping();
        test_mixed_cross_power_renormalization();
        test_reconstructed_stochastic_map();
        test_fixed_poisson_estimator_matching();
        std::cout<<"EFT-v2 stochastic checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 stochastic test failed: "<<error.what()<<"\n";
        return 1;
    }
}
