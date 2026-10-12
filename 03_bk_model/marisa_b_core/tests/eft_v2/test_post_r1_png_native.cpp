#include "marisa_b_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "PowerSpectrum.h"

namespace native=marisa_b_native;

namespace {

int checks=0;

void require(bool condition,const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,
    double expected,
    double relative,
    double absolute,
    const std::string& message) {
    ++checks;
    const double error=std::fabs(actual-expected);
    const double scale=std::max(std::fabs(actual),std::fabs(expected));
    if (error>absolute && error>relative*scale) {
        std::ostringstream details;
        details.precision(17);
        details<<message<<": actual="<<actual
               <<", expected="<<expected
               <<", absolute_error="<<error;
        throw std::runtime_error(details.str());
    }
}

class SmoothPower final:public PowerSpectrum {
public:
    real Evaluate(real k) const override {
        return k>0.0
            ?1200.0*std::exp(-3.0*k*k)/(1.0+2.0*k)
            :0.0;
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("SmoothPower has no cosmology");
    }
};

class SmoothTransfer final:public PowerSpectrum {
public:
    real Evaluate(real k) const override {
        return k>0.0 ?0.35+4.0*k*k :0.0;
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("SmoothTransfer has no cosmology");
    }
};

native::ClosedTriangleVectors canonical_vectors(
    const native::Triangle& triangle) {
    const double sine=std::sqrt(
        std::max(0.0,1.0-triangle.mu12*triangle.mu12));
    native::ClosedTriangleVectors result{{
        {{0.0,0.0,triangle.k1}},
        {{triangle.k2*sine,0.0,triangle.k2*triangle.mu12}},
        {{-triangle.k2*sine,0.0,
          -(triangle.k1+triangle.k2*triangle.mu12)}}}};
    return result;
}

std::array<double,3> rotate_vector(
    const std::array<double,3>& vector) {
    const std::array<double,3> raw_axis{{1.0,2.0,3.0}};
    const double norm=std::sqrt(14.0);
    const std::array<double,3> axis{{
        raw_axis[0]/norm,
        raw_axis[1]/norm,
        raw_axis[2]/norm}};
    const double angle=0.371;
    const double cosine=std::cos(angle);
    const double sine=std::sin(angle);
    const double dot=
        axis[0]*vector[0]
        +axis[1]*vector[1]
        +axis[2]*vector[2];
    const std::array<double,3> cross{{
        axis[1]*vector[2]-axis[2]*vector[1],
        axis[2]*vector[0]-axis[0]*vector[2],
        axis[0]*vector[1]-axis[1]*vector[0]}};
    return {{
        cosine*vector[0]+sine*cross[0]
            +(1.0-cosine)*dot*axis[0],
        cosine*vector[1]+sine*cross[1]
            +(1.0-cosine)*dot*axis[1],
        cosine*vector[2]+sine*cross[2]
            +(1.0-cosine)*dot*axis[2]}};
}

native::ClosedTriangleVectors rotated_vectors(
    const native::ClosedTriangleVectors& input) {
    native::ClosedTriangleVectors result{};
    for (std::size_t index=0;index<input.size();++index) {
        result[index]=rotate_vector(input[index]);
    }
    return result;
}

double dot(
    const std::array<double,3>& left,
    const std::array<double,3>& right) {
    return left[0]*right[0]+left[1]*right[1]+left[2]*right[2];
}

std::array<double,3> add(
    const std::array<double,3>& left,
    const std::array<double,3>& right) {
    return {{
        left[0]+right[0],
        left[1]+right[1],
        left[2]+right[2]}};
}

double sinc(double value) {
    return std::fabs(value)<1.0e-8
        ?1.0-value*value/6.0
        :std::sin(value)/value;
}

double finite_adaptive_brec_gaussian_tree_piece(
    const SmoothPower& power,
    const SmoothTransfer& transfer,
    const native::ClosedTriangleVectors& vectors,
    const native::NativeConfig& config,
    double b1,
    double bphi_recon,
    double fnl_recon) {
    const int pairs[3][2]={{0,1},{1,2},{2,0}};
    double result=0.0;
    for (const auto& pair:pairs) {
        const int i=pair[0];
        const int j=pair[1];
        const auto output=add(vectors[i],vectors[j]);
        double shift_sum=0.0;
        for (const int index:{i,j}) {
            const auto& momentum=vectors[index];
            const double k2=dot(momentum,momentum);
            const double k=std::sqrt(k2);
            const double gaussian=std::exp(
                -0.5*k2*config.smoothing_radius
                    *config.smoothing_radius);
            const double half_cell=0.5*config.recon_cellsize;
            const double cic=
                config.recon_cic_window_power<=0
                ?1.0
                :std::pow(
                    sinc(momentum[0]*half_cell)
                    *sinc(momentum[1]*half_cell)
                    *sinc(momentum[2]*half_cell),
                    config.recon_cic_window_power);
            const double denominator=
                config.bias_recon
                +fnl_recon*bphi_recon/transfer(k);
            shift_sum+=-dot(output,momentum)/k2
                       *gaussian*cic/denominator;
        }
        const double pi=power(std::sqrt(dot(vectors[i],vectors[i])));
        const double pj=power(std::sqrt(dot(vectors[j],vectors[j])));
        result+=b1*b1*b1*b1*pi*pj*shift_sum;
    }
    return result;
}

double finite_adaptive_brec_png_linear_reconstruction_piece(
    const SmoothPower& power,
    const SmoothTransfer& transfer,
    const native::ClosedTriangleVectors& vectors,
    const native::NativeConfig& config,
    double b1,
    double bphi,
    double bphi_recon,
    double fnl_recon) {
    const int pairs[3][2]={{0,1},{1,2},{2,0}};
    double result=0.0;
    for (const auto& pair:pairs) {
        const int i=pair[0];
        const int j=pair[1];
        const auto output=add(vectors[i],vectors[j]);
        double shift_sum=0.0;
        double inverse_m_sum=0.0;
        for (const int index:{i,j}) {
            const auto& momentum=vectors[index];
            const double k2=dot(momentum,momentum);
            const double k=std::sqrt(k2);
            const double transfer_value=transfer(k);
            const double gaussian=std::exp(
                -0.5*k2*config.smoothing_radius
                    *config.smoothing_radius);
            const double half_cell=0.5*config.recon_cellsize;
            const double cic=
                config.recon_cic_window_power<=0
                ?1.0
                :std::pow(
                    sinc(momentum[0]*half_cell)
                    *sinc(momentum[1]*half_cell)
                    *sinc(momentum[2]*half_cell),
                    config.recon_cic_window_power);
            const double denominator=
                config.bias_recon
                +fnl_recon*bphi_recon/transfer_value;
            shift_sum+=-dot(output,momentum)/k2
                       *gaussian*cic/denominator;
            inverse_m_sum+=1.0/transfer_value;
        }
        const double pi=power(std::sqrt(dot(vectors[i],vectors[i])));
        const double pj=power(std::sqrt(dot(vectors[j],vectors[j])));
        result+=2.0*b1*b1*b1*bphi*pi*pj
                *inverse_m_sum*shift_sum;
    }
    return result;
}

native::NativeConfig post_config() {
    native::NativeConfig config;
    config.smoothing_radius=15.0;
    config.bias_recon=2.7340475186190334;
    config.recon_cellsize=8.0;
    config.recon_cic_window_power=4;
    config.singular_floor=1.0e-8;
    return config;
}

native::HaloBiasV1Params halo_bias() {
    native::HaloBiasV1Params bias;
    bias.b1=2.73;
    bias.b2=0.41;
    bias.bK2=-0.28;
    bias.bphi=5.7;
    bias.bphidelta=3.2;
    bias.bphi2=-1.1;
    return bias;
}

void compare_tree_components(
    const native::ComponentResult& actual,
    const native::ComponentResult& expected,
    double relative,
    double absolute,
    const std::string& label) {
#define CHECK_FIELD(name) \
    require_close( \
        actual.name,expected.name,relative,absolute, \
        label+" " #name)
    CHECK_FIELD(Btree);
    CHECK_FIELD(dBdfNL_local_tree);
    CHECK_FIELD(dBdfNL_local_primordial);
    CHECK_FIELD(dBdfNL_local_bphi_f2);
    CHECK_FIELD(dBdfNL_local_bphi_advection);
    CHECK_FIELD(dBdfNL_local_bphidelta);
    CHECK_FIELD(dBdfNL_local_bphi_b2);
    CHECK_FIELD(dBdfNL_local_bphi_bK2);
    CHECK_FIELD(dBdfNL_local_bphi_reconstruction);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_B0);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_sq_advection);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_sq_F2);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_sq_b2);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_sq_bK2);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_sq_reconstruction);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi_bphidelta);
    CHECK_FIELD(Bhalo_tree_fNL2_bphi2_operator);
    CHECK_FIELD(Bhalo_tree_fNL2_deterministic);
    CHECK_FIELD(Bhalo_tree_fNL2_stochastic_alpha3PNG_basis);
#undef CHECK_FIELD
}

void test_closed_vector_contract() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.047,0.063,-0.27};
    auto invalid=canonical_vectors(triangle);
    invalid[2][0]+=1.0e-4;
    const auto config=post_config();
    const auto bias=halo_bias();
    bool tree_rejected=false;
    bool response_rejected=false;
    bool b112_rejected=false;
    try {
        (void)native::
            compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
                power,transfer,invalid,config,bias);
    } catch (const std::invalid_argument&) {
        tree_rejected=true;
    }
    try {
        (void)native::compute_post_recon_local_png_1loop_dfNL_vectors(
            power,transfer,invalid,config,1.0);
    } catch (const std::invalid_argument&) {
        response_rejected=true;
    }
    try {
        (void)native::
            compute_post_recon_local_png_B112II_fNL2_coefficient_vectors(
                power,transfer,invalid,config);
    } catch (const std::invalid_argument&) {
        b112_rejected=true;
    }
    require(tree_rejected,"tree vector API rejects non-closure");
    require(response_rejected,"matter-response vector API rejects non-closure");
    require(b112_rejected,"B112 vector API rejects non-closure");
}

void test_tree_orientation_and_limits() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.047,0.063,-0.27};
    const auto vectors=canonical_vectors(triangle);
    const auto rotated=rotated_vectors(vectors);
    const auto bias=halo_bias();
    auto config=post_config();

    const auto scalar=
        native::compute_post_recon_halo_bias_v1_local_png_tree_dfNL(
            power,transfer,triangle,config,bias);
    const auto vector=
        native::
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,transfer,vectors,config,bias);
    compare_tree_components(
        vector,scalar,3.0e-13,3.0e-8,
        "canonical scalar/vector identity");

    auto isotropic=config;
    isotropic.recon_cic_window_power=0;
    const auto isotropic_canonical=
        native::
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,transfer,vectors,isotropic,bias);
    const auto isotropic_rotated=
        native::
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,transfer,rotated,isotropic,bias);
    compare_tree_components(
        isotropic_rotated,isotropic_canonical,
        2.0e-12,3.0e-8,
        "CIC-off rotation invariance");

    const auto cic_rotated=
        native::
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,transfer,rotated,config,bias);
    const double orientation_change=std::fabs(
        cic_rotated.dBdfNL_local_bphi_reconstruction
        -vector.dBdfNL_local_bphi_reconstruction);
    require(
        orientation_change
            >1.0e-10*std::max(
                std::fabs(
                    vector.dBdfNL_local_bphi_reconstruction),
                1.0),
        "CIC4 retains absolute-orientation dependence");

    auto infinite=config;
    infinite.smoothing_radius=1.0e6;
    const auto post_infinite=
        native::
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
            power,transfer,rotated,infinite,bias);
    const auto pre=
        native::compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(
            power,transfer,triangle,bias);
    compare_tree_components(
        post_infinite,pre,3.0e-12,3.0e-8,
        "R-infinity post/pre identity");

    const auto finite_bases=native::
        compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(
            power,transfer,vectors,config);
    const double basis=finite_bases.gaussian_linear;
    const double b1=bias.b1;
    const double bphi_recon=bias.bphi;
    const double step=1.0e-5;
    const double finite_difference=
        (
            finite_adaptive_brec_gaussian_tree_piece(
                power,transfer,vectors,config,
                b1,bphi_recon,step)
            -finite_adaptive_brec_gaussian_tree_piece(
                power,transfer,vectors,config,
                b1,bphi_recon,-step)
        )/(2.0*step);
    const double expected=
        b1*b1*b1*b1*bphi_recon/config.bias_recon*basis;
    require_close(
        finite_difference,expected,5.0e-9,3.0e-8,
        "adaptive b_rec denominator analytic/finite difference");

    const double curvature_step=5.0e-3;
    const double zero_gaussian=
        finite_adaptive_brec_gaussian_tree_piece(
            power,transfer,vectors,config,
            b1,bphi_recon,0.0);
    const auto gaussian_quadratic_at_step=
        [&](double local_step) {
            return
        (
            finite_adaptive_brec_gaussian_tree_piece(
                power,transfer,vectors,config,
                b1,bphi_recon,local_step)
            +finite_adaptive_brec_gaussian_tree_piece(
                power,transfer,vectors,config,
                b1,bphi_recon,-local_step)
            -2.0*zero_gaussian
        )/(2.0*local_step*local_step);
        };
    const double gaussian_quadratic_coarse=
        gaussian_quadratic_at_step(curvature_step);
    const double gaussian_quadratic_fine=
        gaussian_quadratic_at_step(0.5*curvature_step);
    const double gaussian_quadratic_finite=
        (4.0*gaussian_quadratic_fine-gaussian_quadratic_coarse)/3.0;
    const double gaussian_quadratic_expected=
        b1*b1*b1*b1
        *std::pow(bphi_recon/config.bias_recon,2)
        *finite_bases.gaussian_quadratic;
    require_close(
        gaussian_quadratic_finite,gaussian_quadratic_expected,
        2.0e-7,3.0e-6,
        "adaptive b_rec Gaussian quadratic analytic/finite difference");

    const double tracer_bphi=bias.bphi;
    const double png_cross_finite=
        (
            finite_adaptive_brec_png_linear_reconstruction_piece(
                power,transfer,vectors,config,
                b1,tracer_bphi,bphi_recon,step)
            -finite_adaptive_brec_png_linear_reconstruction_piece(
                power,transfer,vectors,config,
                b1,tracer_bphi,bphi_recon,-step)
        )/(2.0*step);
    const double png_cross_expected=
        b1*b1*b1*tracer_bphi
        *(bphi_recon/config.bias_recon)
        *finite_bases.png_linear_cross;
    require_close(
        png_cross_finite,png_cross_expected,5.0e-9,3.0e-8,
        "adaptive b_rec PNG-linear cross analytic/finite difference");

    const double rotated_basis=native::
        compute_post_recon_halo_local_png_brec_denominator_tree_basis_vectors(
            power,transfer,rotated,config);
    require(
        std::fabs(rotated_basis-basis)
            >1.0e-10*std::max(std::fabs(basis),1.0),
        "adaptive b_rec basis retains CIC orientation dependence");
    const double infinite_basis=native::
        compute_post_recon_halo_local_png_brec_denominator_tree_basis_vectors(
            power,transfer,rotated,infinite);
    require_close(
        infinite_basis,0.0,0.0,1.0e-20,
        "adaptive b_rec basis vanishes for R infinity");
}

native::ComponentResult matter_response(
    const SmoothPower& power,
    const SmoothTransfer& transfer,
    const native::ClosedTriangleVectors& vectors,
    double lambda,
    double png_ir_cutoff=0.0,
    bool multicenter_qmc=true,
    int qmc_power=9,
    int qmc_replicates=4) {
    auto config=post_config();
    config.bias_recon=
        lambda==0.0
            ?std::numeric_limits<double>::infinity()
            :1.0/lambda;
    config.qmin=0.012;
    config.qmax=0.015;
    config.png_ir_cutoff=png_ir_cutoff;
    config.png_linear_multicenter_qmc=multicenter_qmc;
    config.png_b112ii_qmc_power=qmc_power;
    config.png_b112ii_qmc_replicates=qmc_replicates;
    config.epsrel=0.001;
    config.p13_epsrel=0.001;
    config.epsabs=1.0e-8;
    return native::compute_post_recon_local_png_1loop_dfNL_vectors(
        power,transfer,vectors,config,1.0);
}

void require_stats_agree(
    double qmc_value,
    const native::ComponentStats& qmc_stats,
    double adaptive_value,
    const native::ComponentStats& adaptive_stats,
    const std::string& label);

void test_matter_response_r_infinity() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.041,0.057,-0.31};
    const auto vectors=canonical_vectors(triangle);
    const auto post=matter_response(
        power,transfer,vectors,0.0,0.0,false);
    auto config=post_config();
    config.qmin=0.012;
    config.qmax=0.015;
    config.png_ir_cutoff=0.0;
    config.png_linear_multicenter_qmc=false;
    config.epsrel=0.001;
    config.p13_epsrel=0.001;
    config.epsabs=1.0e-8;
    const auto pre=native::compute_pre_recon_local_png_1loop_dfNL(
        power,transfer,triangle,config,1.0);
    require_close(
        post.dBdfNL_local_tree,
        pre.dBdfNL_local_tree,
        3.0e-12,3.0e-8,
        "R-infinity matter response post/pre tree");
#define CHECK_PRE_POST_STATS(value,stats) \
    require_stats_agree( \
        post.value,post.stats,pre.value,pre.stats, \
        "R-infinity matter response post/pre " #value)
    CHECK_PRE_POST_STATS(
        dBdfNL_local_B122I,
        dBdfNL_local_B122I_stats);
    CHECK_PRE_POST_STATS(
        dBdfNL_local_B122II,
        dBdfNL_local_B122II_stats);
    CHECK_PRE_POST_STATS(
        dBdfNL_local_B113I,
        dBdfNL_local_B113I_stats);
    CHECK_PRE_POST_STATS(
        dBdfNL_local_B113II,
        dBdfNL_local_B113II_stats);
    CHECK_PRE_POST_STATS(
        dBdfNL_local_1loop,
        dBdfNL_local_1loop_stats);
    CHECK_PRE_POST_STATS(
        dBdfNL_local_total,
        dBdfNL_local_total_stats);
#undef CHECK_PRE_POST_STATS
}

void require_stats_agree(
    double qmc_value,
    const native::ComponentStats& qmc_stats,
    double adaptive_value,
    const native::ComponentStats& adaptive_stats,
    const std::string& label) {
    ++checks;
    const double difference=std::fabs(qmc_value-adaptive_value);
    const double scale=
        std::max(std::fabs(qmc_value),std::fabs(adaptive_value));
    const double allowance=
        8.0*(qmc_stats.abserr+adaptive_stats.abserr)
        +1.0e-3*scale
        +1.0e-5;
    if (difference>allowance) {
        std::ostringstream details;
        details.precision(17);
        details<<label
               <<": qmc="<<qmc_value
               <<", adaptive="<<adaptive_value
               <<", difference="<<difference
               <<", allowance="<<allowance
               <<", qmc_error="<<qmc_stats.abserr
               <<", adaptive_error="<<adaptive_stats.abserr;
        throw std::runtime_error(details.str());
    }
}

void test_multicenter_qmc_adaptive_benign() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.041,0.057,-0.31};
    const auto vectors=canonical_vectors(triangle);
    const auto qmc=matter_response(
        power,transfer,vectors,1.0,0.0,true,12,8);
    const auto adaptive=matter_response(
        power,transfer,vectors,1.0,0.0,false);
    require_stats_agree(
        qmc.dBdfNL_local_B122I,
        qmc.dBdfNL_local_B122I_stats,
        adaptive.dBdfNL_local_B122I,
        adaptive.dBdfNL_local_B122I_stats,
        "high-resolution QMC/adaptive B122I");
    require_stats_agree(
        qmc.dBdfNL_local_B122II,
        qmc.dBdfNL_local_B122II_stats,
        adaptive.dBdfNL_local_B122II,
        adaptive.dBdfNL_local_B122II_stats,
        "high-resolution QMC/adaptive B122II");
    require_stats_agree(
        qmc.dBdfNL_local_B113II,
        qmc.dBdfNL_local_B113II_stats,
        adaptive.dBdfNL_local_B113II,
        adaptive.dBdfNL_local_B113II_stats,
        "high-resolution QMC/adaptive B113II");
    require_stats_agree(
        qmc.dBdfNL_local_total,
        qmc.dBdfNL_local_total_stats,
        adaptive.dBdfNL_local_total,
        adaptive.dBdfNL_local_total_stats,
        "high-resolution QMC/adaptive total");
}

void test_local_png_finite_box_contract() {
    auto config=post_config();
    config.qmin=0.006;
    config.png_ir_cutoff=0.0;
    require_close(
        native::local_png_ir_cutoff(config),
        0.006,0.0,0.0,
        "zero PNG cutoff inherits qmin");
    config.png_ir_cutoff=0.008;
    require_close(
        native::local_png_ir_cutoff(config),
        0.008,0.0,0.0,
        "explicit PNG cutoff is registered");
    require(
        native::local_png_primordial_triplet_is_resolved(
            0.008,0.011,0.019,config),
        "all B0 legs at or above kf are resolved");
    require(
        !native::local_png_primordial_triplet_is_resolved(
            0.0079,0.011,0.019,config),
        "first unresolved B0 leg is rejected");
    require(
        !native::local_png_primordial_triplet_is_resolved(
            0.011,0.0079,0.019,config),
        "second unresolved B0 leg is rejected");
    require(
        !native::local_png_primordial_triplet_is_resolved(
            0.011,0.019,0.0079,config),
        "third unresolved B0 leg is rejected");
}

void test_multicenter_qmc_measure() {
    native::NativeConfig config;
    config.qmin=0.1;
    config.qmax=2.0;
    config.png_b112ii_qmc_power=15;
    config.png_b112ii_qmc_replicates=8;
    const double shifted_cutoff=0.2;
    const auto measured=
        native::audit_local_png_multicenter_qmc_excision_volume(
            std::array<double,3>{{1.0,0.0,0.0}},
            shifted_cutoff,
            config);
    const double pi=std::acos(-1.0);
    const double physical_volume=
        4.0*pi/3.0
        *(std::pow(config.qmax,3)
          -std::pow(config.qmin,3)
          -std::pow(shifted_cutoff,3));
    const double expected=
        physical_volume/std::pow(2.0*pi,3);
    require(
        std::isfinite(measured.value)
        &&std::isfinite(measured.abserr)
        &&measured.abserr>=0.0,
        "multicenter QMC analytic-volume audit is finite");
    require(
        std::fabs(measured.value-expected)
        <=8.0*measured.abserr+2.0e-5*expected,
        "multicenter QMC reproduces the shifted-excision volume");
    require(
        measured.neval==(1<<15)*2*8,
        "multicenter QMC reports the exact proposal evaluation count");
}

double quadratic_prediction(
    double y0,double y1,double y2,double lambda) {
    const double a2=0.5*(y2-2.0*y1+y0);
    const double a1=y1-y0-a2;
    return y0+a1*lambda+a2*lambda*lambda;
}

void test_matter_lambda_polynomial_and_cache() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.041,0.057,-0.31};
    const auto vectors=canonical_vectors(triangle);
    const auto rotated=rotated_vectors(vectors);
    const auto at_zero=matter_response(
        power,transfer,vectors,0.0);
    const auto at_one=matter_response(
        power,transfer,vectors,1.0);
    const auto at_two=matter_response(
        power,transfer,vectors,2.0);
    const double validation_lambda=1.37;
    const auto validation=matter_response(
        power,transfer,vectors,validation_lambda);

#define CHECK_QUADRATIC(name) \
    require_close( \
        validation.name, \
        quadratic_prediction( \
            at_zero.name,at_one.name,at_two.name,validation_lambda), \
        3.0e-11,1.0e-6, \
        "matter response lambda quadratic " #name)
    CHECK_QUADRATIC(dBdfNL_local_B122I);
    CHECK_QUADRATIC(dBdfNL_local_B122II);
    CHECK_QUADRATIC(dBdfNL_local_B113I);
    CHECK_QUADRATIC(dBdfNL_local_B113II);
    CHECK_QUADRATIC(dBdfNL_local_1loop);
    CHECK_QUADRATIC(dBdfNL_local_total);
#undef CHECK_QUADRATIC

    const auto first=matter_response(
        power,transfer,vectors,1.0);
    (void)matter_response(power,transfer,rotated,0.73);
    const auto repeated=matter_response(
        power,transfer,vectors,1.0);
#define CHECK_CACHE(name) \
    require( \
        first.name==repeated.name, \
        "A-B-A cache identity " #name)
    CHECK_CACHE(dBdfNL_local_B122I);
    CHECK_CACHE(dBdfNL_local_B122II);
    CHECK_CACHE(dBdfNL_local_B113I);
    CHECK_CACHE(dBdfNL_local_B113II);
    CHECK_CACHE(dBdfNL_local_1loop);
    CHECK_CACHE(dBdfNL_local_total);
#undef CHECK_CACHE

    require(
        first.P12_png_k1_stats.neval>0
        &&first.P12_png_k2_stats.neval>0
        &&first.P12_png_k3_stats.neval>0,
        "matter-linear QMC records positive P12 evaluation counts");
    require(
        first.dBdfNL_local_B122II_stats.neval>0
        &&first.dBdfNL_local_B113II_stats.neval>0,
        "matter-linear QMC records positive direct-loop counts");
    require(
        std::isfinite(first.dBdfNL_local_total_stats.abserr)
        &&first.dBdfNL_local_total_stats.abserr>=0.0,
        "matter-linear QMC records a finite total error estimate");
    require_close(
        first.dBdfNL_local_total_stats.value,
        first.dBdfNL_local_total,
        0.0,0.0,
        "matter-linear total ComponentStats value closes");

    const auto adaptive=matter_response(
        power,transfer,vectors,1.0,0.0,false);
    require_close(
        first.dBdfNL_local_total,
        adaptive.dBdfNL_local_total,
        2.5e-1,5.0,
        "multicenter QMC agrees with independent adaptive response");

    const auto finite_box_tighter=matter_response(
        power,transfer,vectors,1.0,0.014);
    require(
        std::fabs(
            finite_box_tighter.dBdfNL_local_B122I
            -first.dBdfNL_local_B122I)
        >1.0e-12*std::max(
            1.0,
            std::fabs(first.dBdfNL_local_B122I)),
        "matter P12/B122I responds to the primordial-mode cutoff");
    require(
        std::fabs(
            finite_box_tighter.dBdfNL_local_B122II
            -first.dBdfNL_local_B122II)
        >1.0e-12*std::max(
            1.0,
            std::fabs(first.dBdfNL_local_B122II)),
        "matter B122II responds to the primordial-mode cutoff");
    require(
        std::fabs(
            finite_box_tighter.dBdfNL_local_B113II
            -first.dBdfNL_local_B113II)
        >1.0e-12*std::max(
            1.0,
            std::fabs(first.dBdfNL_local_B113II)),
        "matter B113II responds to the primordial-mode cutoff");
    const auto first_again=matter_response(
        power,transfer,vectors,1.0);
    require(
        first_again.dBdfNL_local_B122I
            ==first.dBdfNL_local_B122I,
        "P12 cache key includes and restores the PNG cutoff");
}

double b112_coefficient(
    const SmoothPower& power,
    const SmoothTransfer& transfer,
    const native::ClosedTriangleVectors& vectors,
    double lambda) {
    auto config=post_config();
    config.bias_recon=
        lambda==0.0
            ?std::numeric_limits<double>::infinity()
            :1.0/lambda;
    config.qmin=0.008;
    config.qmax=0.08;
    config.png_ir_cutoff=0.008;
    config.png_b112ii_multicenter_qmc=true;
    config.png_b112ii_qmc_power=6;
    config.png_b112ii_qmc_replicates=2;
    return native::
        compute_post_recon_local_png_B112II_fNL2_coefficient_vectors(
            power,transfer,vectors,config)
        .B112II_fNL2_coefficient;
}

void test_b112_r_infinity() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.043,0.061,-0.22};
    const auto vectors=canonical_vectors(triangle);
    auto config=post_config();
    config.bias_recon=std::numeric_limits<double>::infinity();
    config.qmin=0.008;
    config.qmax=0.08;
    config.png_ir_cutoff=0.008;
    config.png_b112ii_multicenter_qmc=true;
    config.png_b112ii_qmc_power=6;
    config.png_b112ii_qmc_replicates=2;
    const auto post=native::
        compute_post_recon_local_png_B112II_fNL2_coefficient_vectors(
            power,transfer,vectors,config);
    const auto pre=native::
        compute_pre_recon_local_png_B112II_fNL2_coefficient(
            power,transfer,triangle,config);
    require_stats_agree(
        post.B112II_fNL2_coefficient,
        post.B112II_fNL2_coefficient_stats,
        pre.B112II_fNL2_coefficient,
        pre.B112II_fNL2_coefficient_stats,
        "R-infinity B112 coefficient post/pre");
}

void test_b112_lambda_linearity() {
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.043,0.061,-0.22};
    const auto vectors=canonical_vectors(triangle);
    const double at_zero=b112_coefficient(
        power,transfer,vectors,0.0);
    const double at_one=b112_coefficient(
        power,transfer,vectors,1.0);
    const double lambda=1.37;
    const double validation=b112_coefficient(
        power,transfer,vectors,lambda);
    const double prediction=
        at_zero+lambda*(at_one-at_zero);
    require_close(
        validation,prediction,3.0e-12,3.0e-7,
        "post B112 coefficient is linear in lambda");
}

void test_nonlocal_primordial_templates() {
    using Shape=native::PrimordialBispectrumShape;
    require(
        native::parse_primordial_bispectrum_shape("EQ")
            ==Shape::Equilateral,
        "EQ shape alias parses");
    require(
        native::parse_primordial_bispectrum_shape("OR_LSS")
            ==Shape::OrthogonalLSS,
        "OR_LSS shape alias parses");
    require(
        std::string(native::primordial_bispectrum_shape_name(
            Shape::OrthogonalLSS))=="orthogonal-LSS",
        "OR-LSS canonical shape name");

    const double equal_power=2.7;
    const double expected_equal=6.0*equal_power*equal_power;
    for (const auto shape:{
             Shape::Local,Shape::Equilateral,Shape::OrthogonalLSS}) {
        require_close(
            native::primordial_potential_bispectrum_dfNL(
                shape,0.07,0.07,0.07,
                equal_power,equal_power,equal_power),
            expected_equal,2.0e-13,2.0e-13,
            std::string("equilateral normalization ")
                +native::primordial_bispectrum_shape_name(shape));
    }

    const std::array<double,3> k{{0.04,0.07,0.09}};
    const std::array<double,3> pphi{{3.1,1.7,0.9}};
    const int permutations[6][3]={
        {0,1,2},{0,2,1},{1,0,2},
        {1,2,0},{2,0,1},{2,1,0}};
    for (const auto shape:{Shape::Equilateral,Shape::OrthogonalLSS}) {
        const double reference=
            native::primordial_potential_bispectrum_dfNL(
                shape,k[0],k[1],k[2],
                pphi[0],pphi[1],pphi[2]);
        for (const auto& permutation:permutations) {
            require_close(
                native::primordial_potential_bispectrum_dfNL(
                    shape,
                    k[permutation[0]],
                    k[permutation[1]],
                    k[permutation[2]],
                    pphi[permutation[0]],
                    pphi[permutation[1]],
                    pphi[permutation[2]]),
                reference,3.0e-13,3.0e-13,
                std::string("permutation symmetry ")
                    +native::primordial_bispectrum_shape_name(shape));
        }
    }

    const double squeezed=0.01;
    const double squeezed_power=std::pow(squeezed,-3.0);
    const double local=native::primordial_potential_bispectrum_dfNL(
        Shape::Local,squeezed,1.0,1.0,squeezed_power,1.0,1.0);
    const double equilateral=
        native::primordial_potential_bispectrum_dfNL(
            Shape::Equilateral,
            squeezed,1.0,1.0,squeezed_power,1.0,1.0);
    const double orthogonal_lss=
        native::primordial_potential_bispectrum_dfNL(
            Shape::OrthogonalLSS,
            squeezed,1.0,1.0,squeezed_power,1.0,1.0);
    require_close(local,4000002.0,2.0e-13,2.0e-9,
                  "local squeezed reference");
    require_close(equilateral,1194.0,2.0e-12,2.0e-9,
                  "equilateral squeezed reference");
    require_close(orthogonal_lss,-661.0771879786622,
                  5.0e-9,5.0e-6,
                  "OR-LSS squeezed reference and sign");
    require(
        std::fabs(equilateral/local)<3.0e-4
        &&std::fabs(orthogonal_lss/local)<2.0e-4,
        "nonlocal templates have the expected squeezed suppression");

    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.057,0.057,-0.5};
    const double b1=1.9326526728794324;
    const double legacy=native::local_png_tree_dfNL(
        power,transfer,triangle,b1);
    const double generic_local=native::primordial_png_tree_dfNL(
        Shape::Local,power,transfer,triangle,b1);
    const double generic_eq=native::primordial_png_tree_dfNL(
        Shape::Equilateral,power,transfer,triangle,b1);
    const double generic_orlss=native::primordial_png_tree_dfNL(
        Shape::OrthogonalLSS,power,transfer,triangle,b1);
    require_close(generic_local,legacy,0.0,0.0,
                  "generic local tree is bitwise legacy-compatible");
    require_close(generic_eq,legacy,2.0e-13,2.0e-9,
                  "EQ matter tree equilateral normalization");
    require_close(generic_orlss,legacy,2.0e-12,2.0e-9,
                  "OR-LSS matter tree equilateral normalization");
}

native::HaloDelta2PNGBiasParams delta2_halo_bias() {
    native::HaloDelta2PNGBiasParams bias;
    bias.b1=1.93;
    bias.b2=-0.42;
    bias.bK2=-0.27;
    bias.bPsi=-0.016;
    bias.bPsiDelta=0.031;
    bias.bPsiS=-0.024;
    bias.k_pivot=0.05;
    return bias;
}

double norm(const std::array<double,3>& vector) {
    return std::sqrt(dot(vector,vector));
}

double fixed_reconstruction_shift(
    const std::array<double,3>& total,
    const std::array<double,3>& source,
    const native::NativeConfig& config) {
    const double source2=dot(source,source);
    const double k=std::sqrt(source2);
    const double gaussian=std::exp(
        -0.5*source2*config.smoothing_radius*config.smoothing_radius);
    const double half_cell=0.5*config.recon_cellsize;
    const double cic=
        config.recon_cic_window_power<=0
        ?1.0
        :std::pow(
            sinc(source[0]*half_cell)
            *sinc(source[1]*half_cell)
            *sinc(source[2]*half_cell),
            config.recon_cic_window_power);
    (void)k;
    return -dot(total,source)/source2
           *gaussian*cic/config.bias_recon;
}

/*
 * 独立 finite-fNL evaluator，只用于核对新 Delta=2 native product rule。
 * 它不调用待测的 native tree API，也不加入 primordial B111；中心差分应等于
 * native deterministic total 减去 primordial 分量。
 */
double finite_delta2_gaussian_pair_tree(
    const SmoothPower& power,
    const SmoothTransfer& transfer,
    const native::ClosedTriangleVectors& vectors,
    const native::HaloDelta2PNGBiasParams& bias,
    double fnl,
    const native::NativeConfig* config) {
    const int pairs[3][2]={{0,1},{1,2},{2,0}};
    double result=0.0;
    for (const auto& pair:pairs) {
        const int i=pair[0];
        const int j=pair[1];
        const double ki=norm(vectors[i]);
        const double kj=norm(vectors[j]);
        const double mu=dot(vectors[i],vectors[j])/(ki*kj);
        const double mi=transfer(ki);
        const double mj=transfer(kj);
        const double si=(ki/bias.k_pivot)*(ki/bias.k_pivot)/mi;
        const double sj=(kj/bias.k_pivot)*(kj/bias.k_pivot)/mj;
        const double response_i=bias.bPsi*si;
        const double response_j=bias.bPsi*sj;
        const double k1i=bias.b1+fnl*response_i;
        const double k1j=bias.b1+fnl*response_j;
        const double f2=native::F2eds(ki,kj,mu);
        const double tidal=mu*mu-1.0/3.0;
        const double p2=0.5*(3.0*mu*mu-1.0);
        const double k2_gaussian=
            bias.b1*f2+0.5*bias.b2+bias.bK2*tidal;
        const double k2_response=
            0.5*bias.bPsi*mu*(ki/kj*si+kj/ki*sj)
            +0.5*bias.bPsiDelta*(si+sj)
            +0.5*bias.bPsiS*p2*(si+sj);
        double k2=k2_gaussian+fnl*k2_response;
        if (config!=nullptr) {
            const auto total=add(vectors[i],vectors[j]);
            const double shift_sum=
                fixed_reconstruction_shift(total,vectors[i],*config)
                +fixed_reconstruction_shift(total,vectors[j],*config);
            k2+=0.5*k1i*k1j*shift_sum;
        }
        result+=2.0*k1i*k1j*k2*power(ki)*power(kj);
    }
    return result;
}

void require_delta2_components_close(
    const native::ComponentResult& actual,
    const native::ComponentResult& expected,
    double relative,
    double absolute,
    const std::string& label) {
#define CHECK_DELTA2_FIELD(name) \
    require_close( \
        actual.name,expected.name,relative,absolute, \
        label+" " #name)
    CHECK_DELTA2_FIELD(Btree);
    CHECK_DELTA2_FIELD(dBdfNL_halo_png_tree);
    CHECK_DELTA2_FIELD(dBdfNL_halo_png_primordial);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsi_f2);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsi_advection);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsiDelta);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsi_b2);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsi_bK2);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsiS);
    CHECK_DELTA2_FIELD(dBdfNL_halo_bPsi_reconstruction);
    CHECK_DELTA2_FIELD(dBdfNL_halo_stochastic_PsiP_basis);
    CHECK_DELTA2_FIELD(dBdfNL_halo_stochastic_P_basis);
    CHECK_DELTA2_FIELD(dBdfNL_halo_stochastic_constant_basis);
#undef CHECK_DELTA2_FIELD
}

void test_delta2_halo_tree_response() {
    using Shape=native::PrimordialBispectrumShape;
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.031,0.052,-0.37};
    const auto vectors=canonical_vectors(triangle);
    const auto bias=delta2_halo_bias();
    auto config=post_config();
    config.smoothing_radius=10.0;
    config.bias_recon=1.9326526728794324;
    config.recon_cellsize=8.0;
    config.recon_cic_window_power=2;

    const auto pre=native::compute_pre_recon_halo_delta2_png_tree_dfNL_vectors(
        Shape::Equilateral,power,transfer,vectors,bias);
    const auto post=native::compute_post_recon_halo_delta2_png_tree_dfNL_vectors(
        Shape::Equilateral,power,transfer,vectors,config,bias);
    require(
        pre.png_shape==Shape::Equilateral
        &&post.png_shape==Shape::Equilateral,
        "Delta=2 halo tree records EQ shape");
    require_close(
        pre.dBdfNL_halo_png_tree,
        pre.dBdfNL_halo_png_primordial
            +pre.dBdfNL_halo_bPsi_f2
            +pre.dBdfNL_halo_bPsi_advection
            +pre.dBdfNL_halo_bPsiDelta
            +pre.dBdfNL_halo_bPsi_b2
            +pre.dBdfNL_halo_bPsi_bK2
            +pre.dBdfNL_halo_bPsiS
            +pre.dBdfNL_halo_bPsi_reconstruction,
        2.0e-15,2.0e-10,
        "pre Delta=2 named-component closure");
    require_close(
        post.dBdfNL_halo_png_tree,
        post.dBdfNL_halo_png_primordial
            +post.dBdfNL_halo_bPsi_f2
            +post.dBdfNL_halo_bPsi_advection
            +post.dBdfNL_halo_bPsiDelta
            +post.dBdfNL_halo_bPsi_b2
            +post.dBdfNL_halo_bPsi_bK2
            +post.dBdfNL_halo_bPsiS
            +post.dBdfNL_halo_bPsi_reconstruction,
        2.0e-15,2.0e-10,
        "post Delta=2 named-component closure");

    const double step=2.0e-4;
    const double pre_finite=
        (finite_delta2_gaussian_pair_tree(
             power,transfer,vectors,bias,step,nullptr)
         -finite_delta2_gaussian_pair_tree(
             power,transfer,vectors,bias,-step,nullptr))
        /(2.0*step);
    const double post_finite=
        (finite_delta2_gaussian_pair_tree(
             power,transfer,vectors,bias,step,&config)
         -finite_delta2_gaussian_pair_tree(
             power,transfer,vectors,bias,-step,&config))
        /(2.0*step);
    require_close(
        pre.dBdfNL_halo_png_tree-pre.dBdfNL_halo_png_primordial,
        pre_finite,5.0e-10,3.0e-7,
        "pre Delta=2 analytic/finite product rule");
    require_close(
        post.dBdfNL_halo_png_tree-post.dBdfNL_halo_png_primordial,
        post_finite,5.0e-10,3.0e-7,
        "post Delta=2 analytic/finite product rule");

    auto infinite=config;
    infinite.bias_recon=std::numeric_limits<double>::infinity();
    const auto post_infinite=
        native::compute_post_recon_halo_delta2_png_tree_dfNL_vectors(
            Shape::Equilateral,power,transfer,vectors,infinite,bias);
    require_delta2_components_close(
        post_infinite,pre,3.0e-13,3.0e-9,
        "Delta=2 R-infinity post/pre identity");

    auto pure=bias;
    pure.bPsi=0.0;
    pure.bPsiDelta=0.0;
    pure.bPsiS=0.0;
    const auto pure_pre=
        native::compute_pre_recon_halo_delta2_png_tree_dfNL_vectors(
            Shape::Equilateral,power,transfer,vectors,pure);
    const auto pure_post=
        native::compute_post_recon_halo_delta2_png_tree_dfNL_vectors(
            Shape::Equilateral,power,transfer,vectors,config,pure);
    require_close(
        pure_pre.dBdfNL_halo_png_tree,
        pure_pre.dBdfNL_halo_png_primordial,
        0.0,0.0,"pure EQ halo tree leaves only B111 pre");
    require_close(
        pure_post.dBdfNL_halo_png_tree,
        pure_post.dBdfNL_halo_png_primordial,
        0.0,0.0,"pure EQ halo tree leaves only B111 post");
    require_close(
        pure_post.dBdfNL_halo_png_primordial,
        pure_pre.dBdfNL_halo_png_primordial,
        0.0,0.0,"EQ B111 is bitwise pre/post invariant");

    auto doubled_pivot=bias;
    doubled_pivot.k_pivot=2.0*bias.k_pivot;
    const auto pivot_result=
        native::compute_pre_recon_halo_delta2_png_tree_dfNL_vectors(
            Shape::Equilateral,power,transfer,vectors,doubled_pivot);
    require_close(
        pivot_result.dBdfNL_halo_png_tree
            -pivot_result.dBdfNL_halo_png_primordial,
        0.25*(pre.dBdfNL_halo_png_tree
              -pre.dBdfNL_halo_png_primordial),
        5.0e-14,3.0e-10,
        "Delta=2 response follows the frozen k-pivot convention");

    bool local_rejected=false;
    try {
        (void)native::compute_pre_recon_halo_delta2_png_tree_dfNL_vectors(
            Shape::Local,power,transfer,vectors,bias);
    } catch (const std::invalid_argument&) {
        local_rejected=true;
    }
    require(local_rejected,"Delta=2 halo API rejects local PNG");
}

void test_nonlocal_matter_loop_components() {
    using Shape=native::PrimordialBispectrumShape;
    const SmoothPower power;
    const SmoothTransfer transfer;
    const native::Triangle triangle{0.041,0.057,-0.31};
    const auto vectors=canonical_vectors(triangle);
    auto config=post_config();
    config.qmin=0.012;
    config.qmax=0.015;
    config.png_ir_cutoff=0.012;
    config.png_linear_multicenter_qmc=false;
    config.epsrel=0.001;
    config.p13_epsrel=0.001;
    config.epsabs=1.0e-8;
    config.bias_recon=std::numeric_limits<double>::infinity();

    for (const auto shape:{Shape::Equilateral,Shape::OrthogonalLSS}) {
        const auto pre=native::compute_pre_recon_png_1loop_dfNL(
            shape,power,transfer,triangle,config,1.0);
        const auto post=native::compute_post_recon_png_1loop_dfNL_vectors(
            shape,power,transfer,vectors,config,1.0);
        const std::string label=
            native::primordial_bispectrum_shape_name(shape);
        require(
            pre.png_shape==shape &&post.png_shape==shape,
            label+" component result records its shape");
        require_close(
            post.dBdfNL_png_tree,pre.dBdfNL_png_tree,
            3.0e-12,3.0e-8,
            label+" R-infinity post/pre B111");
#define CHECK_NONLOCAL_PRE_POST(value,stats) \
        require_stats_agree( \
            post.value,post.stats,pre.value,pre.stats, \
            label+" R-infinity post/pre " #value)
        CHECK_NONLOCAL_PRE_POST(
            dBdfNL_png_B122I,dBdfNL_png_B122I_stats);
        CHECK_NONLOCAL_PRE_POST(
            dBdfNL_png_B122II,dBdfNL_png_B122II_stats);
        CHECK_NONLOCAL_PRE_POST(
            dBdfNL_png_B113I,dBdfNL_png_B113I_stats);
        CHECK_NONLOCAL_PRE_POST(
            dBdfNL_png_B113II,dBdfNL_png_B113II_stats);
        CHECK_NONLOCAL_PRE_POST(
            dBdfNL_png_total,dBdfNL_png_total_stats);
#undef CHECK_NONLOCAL_PRE_POST
        require_close(
            pre.dBdfNL_png_1loop,
            pre.dBdfNL_png_B122I+pre.dBdfNL_png_B122II
                +pre.dBdfNL_png_B113I+pre.dBdfNL_png_B113II,
            2.0e-15,2.0e-10,
            label+" four-topology one-loop closure");
        require_close(
            pre.dBdfNL_png_total,
            pre.dBdfNL_png_tree+pre.dBdfNL_png_1loop,
            2.0e-15,2.0e-10,
            label+" B111 plus one-loop closure");
        require(
            std::isfinite(pre.dBdfNL_png_total)
            &&pre.dBdfNL_png_B122I_stats.neval>0
            &&pre.dBdfNL_png_B122II_stats.neval>0
            &&pre.dBdfNL_png_B113I_stats.neval>0
            &&pre.dBdfNL_png_B113II_stats.neval>0,
            label+" exports finite components and evaluation counts");
    }

    const auto eq_first=native::compute_pre_recon_png_1loop_dfNL(
        Shape::Equilateral,power,transfer,triangle,config,1.0);
    (void)native::compute_pre_recon_png_1loop_dfNL(
        Shape::OrthogonalLSS,power,transfer,triangle,config,1.0);
    const auto eq_repeated=native::compute_pre_recon_png_1loop_dfNL(
        Shape::Equilateral,power,transfer,triangle,config,1.0);
    require(
        eq_first.dBdfNL_png_total==eq_repeated.dBdfNL_png_total,
        "shape-aware P12 cache preserves EQ A-B-A identity");

    const auto generic_local=native::compute_pre_recon_png_1loop_dfNL(
        Shape::Local,power,transfer,triangle,config,1.0);
    const auto legacy_local=native::compute_pre_recon_local_png_1loop_dfNL(
        power,transfer,triangle,config,1.0);
#define CHECK_GENERIC_LOCAL(name) \
    require( \
        generic_local.dBdfNL_png_##name \
            ==legacy_local.dBdfNL_local_##name, \
        "generic/legacy local bitwise identity " #name)
    CHECK_GENERIC_LOCAL(tree);
    CHECK_GENERIC_LOCAL(B122I);
    CHECK_GENERIC_LOCAL(B122II);
    CHECK_GENERIC_LOCAL(B113I);
    CHECK_GENERIC_LOCAL(B113II);
    CHECK_GENERIC_LOCAL(1loop);
    CHECK_GENERIC_LOCAL(total);
#undef CHECK_GENERIC_LOCAL
}

}  // namespace

int main() {
    try {
        test_closed_vector_contract();
        test_tree_orientation_and_limits();
        test_matter_response_r_infinity();
        test_local_png_finite_box_contract();
        test_multicenter_qmc_measure();
        test_multicenter_qmc_adaptive_benign();
        test_matter_lambda_polynomial_and_cache();
        test_b112_r_infinity();
        test_b112_lambda_linearity();
        test_nonlocal_primordial_templates();
        test_delta2_halo_tree_response();
        test_nonlocal_matter_loop_components();
        std::cout
            <<"post-R1 finite-PNG native checks passed: "
            <<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            <<"post-R1 finite-PNG native test failed: "
            <<error.what()<<"\n";
        return 1;
    }
}
