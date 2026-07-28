#include "shell_projector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "PowerSpectrum.h"

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

void require_finite_polynomial(
    const eft::SparsePolynomial& value,
    const std::string& message) {
    require(!value.terms().empty(),message+" is non-empty");
    for (const auto& term:value.terms()) {
        require(std::isfinite(term.second),
                message+" finite "+term.first.canonical_string());
    }
}

class GaussianPower final:public PowerSpectrum {
public:
    real Evaluate(real k) const override { return 900.0*std::exp(-2.0*k*k); }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("GaussianPower has no cosmology");
    }
};

void test_shell_average() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    const shell::ShellBin bin{0.03,0.05,0.05,0.07};
    shell::ShellQuadratureConfig shell_config;
    shell_config.n_radial=1;
    shell_config.n_internal_mu=2;
    shell_config.average_grid_orientation=false;
    shell_config.radial_measure=shell::ShellRadialMeasure::ContinuumVolume;
    eft::DirectIntegrationConfig loop;
    loop.qmin=1.0e-3; loop.qmax=0.20;
    loop.n_radial=2; loop.n_mu=4; loop.n_phi=4;
    loop.ir_safe=true; loop.uv_subtract=true;
    eft::StochasticIntegrationConfig stochastic;
    stochastic.qmin=1.0e-3; stochastic.qmax=0.40;
    stochastic.n_radial=2; stochastic.n_mu=4;
    const auto projected=eft::compute_eft_shell_templates(
        power,power,bin,provider,loop,shell_config,stochastic,0.30);
    const auto nodes=shell::make_shell_nodes(bin,shell_config);
    require(projected.shell_nodes==nodes.size(),"EFT shell node count");
    double weight_sum=0.0;
    std::array<double,eft::kParameterCount> parameters={};
    parameters[static_cast<std::size_t>(eft::ParameterId::B1)]=2.2;
    parameters[static_cast<std::size_t>(eft::ParameterId::B2)]=0.3;
    parameters[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.15;
    parameters[static_cast<std::size_t>(eft::ParameterId::B3)]=0.1;
    parameters[static_cast<std::size_t>(eft::ParameterId::BNabla2Delta)]=0.4;
    parameters[static_cast<std::size_t>(eft::ParameterId::AshotResidual)]=-0.2;
    parameters[static_cast<std::size_t>(eft::ParameterId::BshotResidual)]=0.3;
    double tree=0.0,loop_total=0.0,counterterm=0.0,stochastic_value=0.0;
    for (const auto& node:nodes) {
        weight_sum+=node.weight;
        const auto diagrams=eft::evaluate_direct(power,node.closed_vectors,provider,loop);
        std::array<double,3> external_power={};
        for (int leg=0;leg<3;++leg) {
            external_power[leg]=power(marisa_b_halo_v1::norm(node.closed_vectors[leg]));
        }
        const auto counter=eft::counterterm_bispectrum(
            node.closed_vectors,external_power,0.30);
        const auto stoch=eft::stochastic_bispectrum(
            power,node.closed_vectors,0.30,stochastic);
        tree+=node.weight*diagrams.tree.evaluate(parameters);
        loop_total+=node.weight*diagrams.one_loop.evaluate(parameters);
        counterterm+=node.weight*counter.evaluate(parameters);
        stochastic_value+=node.weight*stoch.evaluate(parameters,2.0e-4);
    }
    require_close(weight_sum,1.0,0.0,2.0e-15,"shell weights normalized");
    const auto value=eft::evaluate_eft_shell_templates(
        projected,parameters,2.0e-4);
    require_close(value.tree,tree,2.0e-12,1.0e-8,"shell tree projection");
    require_close(value.renormalized_loop,loop_total,2.0e-10,1.0e-8,
                  "shell loop projection");
    require_close(value.counterterm,counterterm,2.0e-12,1.0e-8,
                  "shell counterterm projection");
    require_close(value.stochastic,stochastic_value,2.0e-10,1.0e-6,
                  "shell stochastic projection");
    require_close(value.total,tree+loop_total+counterterm+stochastic_value,
                  2.0e-10,1.0e-6,"shell total recomposition");
    require(!projected.fft_lattice_radial_measure,"continuum shell metadata");
}

void test_hybrid_shell_average() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    const shell::ShellBin bin{0.03,0.05,0.05,0.07};
    shell::ShellQuadratureConfig shell_config;
    shell_config.n_radial=1;
    shell_config.n_internal_mu=2;
    shell_config.average_grid_orientation=false;
    shell_config.radial_measure=shell::ShellRadialMeasure::ContinuumVolume;

    eft::HybridShellIntegrationConfig hybrid;
    hybrid.convolution.qmin=1.0e-3;
    hybrid.convolution.qmax=0.20;
    hybrid.convolution.n_radial=2;
    hybrid.convolution.n_mu=4;
    hybrid.convolution.n_phi=4;
    hybrid.convolution.ir_safe=true;
    hybrid.convolution.uv_subtract=true;
    hybrid.factorized_p13.qmin=hybrid.convolution.qmin;
    hybrid.factorized_p13.qmax=hybrid.convolution.qmax;
    hybrid.factorized_p13.n_radial=2;
    hybrid.factorized_p13.n_mu=4;
    hybrid.analytic.regulator.coarse_epsilon=1.0e-8;
    hybrid.analytic.regulator.refinement_ratio=10.0;
    hybrid.analytic.uv_restoration.enabled=false;

    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-3;
    fftlog.kmax=2.0;
    fftlog.frequency_count=8;
    fftlog.reconstruction_grid_size=32;
    fftlog.bias_nu=-0.3;
    fftlog.reconstruct_interpolated_power=false;
    fftlog.endpoint_inclusive_sampling=true;
    const eft::FftlogDrOracle b411_oracle(power,fftlog);

    eft::StochasticIntegrationConfig stochastic;
    stochastic.qmin=1.0e-3;
    stochastic.qmax=0.40;
    stochastic.n_radial=2;
    stochastic.n_mu=4;
    const auto projected=eft::compute_eft_shell_templates_hybrid(
        power,power,bin,provider,b411_oracle,hybrid,
        shell_config,stochastic,0.30);

    require(projected.hybrid_factorized_analytic_tadpoles,
            "hybrid shell metadata flag");
    require(projected.cached_p13_nodes>0,
            "hybrid shell records cached P13 nodes");
    require(projected.total_loop_nodes>projected.cached_p13_nodes,
            "hybrid shell records convolution and analytic work");
    require_finite_polynomial(
        projected.diagrams.B321II,"hybrid factorized B321II");
    require_finite_polynomial(
        projected.diagrams.B411,"hybrid analytic B411");
    require_polynomial_close(
        projected.diagrams.B321II,
        projected.diagrams.B321II_bare
            -projected.diagrams.B321II_uv_subtraction
            +projected.diagrams.B321II_uv_restoration,
        2.0e-13,1.0e-8,
        "hybrid B321II bare-minus-subtraction-plus-tail");
    require_polynomial_close(
        projected.diagrams.B411,
        projected.diagrams.B411_bare
            +projected.diagrams.B411_uv_restoration,
        2.0e-13,1.0e-8,
        "hybrid B411 DR-plus-restoration");
    require_polynomial_close(
        projected.diagrams.one_loop,
        projected.diagrams.B222+projected.diagrams.B321I
            +projected.diagrams.B321II+projected.diagrams.B411,
        2.0e-13,1.0e-8,
        "hybrid one-loop topology recomposition");

    std::array<double,eft::kParameterCount> parameters={};
    parameters[static_cast<std::size_t>(eft::ParameterId::B1)]=2.0;
    parameters[static_cast<std::size_t>(eft::ParameterId::B2)]=0.2;
    parameters[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.1;
    parameters[static_cast<std::size_t>(eft::ParameterId::B3)]=0.05;
    const auto evaluated=eft::evaluate_eft_shell_templates(
        projected,parameters,2.0e-4);
    require(std::isfinite(evaluated.total),
            "hybrid evaluated shell is finite");
    require_close(
        evaluated.renormalized_loop,
        evaluated.B222+evaluated.B321I
            +evaluated.B321II+evaluated.B411,
        2.0e-13,1.0e-8,
        "hybrid evaluated loop recomposition");

    hybrid.analytic_convolutions=true;
    hybrid.analytic.contour_sectors.enabled=true;
    const auto analytic_projected=
        eft::compute_eft_shell_templates_hybrid(
            power,power,bin,provider,b411_oracle,
            hybrid,shell_config,stochastic,0.30);
    require(
        analytic_projected.fftlog_analytic_convolutions,
        "analytic hybrid shell metadata flag");
    require(
        !analytic_projected.diagrams.ir_safe,
        "analytic hybrid shell does not claim pointwise IR-safe remapping");
    require_finite_polynomial(
        analytic_projected.diagrams.B222,
        "analytic hybrid B222");
    require_finite_polynomial(
        analytic_projected.diagrams.B321I,
        "analytic hybrid B321I");
    require_finite_polynomial(
        analytic_projected.diagrams.B411,
        "analytic hybrid B411");
    require_polynomial_close(
        analytic_projected.diagrams.one_loop,
        analytic_projected.diagrams.B222
            +analytic_projected.diagrams.B321I
            +analytic_projected.diagrams.B321II
            +analytic_projected.diagrams.B411,
        2.0e-13,1.0e-8,
        "analytic hybrid one-loop topology recomposition");
}

struct DirectLatticeMode {
    int nx=0;
    int ny=0;
    int nz=0;
    float radius=0.0F;
};

struct DirectLatticeShell {
    std::uint64_t full_modes=0;
    std::vector<DirectLatticeMode> nonzero_modes;
};

DirectLatticeShell enumerate_lattice_shell(
    double lower_bound,double upper_bound,
    double box_size,int mesh_size) {
    constexpr double two_pi=
        6.283185307179586476925286766559005768;
    const float fundamental=static_cast<float>(
        two_pi/static_cast<float>(box_size));
    const float lower=static_cast<float>(lower_bound);
    const float upper=static_cast<float>(upper_bound);
    const int negative_limit=-mesh_size/2;
    const int positive_limit=(mesh_size-1)/2;
    const int coordinate_limit=
        static_cast<int>(std::ceil(upper/fundamental))+1;
    const int minimum_mode=
        std::max(negative_limit,-coordinate_limit);
    const int maximum_mode=
        std::min(positive_limit,coordinate_limit);
    DirectLatticeShell result;
    for (int nx=minimum_mode;nx<=maximum_mode;++nx) {
        const float kx=static_cast<float>(nx)*fundamental;
        const float kx_squared=kx*kx;
        for (int ny=minimum_mode;ny<=maximum_mode;++ny) {
            const float ky=static_cast<float>(ny)*fundamental;
            const float transverse_squared=
                kx_squared+ky*ky;
            for (int nz=minimum_mode;nz<=maximum_mode;++nz) {
                const float kz=static_cast<float>(nz)*fundamental;
                const float radius=
                    std::sqrt(transverse_squared+kz*kz);
                if (radius>=lower && radius<upper) {
                    ++result.full_modes;
                    if (radius>0.0F) {
                        result.nonzero_modes.push_back(
                            DirectLatticeMode{
                                nx,ny,nz,radius});
                    }
                }
            }
        }
    }
    return result;
}

long double test_legendre(int degree,long double value) {
    if (degree==0) return 1.0L;
    if (degree==1) return value;
    long double previous=1.0L;
    long double current=value;
    for (int index=2;index<=degree;++index) {
        const long double next=
            ((2*index-1)*value*current
             -(index-1)*previous)/index;
        previous=current;
        current=next;
    }
    return current;
}

std::uint64_t direct_closing_pair_count(
    const DirectLatticeShell& first,
    const DirectLatticeShell& second) {
    std::uint64_t result=0;
    for (const auto& left:first.nonzero_modes) {
        for (const auto& right:second.nonzero_modes) {
            if (left.nx==-right.nx
                &&left.ny==-right.ny
                &&left.nz==-right.nz) {
                ++result;
            }
        }
    }
    return result;
}

long double direct_lattice_moment(
    const DirectLatticeShell& first,
    const DirectLatticeShell& second,
    int first_power,int second_power,int degree) {
    long double result=0.0L;
    for (const auto& left:first.nonzero_modes) {
        const long double left_integer_radius=std::sqrt(
            static_cast<long double>(
                left.nx*left.nx+left.ny*left.ny
                +left.nz*left.nz));
        for (const auto& right:second.nonzero_modes) {
            if (left.nx==-right.nx
                &&left.ny==-right.ny
                &&left.nz==-right.nz) {
                continue;
            }
            const long double right_integer_radius=std::sqrt(
                static_cast<long double>(
                    right.nx*right.nx+right.ny*right.ny
                    +right.nz*right.nz));
            const long double dot=
                static_cast<long double>(left.nx)*right.nx
                +static_cast<long double>(left.ny)*right.ny
                +static_cast<long double>(left.nz)*right.nz;
            const long double mu=std::clamp(
                dot/(left_integer_radius*right_integer_radius),
                -1.0L,1.0L);
            result+=
                std::pow(
                    static_cast<long double>(left.radius),
                    first_power)
                *std::pow(
                    static_cast<long double>(right.radius),
                    second_power)
                *test_legendre(degree,mu);
        }
    }
    return result/
        (static_cast<long double>(first.full_modes)
         *static_cast<long double>(second.full_modes));
}

long double projected_lattice_moment(
    const std::vector<shell::ShellNode>& nodes,
    int first_power,int second_power,int degree) {
    long double result=0.0L;
    for (const auto& node:nodes) {
        result+=
            static_cast<long double>(node.weight)
            *std::pow(
                static_cast<long double>(node.k1),
                first_power)
            *std::pow(
                static_cast<long double>(node.k2),
                second_power)
            *test_legendre(
                degree,
                static_cast<long double>(
                    node.internal_mu));
    }
    return result;
}

std::array<marisa_b_halo_v1::Vec3,3> direct_closed_vectors(
    const DirectLatticeMode& first,
    const DirectLatticeMode& second,
    double box_size) {
    constexpr double two_pi=
        6.283185307179586476925286766559005768;
    const double fundamental=two_pi/box_size;
    const marisa_b_halo_v1::Vec3 k1{
        fundamental*first.nx,
        fundamental*first.ny,
        fundamental*first.nz};
    const marisa_b_halo_v1::Vec3 k2{
        fundamental*second.nx,
        fundamental*second.ny,
        fundamental*second.nz};
    return {{
        k1,k2,
        marisa_b_halo_v1::negate(
            marisa_b_halo_v1::add(k1,k2))}};
}

double cubic_window_probe(
    const std::array<marisa_b_halo_v1::Vec3,3>& triangle) {
    marisa_b_halo_v1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    const auto W=[&](const marisa_b_halo_v1::Vec3& value) {
        return marisa_b_halo_v1::reconstruction_window(
            value,reconstruction);
    };
    /*
     * This contains one- and two-block Cartesian-window products of the same
     * type generated by reconstructed K2--K4 kernels.  It is inexpensive
     * enough to compare with a literal pair enumeration.
     */
    return W(triangle[0])+0.7*W(triangle[1])
           +0.3*W(triangle[2])
           +0.2*W(triangle[0])*W(triangle[1]);
}

double direct_cubic_lattice_probe(
    const DirectLatticeShell& first,
    const DirectLatticeShell& second,
    double box_size) {
    long double total=0.0L;
    for (const auto& left:first.nonzero_modes) {
        for (const auto& right:second.nonzero_modes) {
            if (left.nx==-right.nx
                &&left.ny==-right.ny
                &&left.nz==-right.nz) {
                continue;
            }
            total+=cubic_window_probe(
                direct_closed_vectors(
                    left,right,box_size));
        }
    }
    return static_cast<double>(
        total
        /(static_cast<long double>(first.full_modes)
          *static_cast<long double>(second.full_modes)));
}

double projected_cubic_probe(
    const std::vector<shell::ShellNode>& nodes) {
    long double total=0.0L;
    for (const auto& node:nodes) {
        total+=static_cast<long double>(node.weight)
               *cubic_window_probe(node.closed_vectors);
    }
    return static_cast<double>(total);
}

double loop_window_probe(
    const std::array<marisa_b_halo_v1::Vec3,3>& triangle) {
    marisa_b_halo_v1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    const auto W=[&](const marisa_b_halo_v1::Vec3& value) {
        return marisa_b_halo_v1::reconstruction_window(
            value,reconstruction);
    };
    const double inverse_sqrt2=1.0/std::sqrt(2.0);
    const double inverse_sqrt3=1.0/std::sqrt(3.0);
    const std::array<marisa_b_halo_v1::Vec3,7> q_nodes={{
        {0.08,0.0,0.0},
        {0.0,0.08,0.0},
        {0.0,0.0,0.08},
        {0.11*inverse_sqrt2,0.11*inverse_sqrt2,0.0},
        {0.11*inverse_sqrt2,0.0,0.11*inverse_sqrt2},
        {0.0,0.11*inverse_sqrt2,0.11*inverse_sqrt2},
        {0.14*inverse_sqrt3,0.14*inverse_sqrt3,
            0.14*inverse_sqrt3}}};
    double total=0.0;
    for (const auto& positive:q_nodes) {
        for (double sign:{-1.0,1.0}) {
            const marisa_b_halo_v1::Vec3 q{
                sign*positive.x,
                sign*positive.y,
                sign*positive.z};
            total+=
                W(q)
                *W(marisa_b_halo_v1::subtract(
                    triangle[0],q))
                +0.6
                *W(marisa_b_halo_v1::add(
                    triangle[1],q))
                *W(marisa_b_halo_v1::subtract(
                    triangle[2],q))
                +0.2*W(q)
                *W(marisa_b_halo_v1::add(
                    triangle[0],q))
                *W(marisa_b_halo_v1::subtract(
                    triangle[1],q));
        }
    }
    return total/(2.0*q_nodes.size());
}

double direct_loop_window_lattice_probe(
    const DirectLatticeShell& first,
    const DirectLatticeShell& second,
    double box_size) {
    long double total=0.0L;
    for (const auto& left:first.nonzero_modes) {
        for (const auto& right:second.nonzero_modes) {
            if (left.nx==-right.nx
                &&left.ny==-right.ny
                &&left.nz==-right.nz) {
                continue;
            }
            total+=loop_window_probe(
                direct_closed_vectors(
                    left,right,box_size));
        }
    }
    return static_cast<double>(
        total
        /(static_cast<long double>(first.full_modes)
          *static_cast<long double>(second.full_modes)));
}

double projected_loop_window_probe(
    const std::vector<shell::ShellNode>& nodes) {
    long double total=0.0L;
    for (const auto& node:nodes) {
        total+=static_cast<long double>(node.weight)
            *loop_window_probe(node.closed_vectors);
    }
    return static_cast<double>(total);
}

long double k3_moment(
    const std::vector<shell::ShellNode>& nodes,
    int degree) {
    long double result=0.0L;
    for (const auto& node:nodes) {
        result+=
            static_cast<long double>(node.weight)
            *std::pow(
                static_cast<long double>(
                    marisa_b_halo_v1::norm(
                        node.closed_vectors[2])),
                degree);
    }
    return result;
}

void test_exact_lattice_rules() {
    eft::ExactLatticeShellRuleConfig registered;
    registered.radial_order=4;
    registered.angular_order=16;
    registered.fft_box_size=1000.0;
    registered.fft_mesh_size=256;
    const shell::ShellBin zero_bin{
        0.0,0.02035714285714286,
        0.0,0.02035714285714286};
    const auto zero_rule=
        eft::make_exact_lattice_shell_rule(
            zero_bin,registered);
    require(
        zero_rule.nodes.size()==256,
        "registered exact-lattice node count");
    require(
        zero_rule.diagnostics.first_full_modes==147,
        "registered exact-lattice full shell-zero count");
    require(
        zero_rule.diagnostics.first_nonzero_modes==146,
        "registered exact-lattice nonzero shell-zero count");
    require(
        zero_rule.diagnostics.closing_zero_pairs==146,
        "registered exact-lattice closing-zero count");
    require_close(
        zero_rule.diagnostics.total_weight,
        0.9796843907631081,
        0.0,2.0e-15,
        "registered exact-lattice valid-pair weight");

    eft::ExactLatticeShellRuleConfig high;
    high.radial_order=2;
    high.angular_order=12;
    high.fft_box_size=100.0;
    high.fft_mesh_size=16;
    const shell::ShellBin overlap{
        0.0,0.15,0.07,0.19};
    const auto exact=
        eft::make_exact_lattice_shell_rule(overlap,high);
    const shell::ShellBin diagonal{
        overlap.k1_lower,overlap.k1_upper,
        overlap.k1_lower,overlap.k1_upper};
    const auto diagonal_exact=
        eft::make_exact_lattice_shell_rule(
            diagonal,high);
    eft::MultilevelLatticeShellRuleConfig low;
    low.exact=high;
    low.interpolation_order=6;
    const auto multilevel=
        eft::make_multilevel_lattice_shell_rule(
            overlap,low);
    require(exact.nodes.size()==48,
            "exact-lattice high-rule node count");
    require(multilevel.nodes.size()==24,
            "multilevel low-rule node count");
    require(
        multilevel.diagnostics.maximum_sampled_lebesgue<20.0,
        "multilevel sampled Lebesgue constant");
    const DirectLatticeShell first=
        enumerate_lattice_shell(
            overlap.k1_lower,overlap.k1_upper,
            high.fft_box_size,high.fft_mesh_size);
    const DirectLatticeShell second=
        enumerate_lattice_shell(
            overlap.k2_lower,overlap.k2_upper,
            high.fft_box_size,high.fft_mesh_size);
    require(
        diagonal_exact.diagnostics.first_full_modes
            ==first.full_modes,
        "direct diagonal first full-mode count");
    require(
        exact.diagnostics.second_full_modes
            ==second.full_modes,
        "direct overlap second full-mode count");
    require(
        diagonal_exact.diagnostics.closing_zero_pairs
            ==direct_closing_pair_count(first,first),
        "direct diagonal closing-pair count");
    require(
        exact.diagnostics.closing_zero_pairs
            ==direct_closing_pair_count(first,second),
        "direct overlap closing-pair count");
    for (const auto& comparison:
         std::vector<std::tuple<
             const DirectLatticeShell*,
             const DirectLatticeShell*,
             const std::vector<shell::ShellNode>*,
             std::string>>{
             {&first,&first,&diagonal_exact.nodes,
              "diagonal"},
             {&first,&second,&exact.nodes,
              "overlap"}}) {
        for (int degree=0;
             degree<high.angular_order;
             ++degree) {
            for (int first_power=0;
                 first_power<high.radial_order;
                 ++first_power) {
                for (int second_power=0;
                     second_power<high.radial_order;
                     ++second_power) {
                    require_close(
                        static_cast<double>(
                            projected_lattice_moment(
                                *std::get<2>(comparison),
                                first_power,second_power,
                                degree)),
                        static_cast<double>(
                            direct_lattice_moment(
                                *std::get<0>(comparison),
                                *std::get<1>(comparison),
                                first_power,second_power,
                                degree)),
                        0.0,2.0e-12,
                        "exact-lattice direct "
                        +std::get<3>(comparison)
                        +" moment l="
                        +std::to_string(degree));
                }
            }
        }
    }
    for (int degree=0;degree<6;++degree) {
        require_close(
            static_cast<double>(
                k3_moment(multilevel.nodes,degree)),
            static_cast<double>(
                k3_moment(exact.nodes,degree)),
            0.0,8.0e-13,
            "multilevel k3 polynomial identity");
    }

    eft::ExactLatticeShellRuleConfig cubic_exact=high;
    cubic_exact.radial_order=4;
    cubic_exact.angular_order=24;
    const auto cubic_invariant=
        eft::make_exact_lattice_shell_rule(
            overlap,cubic_exact);
    eft::HaarOrientedLatticeShellRuleConfig oriented_config;
    oriented_config.exact=cubic_exact;
    oriented_config.n_alpha=4;
    oriented_config.n_cos_beta=3;
    oriented_config.n_gamma=4;
    const auto oriented=
        eft::make_haar_oriented_lattice_shell_rule(
            overlap,oriented_config);
    require(
        oriented.diagnostics.invariant_nodes
            ==cubic_invariant.nodes.size(),
        "Haar-oriented invariant node count");
    require(
        oriented.diagnostics.orientation_nodes==48,
        "Haar-oriented SO3 node count");
    require(
        oriented.nodes.size()
            ==48*cubic_invariant.nodes.size(),
        "Haar-oriented expanded node count");
    require_close(
        oriented.diagnostics.total_weight,
        cubic_invariant.diagnostics.total_weight,
        0.0,2.0e-13,
        "Haar-oriented preserves finite-box denominator");
    for (int degree=0;degree<6;++degree) {
        require_close(
            static_cast<double>(
                k3_moment(oriented.nodes,degree)),
            static_cast<double>(
                k3_moment(cubic_invariant.nodes,degree)),
            0.0,2.0e-12,
            "Haar orientation preserves invariant k3 moment");
    }

    eft::HaarOrientedLatticeShellRuleConfig refined_config=
        oriented_config;
    refined_config.n_alpha=8;
    refined_config.n_cos_beta=6;
    refined_config.n_gamma=8;
    const auto refined=
        eft::make_haar_oriented_lattice_shell_rule(
            overlap,refined_config);
    const double direct_cubic=
        direct_cubic_lattice_probe(
            first,second,high.fft_box_size);
    const double coarse_cubic=
        projected_cubic_probe(oriented.nodes);
    const double refined_cubic=
        projected_cubic_probe(refined.nodes);
    require_close(
        coarse_cubic,refined_cubic,
        2.0e-4,2.0e-10,
        "Haar cubic-window orientation convergence");
    require_close(
        refined_cubic,direct_cubic,
        2.0e-5,2.0e-9,
        "Haar-oriented cubic-window direct lattice oracle");

    /*
     * A selected production bin additionally probes the Cartesian windows
     * inside loop-like W(q)W(k-q) products.  This is a literal cubic-lattice
     * outer-pair enumeration, not another use of the invariant shell rule.
     */
    const shell::ShellBin production_bin{
        0.0,0.02035714285714286,
        0.02035714285714286,0.04107142857142857};
    const DirectLatticeShell production_first=
        enumerate_lattice_shell(
            production_bin.k1_lower,
            production_bin.k1_upper,
            1000.0,256);
    const DirectLatticeShell production_second=
        enumerate_lattice_shell(
            production_bin.k2_lower,
            production_bin.k2_upper,
            1000.0,256);
    eft::HaarOrientedLatticeShellRuleConfig
        production_config;
    production_config.exact.radial_order=2;
    production_config.exact.angular_order=8;
    production_config.exact.fft_box_size=1000.0;
    production_config.exact.fft_mesh_size=256;
    production_config.n_alpha=8;
    production_config.n_cos_beta=2;
    production_config.n_gamma=8;
    const auto production_rule=
        eft::make_haar_oriented_lattice_shell_rule(
            production_bin,production_config);
    const double direct_loop_window=
        direct_loop_window_lattice_probe(
            production_first,production_second,
            1000.0);
    const double projected_loop_window=
        projected_loop_window_probe(
            production_rule.nodes);
    require_close(
        projected_loop_window,direct_loop_window,
        3.0e-4,2.0e-10,
        "production-bin Haar loop-window cubic-orbit oracle");
}

void test_multilevel_exact_shell_average() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    const shell::ShellBin bin{
        0.07,0.10,0.11,0.15};
    shell::ShellQuadratureConfig shell_config;
    shell_config.n_radial=1;
    shell_config.n_internal_mu=4;
    shell_config.average_grid_orientation=false;
    shell_config.radial_measure=
        shell::ShellRadialMeasure::FftLattice;
    shell_config.fft_box_size=100.0;
    shell_config.fft_mesh_size=16;

    eft::HybridShellIntegrationConfig hybrid;
    hybrid.convolution.qmin=1.0e-3;
    hybrid.convolution.qmax=0.50;
    hybrid.convolution.n_radial=2;
    hybrid.convolution.n_mu=4;
    hybrid.convolution.n_phi=4;
    hybrid.convolution.ir_safe=true;
    hybrid.convolution.uv_subtract=true;
    hybrid.factorized_p13.qmin=
        hybrid.convolution.qmin;
    hybrid.factorized_p13.qmax=
        hybrid.convolution.qmax;
    hybrid.factorized_p13.n_radial=2;
    hybrid.factorized_p13.n_mu=4;
    hybrid.factorized_p13.restore_p13_uv_tail=false;
    hybrid.analytic.regulator.coarse_epsilon=1.0e-8;
    hybrid.analytic.regulator.refinement_ratio=10.0;
    hybrid.analytic.uv_restoration.enabled=false;
    hybrid.exact_lattice_multilevel=true;
    hybrid.expensive_lattice_radial_order=1;
    hybrid.expensive_k3_interpolation_order=3;

    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-3;
    fftlog.kmax=2.0;
    fftlog.frequency_count=8;
    fftlog.reconstruction_grid_size=32;
    fftlog.bias_nu=-0.3;
    fftlog.reconstruct_interpolated_power=false;
    fftlog.endpoint_inclusive_sampling=true;
    const eft::FftlogDrOracle oracle(power,fftlog);

    eft::StochasticIntegrationConfig stochastic;
    stochastic.qmin=1.0e-3;
    stochastic.qmax=0.40;
    stochastic.n_radial=2;
    stochastic.n_mu=4;
    stochastic.restore_p13_uv_tail=false;
    const auto projected=
        eft::compute_eft_shell_templates_hybrid(
            power,power,bin,provider,oracle,hybrid,
            shell_config,stochastic,0.30);
    require(
        projected.exact_joint_lattice_measure,
        "multilevel exact-joint-lattice metadata");
    require(
        projected.multilevel_external_projection,
        "multilevel external-projection metadata");
    require(projected.cheap_shell_nodes==4,
            "multilevel cheap node count");
    require(projected.expensive_shell_nodes==3,
            "multilevel expensive node count");
    require(projected.shell_nodes==7,
            "multilevel total node count");
    require(projected.cached_p13_nodes>0,
            "multilevel cached P13 work");
    require(
        projected.total_loop_nodes>
            projected.cached_p13_nodes,
        "multilevel expensive loop work");
    require_finite_polynomial(
        projected.diagrams.tree,
        "multilevel tree");
    require_finite_polynomial(
        projected.diagrams.B222,
        "multilevel B222");
    require_finite_polynomial(
        projected.diagrams.B321I,
        "multilevel B321I");
    require_finite_polynomial(
        projected.diagrams.B321II,
        "multilevel B321II");
    require_finite_polynomial(
        projected.diagrams.B411,
        "multilevel B411");
    require_polynomial_close(
        projected.diagrams.B321II,
        projected.diagrams.B321II_bare
            -projected.diagrams.B321II_uv_subtraction
            +projected.diagrams.B321II_uv_restoration,
        2.0e-13,1.0e-8,
        "multilevel B321II recomposition");
    require_polynomial_close(
        projected.diagrams.B411,
        projected.diagrams.B411_bare
            +projected.diagrams.B411_uv_restoration,
        2.0e-13,1.0e-8,
        "multilevel B411 recomposition");
    require_polynomial_close(
        projected.diagrams.one_loop,
        projected.diagrams.B222
            +projected.diagrams.B321I
            +projected.diagrams.B321II
            +projected.diagrams.B411,
        2.0e-13,1.0e-8,
        "multilevel one-loop recomposition");
    require_polynomial_close(
        projected.diagrams.total,
        projected.diagrams.tree
            +projected.diagrams.one_loop,
        2.0e-13,1.0e-8,
        "multilevel total recomposition");
}

void test_reconstructed_shell_average() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider base;
    const shell::ShellBin bin{0.07,0.10,0.11,0.15};
    marisa_b_halo_v1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;

    eft::ReconstructedShellIntegrationConfig integration;
    integration.loop.qmin=1.0e-3;
    integration.loop.qmax=0.30;
    integration.loop.n_radial=1;
    integration.loop.n_mu=2;
    integration.loop.n_phi=2;
    integration.loop.ir_safe=false;
    integration.loop.uv_subtract=true;
    integration.loop.restore_uv_tail=false;
    integration.loop.exploit_phi_reflection=false;
    integration.shell.exact.radial_order=1;
    integration.shell.exact.angular_order=2;
    integration.shell.exact.fft_box_size=100.0;
    integration.shell.exact.fft_mesh_size=16;
    integration.shell.n_alpha=2;
    integration.shell.n_cos_beta=1;
    integration.shell.n_gamma=2;

    const auto projected=
        eft::compute_reconstructed_eft_shell_templates(
            power,power,bin,base,reconstruction,
            integration,{},0.30);
    const auto rule=
        eft::make_haar_oriented_lattice_shell_rule(
            bin,integration.shell);
    require(
        projected.shell_nodes==rule.nodes.size(),
        "post shell expanded node count");
    require(
        projected.invariant_shell_nodes
            ==rule.diagnostics.invariant_nodes,
        "post shell invariant node count");
    require(
        projected.orientation_nodes
            ==rule.diagnostics.orientation_nodes,
        "post shell orientation node count");
    require(
        !projected.exact_joint_lattice_measure,
        "post shell does not overclaim full exact lattice measure");
    require(
        projected.exact_k1_k2_mu_lattice_measure,
        "post shell exact k1-k2-mu lattice metadata");
    require(
        !projected.conditional_cubic_orientation_exact,
        "post shell registers Haar conditional orientation");
    require(
        projected.haar_oriented_cic_projection,
        "post shell Haar CIC metadata");
    require(
        projected.reconstructed_counterterms,
        "post shell reconstructed counterterm metadata");
    require(
        !projected.reconstructed_stochastic,
        "post shell refuses implicit pre stochastic basis");

    const eft::ReconstructedFieldKernelProvider provider(
        base,reconstruction);
    eft::SparsePolynomial tree;
    eft::SparsePolynomial loop;
    eft::CountertermTemplates counterterms;
    for (const auto& node:rule.nodes) {
        const auto diagrams=eft::evaluate_direct(
            power,node.closed_vectors,provider,
            integration.loop);
        const eft::DiagramAssembler assembler(
            power,node.closed_vectors,provider);
        tree+=node.weight*assembler.tree();
        loop+=node.weight*diagrams.one_loop;
        std::array<double,3> external_power={};
        for (int leg=0;leg<3;++leg) {
            external_power[leg]=power(
                marisa_b_halo_v1::norm(
                    node.closed_vectors[leg]));
        }
        const auto counter=
            eft::reconstructed_counterterm_bispectrum(
                base,reconstruction,node.closed_vectors,
                external_power,0.30);
        for (std::size_t direction=0;
             direction<eft::kCountertermCount;
             ++direction) {
            counterterms.BctrI[direction]+=
                node.weight*counter.BctrI[direction];
            counterterms.BctrII[direction]+=
                node.weight*counter.BctrII[direction];
            counterterms.total[direction]+=
                node.weight*counter.total[direction];
        }
    }
    require_polynomial_close(
        projected.diagrams.tree,tree,
        2.0e-13,1.0e-8,
        "post shell reconstructed tree");
    require_polynomial_close(
        projected.diagrams.one_loop,loop,
        2.0e-12,1.0e-7,
        "post shell reconstructed direct loop");
    for (std::size_t direction=0;
         direction<eft::kCountertermCount;
         ++direction) {
        require_polynomial_close(
            projected.counterterms.total[direction],
            counterterms.total[direction],
            2.0e-13,1.0e-8,
            "post shell reconstructed counterterm");
    }

    bool rejected_disabled=false;
    try {
        auto disabled=reconstruction;
        disabled.enabled=false;
        (void)eft::compute_reconstructed_eft_shell_templates(
            power,power,bin,base,disabled,
            integration,{},0.30);
    } catch (const std::invalid_argument&) {
        rejected_disabled=true;
    }
    require(
        rejected_disabled,
        "post shell rejects disabled reconstruction");
    auto stochastic_integration=integration;
    stochastic_integration
        .include_reconstructed_stochastic=true;
    eft::StochasticIntegrationConfig stochastic;
    stochastic.qmin=1.0e-3;
    stochastic.qmax=0.30;
    stochastic.n_radial=1;
    stochastic.n_mu=2;
    stochastic.n_phi=2;
    stochastic.restore_p13_uv_tail=false;
    const auto with_stochastic=
        eft::compute_reconstructed_eft_shell_templates(
            power,power,bin,base,reconstruction,
            stochastic_integration,stochastic,0.30);
    require(
        with_stochastic.reconstructed_stochastic,
        "post shell registers reconstructed stochastic map");
    require(
        with_stochastic.reconstructed_stochastic_decomposed,
        "post shell registers density/noisy-shift decomposition");
    require_finite_polynomial(
        with_stochastic.stochastic.raw[4],
        "post shell reconstructed mixed stochastic");
    for (std::size_t index=0;
         index<eft::kStochasticCount;
         ++index) {
        require_polynomial_close(
            with_stochastic.stochastic.raw[index],
            with_stochastic.stochastic_density_only.raw[index]
                +with_stochastic.stochastic_noisy_shift.raw[index],
            3.0e-12,2.0e-7,
            "post shell stochastic tied recomposition");
    }
    require_polynomial_close(
        with_stochastic.stochastic.bshot_bnabla2_cross,
        with_stochastic.stochastic_density_only
                .bshot_bnabla2_cross
            +with_stochastic.stochastic_noisy_shift
                .bshot_bnabla2_cross,
        3.0e-12,2.0e-7,
        "post shell stochastic cross tied recomposition");
}

void test_b411_near_equal_fft_shell_legs() {
    const GaussianPower power;
    const eft::EftBiasKernelProvider provider;
    shell::ShellQuadratureConfig shell_config;
    shell_config.n_radial=3;
    shell_config.n_internal_mu=12;
    shell_config.average_grid_orientation=false;
    shell_config.radial_measure=shell::ShellRadialMeasure::FftLattice;
    shell_config.fft_box_size=1000.0;
    shell_config.fft_mesh_size=256;
    const shell::ShellBin bin{
        0.061785714285714277,0.08249999999999999,
        0.061785714285714277,0.08249999999999999};
    const auto nodes=shell::make_shell_nodes(bin,shell_config);

    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-3;
    fftlog.kmax=2.0;
    fftlog.frequency_count=8;
    fftlog.reconstruction_grid_size=32;
    fftlog.bias_nu=-0.3;
    fftlog.reconstruct_interpolated_power=false;
    fftlog.endpoint_inclusive_sampling=true;
    const eft::FftlogDrOracle oracle(power,fftlog);

    bool exercised_near_equal_inversion=false;
    for (const auto& node:nodes) {
        const auto canonical=
            marisa_b_halo_v1::canonicalize_closed_vectors(
                node.closed_vectors);
        const std::array<double,3> lengths={{
            marisa_b_halo_v1::norm(canonical.k1),
            marisa_b_halo_v1::norm(canonical.k2),
            marisa_b_halo_v1::norm(canonical.k3)}};
        if (lengths[1]>lengths[2]
            &&lengths[1]-lengths[2]<1.0e-12) {
            const auto value=oracle.evaluate_b411_analytic(
                node.closed_vectors,provider,{}, {},
                1.0e-12,1.0e-8);
            require_finite_polynomial(
                value.value,
                "B411 accepts canonical-tolerance equal-leg inversion");
            exercised_near_equal_inversion=true;
            break;
        }
    }
    require(
        exercised_near_equal_inversion,
        "production FFT shell contains near-equal canonical leg inversion");
}

}  // namespace

int main() {
    try {
        test_shell_average();
        test_hybrid_shell_average();
        test_exact_lattice_rules();
        test_multilevel_exact_shell_average();
        test_reconstructed_shell_average();
        test_b411_near_equal_fft_shell_legs();
        std::cout<<"EFT-v2 shell-projector checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 shell-projector test failed: "<<error.what()<<"\n";
        return 1;
    }
}
