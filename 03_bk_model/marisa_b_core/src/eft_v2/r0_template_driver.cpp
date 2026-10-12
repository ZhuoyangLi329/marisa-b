#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "fftlog_dr_oracle.h"
#include "ir_resummation.h"
#include "shell_projector.h"

namespace eft=marisa_b_eft_v2;
namespace shell=marisa_b_shell_v1;

namespace {

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0,p=0.0;
            if (row>>k>>p) {
                if (!(k>0.0 && p>0.0)) throw std::runtime_error("power table is not positive");
                log_k_.push_back(std::log(k));
                log_p_.push_back(std::log(p));
            }
        }
        if (log_k_.size()<2) throw std::runtime_error("power table is too short");
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) right=log_k_.size()-1;
        else if (x>log_k_.front()) {
            right=std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin();
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(log_p_[left]+fraction*(log_p_[right]-log_p_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

int parse_int(const char* text,const std::string& name,int minimum=1) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size() || value<minimum) {
        throw std::invalid_argument("invalid integer for "+name);
    }
    return value;
}

double parse_double(const char* text,const std::string& name) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid number for "+name);
    }
    return value;
}

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open radial edge file: "+path);
    std::vector<double> edges;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line);
        double value=0.0;
        if (!(row>>value)) throw std::runtime_error("invalid radial edge row");
        edges.push_back(value);
    }
    if (edges.size()!=16) throw std::runtime_error("R0 radial edge file must contain 16 values");
    for (std::size_t index=1;index<edges.size();++index) {
        if (!(edges[index]>edges[index-1])) {
            throw std::runtime_error("R0 radial edges must be strictly increasing");
        }
    }
    return edges;
}

void write_json_string(const std::string& value) {
    std::cout<<'"';
    for (unsigned char character:value) {
        switch (character) {
            case '"': std::cout<<"\\\""; break;
            case '\\': std::cout<<"\\\\"; break;
            case '\n': std::cout<<"\\n"; break;
            case '\r': std::cout<<"\\r"; break;
            case '\t': std::cout<<"\\t"; break;
            default:
                if (character<0x20) {
                    const auto flags=std::cout.flags();
                    const char fill=std::cout.fill();
                    std::cout<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')
                             <<static_cast<int>(character);
                    std::cout.flags(flags); std::cout.fill(fill);
                } else std::cout<<character;
        }
    }
    std::cout<<'"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout<<'{';
    bool first=true;
    for (const auto& term:polynomial.terms()) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(term.first.canonical_string());
        std::cout<<':'<<term.second;
    }
    std::cout<<'}';
}

void write_named_polynomial(const char* name,const eft::SparsePolynomial& value,bool& first) {
    if (!first) std::cout<<',';
    first=false;
    write_json_string(name);
    std::cout<<':';
    write_polynomial(value);
}

void write_diagram_templates(const eft::DiagramTemplates& value) {
    std::cout<<'{';
    bool first=true;
    write_named_polynomial("tree",value.tree,first);
    write_named_polynomial("B222",value.B222,first);
    write_named_polynomial("B321I",value.B321I,first);
    write_named_polynomial("B321II",value.B321II,first);
    write_named_polynomial("B411",value.B411,first);
    write_named_polynomial("B321II_bare",value.B321II_bare,first);
    write_named_polynomial("B411_bare",value.B411_bare,first);
    write_named_polynomial(
        "B321II_uv_subtraction",value.B321II_uv_subtraction,first);
    write_named_polynomial(
        "B411_uv_subtraction",value.B411_uv_subtraction,first);
    write_named_polynomial(
        "uv_subtraction_total",value.uv_subtraction_total,first);
    write_named_polynomial(
        "B321II_uv_restoration",value.B321II_uv_restoration,first);
    write_named_polynomial(
        "B411_uv_restoration",value.B411_uv_restoration,first);
    write_named_polynomial(
        "uv_restoration_total",value.uv_restoration_total,first);
    write_named_polynomial("one_loop",value.one_loop,first);
    write_named_polynomial("total",value.total,first);
    std::cout<<'}';
}

void write_regulator_diagnostic(
    const eft::FftlogRegulatorDiagnostic& value) {
    std::cout<<std::setprecision(17)
             <<"{\"coarse_epsilon\":"<<value.coarse_epsilon
             <<",\"fine_epsilon\":"<<value.fine_epsilon
             <<",\"refinement_ratio\":"<<value.refinement_ratio
             <<",\"maximum_absolute_coarse_to_fine\":"
             <<value.maximum_absolute_coarse_to_fine
             <<",\"maximum_relative_coarse_to_fine\":"
             <<value.maximum_relative_coarse_to_fine
             <<",\"maximum_absolute_extrapolation_correction\":"
             <<value.maximum_absolute_extrapolation_correction
             <<",\"maximum_relative_extrapolation_correction\":"
             <<value.maximum_relative_extrapolation_correction<<'}';
}

void add_factorized_p13_uv_tail_to_direct(
    eft::DiagramTemplates& direct,
    const PowerSpectrum& power,
    const std::array<eft::Vec3,3>& triangle,
    const eft::FieldKernelProvider& provider,
    const eft::TracerPowerIntegrationConfig& integration) {
    if (!direct.B321II_uv_restoration.empty()) {
        throw std::logic_error(
            "selected direct result already contains a B321II "
            "UV-tail restoration");
    }
    const eft::FactorizedB321IITemplates factorized=
        eft::factorized_b321ii_templates(
            power,triangle,provider,integration);
    direct.B321II_uv_restoration=
        factorized.uv_tail_restoration;
    direct.uv_restoration_total=
        direct.B321II_uv_restoration
        +direct.B411_uv_restoration;
    direct.B321II+=direct.B321II_uv_restoration;
    direct.one_loop=
        direct.B222+direct.B321I
        +direct.B321II+direct.B411;
    direct.total=direct.tree+direct.one_loop;
    direct.integration_nodes+=factorized.integration_nodes;
}

void run_selected_dr(
    int argc,char** argv,bool include_analytic,bool include_direct,
    bool direct_ir_safe,bool b222_only,bool b321i_only,
    bool b321i_compare,bool b411_only) {
    if (argc!=22) {
        throw std::invalid_argument(
            "usage: eft_v2_r0_template_driver "
            "[--selected-dr|--selected-analytic|"
            "--selected-b222-analytic|"
            "--selected-b321i-analytic|--selected-b321i-compare|"
            "--selected-b411-analytic|"
            "--selected-direct|"
            "--selected-direct-unmapped] "
            "POWER_TABLE "
            "K1 K2 K3 FFT_KMIN FFT_KMAX FFT_FREQUENCIES FFT_RECON_GRID "
            "FFT_BIAS_OR_AUTO MODE_TOL COARSE_EPSILON REFINEMENT_RATIO "
            "DIRECT_QMIN DIRECT_QMAX DIRECT_NRAD DIRECT_NMU DIRECT_NPHI "
            "MU_REN ASYMPTOTIC_FACTOR LAURENT_VALIDATION_SAMPLES");
    }
    const std::string power_path=argv[2];
    const marisa_b_halo_v1::Triangle sides{
        parse_double(argv[3],"k1"),
        parse_double(argv[4],"k2"),
        parse_double(argv[5],"k3")};
    const marisa_b_halo_v1::CanonicalTriangle canonical=
        marisa_b_halo_v1::canonicalize_triangle(sides);
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    eft::FftlogConfig fftlog;
    fftlog.kmin=parse_double(argv[6],"FFTLog kmin");
    fftlog.kmax=parse_double(argv[7],"FFTLog kmax");
    fftlog.frequency_count=parse_int(argv[8],"FFTLog frequency count",8);
    fftlog.reconstruction_grid_size=
        parse_int(argv[9],"FFTLog reconstruction grid",8);
    fftlog.reconstruct_interpolated_power=include_direct;
    const std::string bias_text=argv[10];
    if (bias_text!="auto") {
        fftlog.bias_nu=parse_double(argv[10],"FFTLog bias");
    }
    fftlog.endpoint_inclusive_sampling=
        include_analytic && !include_direct && bias_text!="auto";
    eft::FftlogAnalyticOneLoopConfig analytic_config;
    analytic_config.mode_coefficient_relative_tolerance=
        parse_double(argv[11],"FFTLog mode tolerance");
    analytic_config.regulator.coarse_epsilon=
        parse_double(argv[12],"coarse epsilon");
    analytic_config.regulator.refinement_ratio=
        parse_double(argv[13],"regulator refinement ratio");
    const int validation_samples=
        parse_int(argv[21],"Laurent validation samples",16);
    analytic_config.B222_laurent.validation_samples=validation_samples;
    analytic_config.B321I_laurent.validation_samples=validation_samples;
    analytic_config.B321II_laurent.validation_samples=validation_samples;
    analytic_config.contour_sectors.enabled=
        include_analytic
        && !b222_only && !b321i_only && !b411_only;

    eft::DirectIntegrationConfig direct_config;
    direct_config.qmin=parse_double(argv[14],"direct qmin");
    direct_config.qmax=parse_double(argv[15],"direct qmax");
    direct_config.n_radial=parse_int(argv[16],"direct radial order");
    direct_config.n_mu=parse_int(argv[17],"direct mu order");
    direct_config.n_phi=parse_int(argv[18],"direct phi order");
    direct_config.ir_safe=direct_ir_safe;
    direct_config.uv_subtract=true;
    direct_config.exploit_phi_reflection=true;
    direct_config.uv.mu_ren=parse_double(argv[19],"renormalization scale");
    direct_config.uv.asymptotic_factor=
        parse_double(argv[20],"UV asymptotic factor");
    analytic_config.B321II_factorized_p13.qmin=direct_config.qmin;
    analytic_config.B321II_factorized_p13.qmax=direct_config.qmax;
    analytic_config.B321II_factorized_p13.n_radial=
        std::max(32,2*direct_config.n_radial);
    analytic_config.B321II_factorized_p13.n_mu=
        std::max(64,2*direct_config.n_mu);
    analytic_config.B321II_factorized_p13.uv=direct_config.uv;

    TabulatedPower linear(power_path);
    const eft::EftBiasKernelProvider provider;
    const eft::FftlogDrOracle oracle(linear,fftlog);
    if (b222_only) {
        const auto value=oracle.evaluate_b222_public_table_analytic(
            triangle,provider,analytic_config.master,
            analytic_config.mode_coefficient_relative_tolerance);
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"selected_b222_fftlog_dr\","
                 <<"\"schema\":\"marisa-b-eft-v2-selected-b222-dr-v1\","
                 <<"\"triangle\":["<<canonical.sides.k1<<','
                 <<canonical.sides.k2<<','<<canonical.sides.k3<<"],"
                 <<"\"power\":"<<oracle.power().metadata_json()<<','
                 <<"\"value\":";
        write_polynomial(value.value);
        std::cout<<",\"master_integral_evaluations\":"
                 <<value.master_integral_evaluations
                 <<",\"maximum_imaginary_to_real\":"
                 <<value.maximum_imaginary_to_real<<"}\n";
        return;
    }
    if (b321i_only) {
        const auto coarse=oracle.evaluate_b321i_public_table_analytic(
            triangle,provider,analytic_config.master,
            analytic_config.mode_coefficient_relative_tolerance,
            analytic_config.regulator.coarse_epsilon);
        const double fine_epsilon=
            analytic_config.regulator.coarse_epsilon
            /analytic_config.regulator.refinement_ratio;
        const auto fine=oracle.evaluate_b321i_public_table_analytic(
            triangle,provider,analytic_config.master,
            analytic_config.mode_coefficient_relative_tolerance,
            fine_epsilon);
        const double ratio=analytic_config.regulator.refinement_ratio;
        const eft::SparsePolynomial extrapolated=
            (ratio*fine.value-coarse.value)*(1.0/(ratio-1.0));
        std::optional<eft::FftlogAnalyticDiagramResult> generic_coarse;
        std::optional<eft::FftlogAnalyticDiagramResult> generic_fine;
        std::optional<eft::SparsePolynomial> generic_extrapolated;
        double generic_maximum_validation_relative=0.0;
        if (b321i_compare) {
            generic_coarse.emplace(oracle.evaluate_b321i_analytic(
                triangle,provider,analytic_config.B321I_laurent,
                analytic_config.master,
                analytic_config.mode_coefficient_relative_tolerance,
                analytic_config.regulator.coarse_epsilon));
            generic_fine.emplace(oracle.evaluate_b321i_analytic(
                triangle,provider,analytic_config.B321I_laurent,
                analytic_config.master,
                analytic_config.mode_coefficient_relative_tolerance,
                fine_epsilon));
            generic_extrapolated.emplace(
                (ratio*generic_fine->value-generic_coarse->value)
                *(1.0/(ratio-1.0)));
            for (const auto& reduction:generic_coarse->route_reductions) {
                generic_maximum_validation_relative=std::max(
                    generic_maximum_validation_relative,
                    reduction.maximum_validation_relative_error);
            }
            for (const auto& reduction:generic_fine->route_reductions) {
                generic_maximum_validation_relative=std::max(
                    generic_maximum_validation_relative,
                    reduction.maximum_validation_relative_error);
            }
        }
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"selected_b321i_fftlog_dr\","
                 <<"\"schema\":\"marisa-b-eft-v2-selected-b321i-dr-v2\","
                 <<"\"triangle\":["<<canonical.sides.k1<<','
                 <<canonical.sides.k2<<','<<canonical.sides.k3<<"],"
                 <<"\"power\":"<<oracle.power().metadata_json()<<','
                 <<"\"coarse_epsilon\":"
                 <<analytic_config.regulator.coarse_epsilon<<','
                 <<"\"fine_epsilon\":"<<fine_epsilon<<','
                 <<"\"coarse\":";
        write_polynomial(coarse.value);
        std::cout<<",\"fine\":";
        write_polynomial(fine.value);
        std::cout<<",\"extrapolated\":";
        write_polynomial(extrapolated);
        if (generic_coarse) {
            std::cout<<",\"generic_laurent\":{\"coarse\":";
            write_polynomial(generic_coarse->value);
            std::cout<<",\"fine\":";
            write_polynomial(generic_fine->value);
            std::cout<<",\"extrapolated\":";
            write_polynomial(*generic_extrapolated);
            std::cout<<",\"maximum_validation_relative\":"
                     <<generic_maximum_validation_relative
                     <<",\"master_integral_evaluations\":"
                     <<generic_coarse->master_integral_evaluations
                        +generic_fine->master_integral_evaluations
                     <<'}';
        }
        std::cout<<",\"master_integral_evaluations\":"
                 <<coarse.master_integral_evaluations
                    +fine.master_integral_evaluations
                 <<",\"maximum_imaginary_to_real\":"
                 <<std::max(coarse.maximum_imaginary_to_real,
                             fine.maximum_imaginary_to_real)
                 <<"}\n";
        return;
    }
    if (b411_only) {
        const auto coarse=oracle.evaluate_b411_analytic(
            triangle,provider,{},analytic_config.master,
            analytic_config.mode_coefficient_relative_tolerance,
            analytic_config.regulator.coarse_epsilon);
        const double fine_epsilon=
            analytic_config.regulator.coarse_epsilon
            /analytic_config.regulator.refinement_ratio;
        const auto fine=oracle.evaluate_b411_analytic(
            triangle,provider,{},analytic_config.master,
            analytic_config.mode_coefficient_relative_tolerance,
            fine_epsilon);
        const double ratio=analytic_config.regulator.refinement_ratio;
        const eft::SparsePolynomial extrapolated=
            (ratio*fine.value-coarse.value)*(1.0/(ratio-1.0));
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"selected_b411_fftlog_dr\","
                 <<"\"schema\":\"marisa-b-eft-v2-selected-b411-dr-v1\","
                 <<"\"triangle\":["<<canonical.sides.k1<<','
                 <<canonical.sides.k2<<','<<canonical.sides.k3<<"],"
                 <<"\"power\":"<<oracle.power().metadata_json()<<','
                 <<"\"coarse_epsilon\":"
                 <<analytic_config.regulator.coarse_epsilon<<','
                 <<"\"fine_epsilon\":"<<fine_epsilon<<','
                 <<"\"coarse\":";
        write_polynomial(coarse.value);
        std::cout<<",\"fine\":";
        write_polynomial(fine.value);
        std::cout<<",\"extrapolated\":";
        write_polynomial(extrapolated);
        std::cout<<",\"master_integral_evaluations\":"
                 <<coarse.master_integral_evaluations
                    +fine.master_integral_evaluations
                 <<",\"maximum_imaginary_to_real\":"
                 <<std::max(coarse.maximum_imaginary_to_real,
                             fine.maximum_imaginary_to_real)
                 <<"}\n";
        return;
    }
    std::optional<eft::DiagramTemplates> direct;
    std::optional<eft::DiagramTemplates> direct_fftlog_power;
    if (include_direct) {
        direct.emplace(eft::evaluate_direct(
            linear,triangle,provider,direct_config));
        add_factorized_p13_uv_tail_to_direct(
            *direct,linear,triangle,provider,
            analytic_config.B321II_factorized_p13);
        direct_fftlog_power.emplace(eft::evaluate_direct(
            oracle.power(),triangle,provider,direct_config));
    }
    std::optional<eft::FftlogAnalyticOneLoopResult> analytic;
    if (include_analytic) {
        analytic.emplace(
            oracle.evaluate_analytic(triangle,provider,analytic_config));
    }

    std::cout<<std::setprecision(17)
             <<"{\"record\":\"selected_fftlog_dr\","
             <<"\"schema\":\"marisa-b-eft-v2-selected-dr-v2\","
             <<"\"triangle\":["<<canonical.sides.k1<<','
             <<canonical.sides.k2<<','<<canonical.sides.k3<<']';
    if (direct) {
        std::cout<<",\"direct\":{\"qmin\":"<<direct_config.qmin
                 <<",\"qmax\":"<<direct_config.qmax
                 <<",\"ir_safe\":"
                 <<(direct_config.ir_safe?"true":"false")
                 <<",\"quadrature\":["<<direct_config.n_radial<<','
                 <<direct_config.n_mu<<','<<direct_config.n_phi<<"],"
                 <<"\"mu_ren\":"<<direct_config.uv.mu_ren
                 <<",\"asymptotic_factor\":"
                 <<direct_config.uv.asymptotic_factor
                 <<",\"p13_uv_tail_restoration\":{\"enabled\":"
                 <<(analytic_config.B321II_factorized_p13
                            .restore_p13_uv_tail
                        ?"true":"false")
                 <<",\"sampled_kmax\":"
                 <<analytic_config.B321II_factorized_p13
                        .uv_tail_kmax
                 <<",\"quadrature_order\":"
                 <<analytic_config.B321II_factorized_p13
                        .uv_tail_quadrature_order
                 <<'}'
                 <<",\"integration_nodes\":"<<direct->integration_nodes
                 <<",\"templates\":";
        write_diagram_templates(*direct);
        std::cout<<"},\"direct_fftlog_power\":{\"integration_nodes\":"
                 <<direct_fftlog_power->integration_nodes
                 <<",\"templates\":";
        write_diagram_templates(*direct_fftlog_power);
        std::cout<<'}';
    }
    if (analytic) {
        std::cout<<",\"analytic\":{\"metadata\":"
                 <<oracle.analytic_metadata_json(analytic_config)
                 <<",\"master_integral_evaluations\":"
                 <<analytic->master_integral_evaluations
                 <<",\"factorized_p13_integration_nodes\":"
                 <<analytic->B321II_factorized.integration_nodes
                 <<",\"master_integral_evaluations_by_topology\":{"
                 <<"\"B222\":"
                 <<analytic->B222.master_integral_evaluations
                 <<",\"B321I\":"
                 <<analytic->B321I.extrapolated.master_integral_evaluations
                 <<",\"B321II\":"
                 <<analytic->B321II.extrapolated.master_integral_evaluations
                 <<",\"B411\":"
                 <<analytic->B411.extrapolated.master_integral_evaluations
                 <<"},\"fftlog_modes_used\":{\"B222\":"
                 <<analytic->B222.fftlog_modes_used
                 <<",\"B321I\":"
                 <<analytic->B321I.extrapolated.fftlog_modes_used
                 <<",\"B321II\":"
                 <<analytic->B321II.extrapolated.fftlog_modes_used
                 <<",\"B411\":"
                 <<analytic->B411.extrapolated.fftlog_modes_used
                 <<"},\"uv_restoration\":{"
                 <<"\"normalized_linear_power_moment\":"
                 <<analytic->linear_power_uv_moment
                 <<",\"B321II_subleading_coefficient\":";
        write_polynomial(
            analytic->uv_subleading_coefficients.B321II);
        std::cout<<",\"B411_subleading_coefficient\":";
        write_polynomial(
            analytic->uv_subleading_coefficients.B411);
        std::cout<<"},\"regulator\":{\"B321I\":";
        write_regulator_diagnostic(analytic->B321I.regulator);
        std::cout<<",\"B321II\":";
        write_regulator_diagnostic(analytic->B321II.regulator);
        std::cout<<",\"B411\":";
        write_regulator_diagnostic(analytic->B411.regulator);
        std::cout<<"},\"B321II_flattened_master_diagnostic\":";
        write_polynomial(analytic->B321II.extrapolated.value);
        std::cout<<",\"templates\":";
        write_diagram_templates(analytic->templates);
        std::cout<<'}';
    }
    std::cout<<"}\n";
}

void write_bin(int index,const eft::EftShellTemplates& value) {
    std::cout<<std::setprecision(17)<<'{'
             <<"\"record\":\"bin\",\"index\":"<<index<<",\"edges\":["
             <<value.bin.k1_lower<<','<<value.bin.k1_upper<<','
             <<value.bin.k2_lower<<','<<value.bin.k2_upper<<"],"
             <<"\"shell_nodes\":"<<value.shell_nodes<<','
             <<"\"total_loop_nodes\":"<<value.total_loop_nodes<<','
             <<"\"cached_p13_nodes\":"<<value.cached_p13_nodes<<','
             <<"\"cheap_shell_nodes\":"<<value.cheap_shell_nodes<<','
             <<"\"expensive_shell_nodes\":"
             <<value.expensive_shell_nodes<<','
             <<"\"hybrid_factorized_analytic_tadpoles\":"
             <<(value.hybrid_factorized_analytic_tadpoles
                    ?"true":"false")<<','
             <<"\"fftlog_analytic_convolutions\":"
             <<(value.fftlog_analytic_convolutions
                    ?"true":"false")<<','
             <<"\"exact_joint_lattice_measure\":"
             <<(value.exact_joint_lattice_measure
                    ?"true":"false")<<','
             <<"\"multilevel_external_projection\":"
             <<(value.multilevel_external_projection
                    ?"true":"false")<<','
             <<"\"exact_lattice_radial_order\":"
             <<value.exact_lattice_radial_order<<','
             <<"\"exact_lattice_angular_order\":"
             <<value.exact_lattice_angular_order<<','
             <<"\"expensive_lattice_radial_order\":"
             <<value.expensive_lattice_radial_order<<','
             <<"\"expensive_k3_interpolation_order\":"
             <<value.expensive_k3_interpolation_order<<','
             <<"\"zero_external_leg_pairs\":"
             <<value.zero_external_leg_pairs<<','
             <<"\"closing_zero_pairs\":"
             <<value.closing_zero_pairs<<','
             <<"\"exact_lattice_valid_pair_fraction\":"
             <<value.exact_lattice_valid_pair_fraction<<','
             <<"\"exact_lattice_total_variation\":"
             <<value.exact_lattice_total_variation<<','
             <<"\"expensive_total_variation\":"
             <<value.expensive_total_variation<<','
             <<"\"expensive_maximum_sampled_lebesgue\":"
             <<value.expensive_maximum_sampled_lebesgue<<','
             <<"\"diagrams\":{";
    bool first=true;
    write_named_polynomial("tree",value.diagrams.tree,first);
    write_named_polynomial("B222",value.diagrams.B222,first);
    write_named_polynomial("B321I",value.diagrams.B321I,first);
    write_named_polynomial("B321II",value.diagrams.B321II,first);
    write_named_polynomial("B411",value.diagrams.B411,first);
    write_named_polynomial("B321II_bare",value.diagrams.B321II_bare,first);
    write_named_polynomial("B411_bare",value.diagrams.B411_bare,first);
    write_named_polynomial("B321II_uv_subtraction",value.diagrams.B321II_uv_subtraction,first);
    write_named_polynomial("B411_uv_subtraction",value.diagrams.B411_uv_subtraction,first);
    write_named_polynomial("B321II_uv_restoration",value.diagrams.B321II_uv_restoration,first);
    write_named_polynomial("B411_uv_restoration",value.diagrams.B411_uv_restoration,first);
    std::cout<<"},\"counterterms\":{";
    first=true;
    for (std::size_t parameter=0;parameter<eft::kCountertermCount;++parameter) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(std::string(eft::parameter_info(
            eft::counterterm_parameter_ids()[parameter]).name));
        std::cout<<':';
        write_polynomial(value.counterterms.total[parameter]);
    }
    std::cout<<"},\"stochastic\":{";
    first=true;
    for (std::size_t parameter=0;parameter<eft::kStochasticCount;++parameter) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(std::string(eft::parameter_info(
            eft::stochastic_parameter_ids()[parameter]).name));
        std::cout<<':';
        write_polynomial(value.stochastic.raw[parameter]);
    }
    std::cout<<"},\"bshot_bnabla2_cross\":";
    write_polynomial(value.stochastic.bshot_bnabla2_cross);
    std::cout<<"}\n";
    // Each bin is an independently valid checkpoint record.  Flush it so a
    // long partial-vector run remains observable and recoverable if the job is
    // preempted after completing one or more bins.
    std::cout.flush();
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc>1 && (std::string(argv[1])=="--selected-dr"
                      || std::string(argv[1])=="--selected-analytic"
                      || std::string(argv[1])
                            =="--selected-b222-analytic"
                      || std::string(argv[1])
                            =="--selected-b321i-analytic"
                      || std::string(argv[1])
                            =="--selected-b321i-compare"
                      || std::string(argv[1])
                            =="--selected-b411-analytic"
                      || std::string(argv[1])=="--selected-direct"
                      || std::string(argv[1])
                            =="--selected-direct-unmapped")) {
            const std::string selected_mode=argv[1];
            run_selected_dr(
                argc,argv,selected_mode=="--selected-dr"
                    || selected_mode=="--selected-analytic"
                    || selected_mode=="--selected-b222-analytic"
                    || selected_mode=="--selected-b321i-analytic"
                    || selected_mode=="--selected-b321i-compare"
                    || selected_mode=="--selected-b411-analytic",
                selected_mode!="--selected-analytic"
                    && selected_mode!="--selected-b222-analytic"
                    && selected_mode!="--selected-b321i-analytic"
                    && selected_mode!="--selected-b321i-compare"
                    && selected_mode!="--selected-b411-analytic",
                selected_mode!="--selected-direct-unmapped",
                selected_mode=="--selected-b222-analytic",
                selected_mode=="--selected-b321i-analytic"
                    || selected_mode=="--selected-b321i-compare",
                selected_mode=="--selected-b321i-compare",
                selected_mode=="--selected-b411-analytic");
            return 0;
        }
        const bool exact_lattice_hybrid_mode=
            argc==34
            &&std::string(argv[22])
                =="hybrid-exact-lattice";
        const bool analytic_hybrid_mode=
            argc==32
            &&std::string(argv[22])=="analytic-hybrid";
        const bool hybrid_mode=
            exact_lattice_hybrid_mode
            ||(argc==32
               &&(std::string(argv[22])=="hybrid"
                  ||analytic_hybrid_mode));
        if (argc!=22 && argc!=23 && !hybrid_mode) {
            std::cerr<<"usage: eft_v2_r0_template_driver POWER_TABLE EDGE_FILE IR0_OR_1 "
                     <<"SHELL_NRAD SHELL_NMU LOOP_NRAD LOOP_NMU LOOP_NPHI QMIN QMAX "
                     <<"STOCH_NRAD STOCH_NMU STOCH_QMAX MU_REN LAMBDA_IR R_S "
                     <<"SIGMA_POWER_0_P11_1_PNW SPLIT_0_EH_RATIO_1_LOGK3 "
                     <<"RADIAL_INDEX_STOP START_BIN STOP_BIN "
                     <<"[tensor-gauss|fibonacci]\n"
                     <<"   or append: [hybrid|analytic-hybrid] "
                     <<"[tensor-gauss|fibonacci] "
                     <<"FFT_KMIN FFT_KMAX FFT_N FFT_BIAS "
                     <<"P13_NRAD P13_NMU COARSE_EPSILON "
                     <<"REFINEMENT_RATIO\n"
                     <<"   or append: hybrid-exact-lattice "
                     <<"[tensor-gauss|fibonacci] "
                     <<"FFT_KMIN FFT_KMAX FFT_N FFT_BIAS "
                     <<"P13_NRAD P13_NMU COARSE_EPSILON "
                     <<"REFINEMENT_RATIO EXPENSIVE_NRAD "
                     <<"EXPENSIVE_K3_ORDER\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string edge_path=argv[2];
        const bool use_ir=parse_int(argv[3],"IR flag",0)!=0;
        shell::ShellQuadratureConfig shell_config;
        shell_config.n_radial=parse_int(argv[4],"shell radial order");
        shell_config.n_internal_mu=parse_int(argv[5],"shell mu order");
        shell_config.average_grid_orientation=false;
        shell_config.radial_measure=shell::ShellRadialMeasure::FftLattice;
        shell_config.fft_box_size=1000.0;
        shell_config.fft_mesh_size=256;
        eft::DirectIntegrationConfig loop;
        loop.n_radial=parse_int(argv[6],"loop radial order");
        loop.n_mu=parse_int(argv[7],"loop mu order");
        loop.n_phi=parse_int(argv[8],"loop phi order");
        loop.qmin=parse_double(argv[9],"qmin");
        loop.qmax=parse_double(argv[10],"qmax");
        loop.ir_safe=true;
        loop.uv_subtract=true;
        loop.exploit_phi_reflection=true;
        if (argc==23 || hybrid_mode) {
            const std::string angular_rule=
                argv[hybrid_mode?23:22];
            if (angular_rule=="tensor-gauss") {
                loop.angular_rule=
                    eft::DirectAngularRule::TensorGaussLegendre;
            } else if (angular_rule=="fibonacci") {
                loop.angular_rule=
                    eft::DirectAngularRule::AntipodalFibonacci;
                loop.exploit_phi_reflection=false;
            } else {
                throw std::invalid_argument(
                    "angular rule must be tensor-gauss or fibonacci");
            }
        }
        eft::StochasticIntegrationConfig stochastic;
        stochastic.n_radial=parse_int(argv[11],"stochastic radial order");
        stochastic.n_mu=parse_int(argv[12],"stochastic mu order");
        stochastic.qmin=loop.qmin;
        stochastic.qmax=parse_double(argv[13],"stochastic qmax");
        loop.uv.mu_ren=parse_double(argv[14],"renormalization scale");
        stochastic.mu_ren=loop.uv.mu_ren;
        eft::FftlogConfig hybrid_fftlog;
        eft::HybridShellIntegrationConfig hybrid_loop;
        if (hybrid_mode) {
            hybrid_fftlog.kmin=
                parse_double(argv[24],"hybrid FFTLog kmin");
            hybrid_fftlog.kmax=
                parse_double(argv[25],"hybrid FFTLog kmax");
            if (!(hybrid_fftlog.kmin>0.0
                  &&hybrid_fftlog.kmax>hybrid_fftlog.kmin)) {
                throw std::invalid_argument(
                    "hybrid FFTLog range must satisfy 0<kmin<kmax");
            }
            hybrid_fftlog.frequency_count=
                parse_int(argv[26],"hybrid FFTLog frequency count",8);
            hybrid_fftlog.reconstruction_grid_size=
                std::max(64,4*hybrid_fftlog.frequency_count);
            hybrid_fftlog.bias_nu=
                parse_double(argv[27],"hybrid FFTLog bias");
            hybrid_fftlog.reconstruct_interpolated_power=false;
            hybrid_fftlog.endpoint_inclusive_sampling=true;
            hybrid_loop.convolution=loop;
            hybrid_loop.factorized_p13.qmin=loop.qmin;
            hybrid_loop.factorized_p13.qmax=loop.qmax;
            hybrid_loop.factorized_p13.n_radial=
                parse_int(argv[28],"hybrid P13 radial order");
            hybrid_loop.factorized_p13.n_mu=
                parse_int(argv[29],"hybrid P13 mu order");
            hybrid_loop.factorized_p13.uv=loop.uv;
            hybrid_loop.analytic.regulator.coarse_epsilon=
                parse_double(argv[30],"hybrid coarse epsilon");
            hybrid_loop.analytic.regulator.refinement_ratio=
                parse_double(argv[31],"hybrid refinement ratio");
            if (!(hybrid_loop.analytic.regulator.coarse_epsilon>0.0
                  &&hybrid_loop.analytic.regulator.coarse_epsilon
                        <1.0e-3
                  &&hybrid_loop.analytic.regulator.refinement_ratio
                        >1.0)) {
                throw std::invalid_argument(
                    "hybrid regulator requires 0<epsilon<1e-3 "
                    "and refinement ratio>1");
            }
            hybrid_loop.analytic.uv_restoration.asymptotic=
                loop.uv;
            hybrid_loop.analytic_convolutions=
                analytic_hybrid_mode;
            hybrid_loop.analytic.contour_sectors.enabled=
                analytic_hybrid_mode;
            hybrid_loop.exact_lattice_multilevel=
                exact_lattice_hybrid_mode;
            if (exact_lattice_hybrid_mode) {
                hybrid_loop.expensive_lattice_radial_order=
                    parse_int(
                        argv[32],
                        "expensive lattice radial order");
                hybrid_loop.expensive_k3_interpolation_order=
                    parse_int(
                        argv[33],
                        "expensive k3 interpolation order");
            }
        }
        eft::IrResummationConfig ir_config;
        ir_config.lambda_ir=parse_double(argv[15],"IR cutoff");
        ir_config.sound_horizon=parse_double(argv[16],"sound horizon");
        const int sigma_mode=parse_int(argv[17],"Sigma power mode",0);
        const int split_mode=parse_int(argv[18],"smooth split mode",0);
        if (sigma_mode>1 || split_mode>1) {
            throw std::invalid_argument("IR Sigma-power and split modes must be zero or one");
        }
        ir_config.sigma_power=sigma_mode==0
            ?eft::SigmaPowerPrescription::FullP11
            :eft::SigmaPowerPrescription::NoWiggle;
        ir_config.split=split_mode==0
            ?eft::SmoothSplitPrescription::GaussianEisensteinHuRatio
            :eft::SmoothSplitPrescription::GaussianLogK3Power;
        const int radial_index_stop=parse_int(argv[19],"radial index stop");
        const int start=parse_int(argv[20],"start bin",0);
        const int stop=parse_int(argv[21],"stop bin",1);
        if (!(start>=0 && stop>start && stop<=120)) {
            throw std::invalid_argument("bin range must satisfy 0<=start<stop<=120");
        }
        if (radial_index_stop>15) {
            throw std::invalid_argument("radial index stop must be in [1,15]");
        }

        const std::vector<double> edges=read_edges(edge_path);
        TabulatedPower linear(power_path);
        const eft::IrResummation ir(linear,ir_config);
        const PowerSpectrum& loop_power=use_ir
            ?static_cast<const PowerSpectrum&>(ir.loop_power()):linear;
        const PowerSpectrum& tree_power=use_ir
            ?static_cast<const PowerSpectrum&>(ir.tree_power()):linear;
        std::unique_ptr<eft::FftlogDrOracle> b411_oracle;
        if (hybrid_mode) {
            b411_oracle=std::make_unique<eft::FftlogDrOracle>(
                loop_power,hybrid_fftlog);
        }
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\",\"schema\":\"marisa-b-eft-v2-r0-jsonl-v1\","
                 <<"\"ir_enabled\":"<<(use_ir?"true":"false")<<','
                 <<"\"ir_metadata\":";
        if (use_ir) std::cout<<ir.metadata_json(); else std::cout<<"null";
        std::cout<<",\"loop_evaluator\":"
                 <<(analytic_hybrid_mode
                    ?"\"fftlog_dr_contour_sector_factorized_p13\""
                    :exact_lattice_hybrid_mode
                        ?"\"hybrid_direct_factorized_p13_fftlog_dr_"
                         "exact_lattice_multilevel\""
                    :hybrid_mode
                        ?"\"hybrid_direct_factorized_p13_fftlog_dr\""
                        :"\"direct_subtracted\"")
                 <<','
                 <<"\"renormalization\":{\"scheme\":"
                 <<(analytic_hybrid_mode
                    ?"\"contour-sector FFTLog/DR B222/B321I/B411; "
                     "factorized renormalized tracer P13 B321II; "
                     "exact B411 and P13 UV restoration\""
                    :exact_lattice_hybrid_mode
                        ?"\"exact-joint-lattice multilevel projection; "
                         "direct IR-safe B222/B321I; factorized "
                         "renormalized tracer P13 B321II; regulated "
                         "FFTLog/DR B411 with exact UV restoration\""
                    :hybrid_mode
                        ?"\"direct IR-safe B222/B321I; factorized "
                         "renormalized tracer P13 B321II; regulated "
                         "FFTLog/DR B411 with exact UV restoration\""
                        :"\"explicit large-q asymptotic subtraction with "
                         "finite lower-bias and k^2 P^2 restoration\"")
                 <<','
                 <<"\"ir_safe_remapping\":"
                 <<(analytic_hybrid_mode?"false":"true")<<','
                 <<"\"uv_subtraction\":true,"
                 <<"\"uv_asymptotic_factor\":"
                 <<loop.uv.asymptotic_factor<<"},"
                 <<"\"implementation_version\":"
                 <<(analytic_hybrid_mode
                    ?"\"eft-v2-analytic-contour-p13-tail-v1\""
                    :exact_lattice_hybrid_mode
                        ?"\"eft-v2-exact-joint-lattice-multilevel-v1\""
                    :hybrid_mode
                        ?"\"eft-v2-hybrid-direct-p13-tail-fftlog-dr-v2\""
                        :"\"eft-v2-exact-opposite-pair-k4-direct-v2\"")
                 <<','
                 <<"\"parameter_registry_sha256\":\""
                 <<eft::registry_sha256()<<"\",\"bin_range\":["
                 <<start<<','<<stop<<"],\"shell_quadrature\":["
                 <<shell_config.n_radial<<','<<shell_config.n_internal_mu<<"],"
                 <<"\"shell_projection\":{\"measure\":\""
                 <<(exact_lattice_hybrid_mode
                        ?"exact joint float32 FFT-lattice k1-k2-mu"
                        :"independent FFT-lattice radial marginals "
                         "with continuum mu")
                 <<"\",\"normalization\":\"raw N1*N2 estimator pairs\","
                 <<"\"zero_external_legs_removed\":"
                 <<(exact_lattice_hybrid_mode?"true":"false")<<','
                 <<"\"closing_zero_atoms_removed\":"
                 <<(exact_lattice_hybrid_mode?"true":"false")<<','
                 <<"\"cheap_radial_order\":"
                 <<shell_config.n_radial<<','
                 <<"\"exact_angular_order\":"
                 <<shell_config.n_internal_mu<<','
                 <<"\"multilevel\":"
                 <<(exact_lattice_hybrid_mode?"true":"false")<<','
                 <<"\"expensive_radial_order\":"
                 <<(exact_lattice_hybrid_mode
                        ?hybrid_loop.expensive_lattice_radial_order
                        :0)
                 <<",\"expensive_k3_interpolation_order\":"
                 <<(exact_lattice_hybrid_mode
                        ?hybrid_loop
                            .expensive_k3_interpolation_order
                        :0)
                 <<"},"
                 <<"\"stochastic_cross_convention\":\"COBRA Eq. (62): one b1*Bshot*b_nabla2 term per cyclic external leg\","
                 <<"\"radial_index_stop\":"<<radial_index_stop<<','
                 <<"\"loop_quadrature\":["<<loop.n_radial<<','<<loop.n_mu<<','
                 <<loop.n_phi<<"],\"q_range\":["<<loop.qmin<<','<<loop.qmax
                 <<"],\"loop_phi_rule\":\""
                 <<(loop.angular_rule
                        ==eft::DirectAngularRule::TensorGaussLegendre
                    ?"Gauss-Legendre on [0,pi] with exact q_y reflection factor 2"
                    :"equal-weight full-sphere antipodal Fibonacci")
                 <<'\"'
                 <<",\"loop_angular_rule\":\""
                 <<eft::direct_angular_rule_name(loop.angular_rule)<<'\"'
                 <<",\"loop_angular_nodes\":"
                 <<(loop.angular_rule
                        ==eft::DirectAngularRule::TensorGaussLegendre
                    ?loop.n_mu*loop.n_phi
                    :std::max(32,2*loop.n_mu*loop.n_phi))
                 <<",\"hybrid_enabled\":"
                 <<(hybrid_mode?"true":"false");
        if (hybrid_mode) {
            std::cout
                 <<",\"hybrid\":{\"direct_diagram_mask\":"
                 <<(analytic_hybrid_mode
                        ?"[]":"[\"B222\",\"B321I\"]")
                 <<",\"analytic_convolutions\":"
                 <<(analytic_hybrid_mode?"true":"false")<<','
                 <<"\"r0_production_candidate\":"
                 <<(analytic_hybrid_mode?"false":"true")<<','
                 <<"\"validation_status\":"
                 <<(analytic_hybrid_mode
                        ?"\"experimental selected-oracle path; per-shell-node "
                         "B222 cost is not production-qualified\""
                        :exact_lattice_hybrid_mode
                            ?"\"exact-joint-lattice multilevel candidate "
                             "pending topology and full R0 gates\""
                            :"\"production candidate pending full R0 gates\"")
                 <<','
                 <<"\"b411_fftlog\":"
                 <<b411_oracle->power().metadata_json()<<','
                 <<"\"b411_regulator\":{\"coarse_epsilon\":"
                 <<hybrid_loop.analytic.regulator.coarse_epsilon
                 <<",\"refinement_ratio\":"
                 <<hybrid_loop.analytic.regulator.refinement_ratio
                 <<"},\"contour_sectors\":{\"enabled\":"
                 <<(hybrid_loop.analytic.contour_sectors.enabled
                        ?"true":"false")
                 <<",\"B222_default_bias\":"
                 <<hybrid_loop.analytic.contour_sectors
                        .B222_default_bias
                 <<",\"B222_constant_bias\":"
                 <<hybrid_loop.analytic.contour_sectors
                        .B222_constant_bias
                 <<",\"B321I_soft_bias\":"
                 <<hybrid_loop.analytic.contour_sectors
                        .B321I_soft_bias
                 <<",\"B321I_local_quadratic_bias\":"
                 <<hybrid_loop.analytic.contour_sectors
                        .B321I_local_quadratic_bias
                 <<",\"B321I_composite_bias\":"
                 <<hybrid_loop.analytic.contour_sectors
                        .B321I_composite_bias
                 <<",\"B411_bias\":"
                 <<hybrid_loop.analytic.contour_sectors.B411_bias
                 <<"},\"b411_uv_restoration\":{\"enabled\":"
                 <<(hybrid_loop.analytic.uv_restoration.enabled
                        ?"true":"false")
                 <<",\"moment_k_range\":["
                 <<hybrid_loop.analytic.uv_restoration.moment_kmin
                 <<','
                 <<hybrid_loop.analytic.uv_restoration.moment_kmax
                 <<"],\"moment_quadrature_order\":"
                 <<hybrid_loop.analytic.uv_restoration
                        .moment_quadrature_order
                 <<"},\"factorized_p13\":{\"q_range\":["
                 <<hybrid_loop.factorized_p13.qmin<<','
                 <<hybrid_loop.factorized_p13.qmax
                 <<"],\"quadrature\":["
                 <<hybrid_loop.factorized_p13.n_radial<<','
                 <<hybrid_loop.factorized_p13.n_mu
                 <<"],\"uv_tail_restoration\":{\"enabled\":"
                 <<(hybrid_loop.factorized_p13.restore_p13_uv_tail
                        ?"true":"false")
                 <<",\"sampled_kmax\":"
                 <<hybrid_loop.factorized_p13.uv_tail_kmax
                 <<",\"quadrature_order\":"
                 <<hybrid_loop.factorized_p13.uv_tail_quadrature_order
                 <<"}}}";
        }
        std::cout
                 <<",\"stochastic_quadrature\":["<<stochastic.n_radial<<','
                 <<stochastic.n_mu<<"],\"stochastic_qmax\":"<<stochastic.qmax
                 <<",\"stochastic_p13_uv_tail_restoration\":{\"enabled\":"
                 <<(stochastic.restore_p13_uv_tail?"true":"false")
                 <<",\"sampled_kmax\":"<<stochastic.uv_tail_kmax
                 <<",\"quadrature_order\":"
                 <<stochastic.uv_tail_quadrature_order<<"}"
                 <<",\"mu_ren\":"<<loop.uv.mu_ren<<"}\n";

        const eft::EftBiasKernelProvider provider;
        int flat=0;
        for (int first=0;first<15;++first) {
            for (int second=first;second<15;++second,++flat) {
                if (flat<start || flat>=stop) continue;
                if (first>=radial_index_stop || second>=radial_index_stop) continue;
                const shell::ShellBin bin{
                    edges[first],edges[first+1],edges[second],edges[second+1]};
                const auto value=hybrid_mode
                    ?eft::compute_eft_shell_templates_hybrid(
                        loop_power,tree_power,bin,provider,
                        *b411_oracle,hybrid_loop,shell_config,
                        stochastic,0.30)
                    :eft::compute_eft_shell_templates(
                        loop_power,tree_power,bin,provider,loop,
                        shell_config,stochastic,0.30);
                write_bin(flat,value);
                std::cout.flush();
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 R0 template driver failed: "<<error.what()<<"\n";
        return 1;
    }
}
