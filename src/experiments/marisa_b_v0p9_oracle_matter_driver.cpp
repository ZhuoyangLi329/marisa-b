#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "diagram_assembler.h"
#include "direct_evaluator.h"
#include "field_kernel_provider.h"
#include "lattice_shell_rule.h"
#include "parameter_registry.h"
#include "shell_projector.h"
#include "stochastic.h"

namespace eft = marisa_b_eft_v2;
namespace shell = marisa_b_shell_v1;

namespace {

constexpr double kBoxSize = 1000.0;
constexpr int kMeshSize = 256;
constexpr int kRadialStop = 7;
constexpr double kSmoothingRadius = 15.0;
constexpr double kHaloReconstructionBias = 2.7340475186190334;
constexpr double kHaloReconstructionPhiBias = 5.84720823278338;
constexpr double kCellSize = 8.0;
constexpr double kReconstructionMinimumK = 2.0 * 3.14159265358979323846 / kBoxSize;

class LinearBiasMatterKernelProvider final : public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        return eft::spt_F(momenta)
               * eft::SparsePolynomial::variable(eft::ParameterId::B1);
    }

    std::string_view name() const noexcept override {
        return "linear_bias_times_matter_spt";
    }
};

constexpr eft::ParameterId kOracleMarker=eft::ParameterId::Gamma31;

bool contains_parameter(
    const eft::SparsePolynomial& polynomial,
    eft::ParameterId parameter) {
    for (const auto& term:polynomial.terms()) {
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==parameter) return true;
        }
    }
    return false;
}

eft::SparsePolynomial build_term(
    double coefficient,
    const std::vector<eft::MonomialFactor>& factors) {
    eft::SparsePolynomial result=
        eft::SparsePolynomial::constant(coefficient);
    for (const eft::MonomialFactor& factor:factors) {
        for (int power=0;power<static_cast<int>(factor.power);++power) {
            result=result*eft::SparsePolynomial::variable(factor.id);
        }
    }
    return result;
}

eft::SparsePolynomial factor_out_one_b1(
    const eft::SparsePolynomial& polynomial) {
    eft::SparsePolynomial result;
    for (const auto& term:polynomial.terms()) {
        bool found=false;
        std::vector<eft::MonomialFactor> factors;
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==eft::ParameterId::B1) {
                if (factor.power<1) {
                    throw std::logic_error("invalid zero b1 power");
                }
                found=true;
                if (factor.power>1) {
                    factors.push_back(eft::MonomialFactor{
                        factor.id,
                        static_cast<std::uint8_t>(factor.power-1)});
                }
            } else {
                factors.push_back(factor);
            }
        }
        if (!found) {
            throw std::logic_error(
                "oracle tagged kernel term lacks the exact common b1 factor");
        }
        result+=build_term(term.second,factors);
    }
    return result;
}

class TaggedOracleMatterKernelProvider final : public eft::FieldKernelProvider {
public:
    TaggedOracleMatterKernelProvider(
        const eft::FieldKernelProvider& base,
        const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
        const PowerSpectrum& transfer)
        :base_(base),reconstruction_(reconstruction),transfer_(transfer) {
        if (reconstruction_.local_png_bias_transfer!=nullptr
            ||reconstruction_.local_png_bias_amplitude!=0.0) {
            throw std::invalid_argument(
                "tagged oracle provider needs a fixed reconstruction map");
        }
    }

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        const eft::FieldKernelVariation variation=
            eft::reconstructed_field_kernel_local_png_denominator_variation(
                base_,reconstruction_,transfer_,
                kReconstructionMinimumK,momenta);
        if (contains_parameter(variation.value,kOracleMarker)
            ||contains_parameter(variation.direction,kOracleMarker)) {
            throw std::logic_error("oracle marker collides with physical kernel");
        }
        const eft::SparsePolynomial value=
            factor_out_one_b1(variation.value);
        eft::SparsePolynomial direction;
        if (!variation.direction.empty()) {
            direction=
                kHaloReconstructionPhiBias
                *factor_out_one_b1(variation.direction);
        }
        return value
               +direction*eft::SparsePolynomial::variable(kOracleMarker);
    }

    std::string_view name() const noexcept override {
        return "tagged_oracle_linear_bias_matter_spt";
    }

    bool supports_triangle_plane_reflection() const noexcept override {
        return false;
    }

private:
    const eft::FieldKernelProvider& base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    const PowerSpectrum& transfer_;
};

eft::SparsePolynomial extract_marker_power_and_restore_b1_cubed(
    const eft::SparsePolynomial& polynomial,
    int requested_marker_power) {
    if (requested_marker_power<0) {
        throw std::invalid_argument("negative oracle marker power");
    }
    eft::SparsePolynomial selected;
    for (const auto& term:polynomial.terms()) {
        int marker_power=0;
        std::vector<eft::MonomialFactor> factors;
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==kOracleMarker) {
                marker_power=static_cast<int>(factor.power);
            } else {
                factors.push_back(factor);
            }
        }
        if (marker_power==requested_marker_power) {
            selected+=build_term(term.second,factors);
        }
    }
    const eft::SparsePolynomial b1=
        eft::SparsePolynomial::variable(eft::ParameterId::B1);
    return selected*b1*b1*b1;
}

eft::EftShellTemplates compute_tagged_oracle_shell_templates(
    const PowerSpectrum& power,
    const shell::ShellBin& bin,
    const eft::FieldKernelProvider& provider,
    const eft::ReconstructedShellIntegrationConfig& integration) {
    const eft::HaarOrientedLatticeShellRule rule=
        eft::make_haar_oriented_lattice_shell_rule(bin,integration.shell);
    if (rule.nodes.empty()) {
        throw std::runtime_error("tagged oracle shell has no projection nodes");
    }
    eft::EftShellTemplates result;
    result.bin=bin;
    result.shell_nodes=rule.nodes.size();
    result.diagrams.ir_safe=integration.loop.ir_safe;
    for (const shell::ShellNode& node:rule.nodes) {
        eft::DiagramTemplates value=
            eft::evaluate_direct(
                power,node.closed_vectors,provider,integration.loop);
        const eft::DiagramAssembler tree(
            power,node.closed_vectors,provider);
        value.tree=tree.tree();
        const double weight=node.weight;
        result.diagrams.tree+=weight*value.tree;
        result.diagrams.B222+=weight*value.B222;
        result.diagrams.B321I+=weight*value.B321I;
        result.diagrams.B321II+=weight*value.B321II;
        result.diagrams.B411+=weight*value.B411;
        result.diagrams.B321II_bare+=weight*value.B321II_bare;
        result.diagrams.B411_bare+=weight*value.B411_bare;
        result.diagrams.B321II_uv_subtraction+=
            weight*value.B321II_uv_subtraction;
        result.diagrams.B411_uv_subtraction+=
            weight*value.B411_uv_subtraction;
        result.diagrams.B321II_uv_restoration+=
            weight*value.B321II_uv_restoration;
        result.diagrams.B411_uv_restoration+=
            weight*value.B411_uv_restoration;
        result.total_loop_nodes+=value.integration_nodes;
    }
    result.diagrams.uv_subtraction_total=
        result.diagrams.B321II_uv_subtraction
        +result.diagrams.B411_uv_subtraction;
    result.diagrams.uv_restoration_total=
        result.diagrams.B321II_uv_restoration
        +result.diagrams.B411_uv_restoration;
    result.diagrams.one_loop=
        result.diagrams.B222+result.diagrams.B321I
        +result.diagrams.B321II+result.diagrams.B411;
    result.diagrams.total=
        result.diagrams.tree+result.diagrams.one_loop;
    result.diagrams.integration_nodes=result.total_loop_nodes;
    result.fft_lattice_radial_measure=true;
    result.exact_k1_k2_mu_lattice_measure=true;
    result.haar_oriented_cic_projection=true;
    result.invariant_shell_nodes=rule.diagnostics.invariant_nodes;
    result.orientation_nodes=rule.diagnostics.orientation_nodes;
    result.exact_lattice_radial_order=
        integration.shell.exact.radial_order;
    result.exact_lattice_angular_order=
        integration.shell.exact.angular_order;
    result.zero_external_leg_pairs=
        rule.diagnostics.exact.zero_external_leg_pairs;
    result.closing_zero_pairs=
        rule.diagnostics.exact.closing_zero_pairs;
    result.exact_lattice_valid_pair_fraction=
        rule.diagnostics.exact.valid_pair_fraction;
    result.exact_lattice_total_variation=
        rule.diagnostics.exact.total_variation;
    return result;
}

std::string file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot hash input: " + path);
    }
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return eft::sha256_hex(payload);
}

class TabulatedPower final : public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open power table: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double value = 0.0;
            if (!(row >> k >> value) || !(k > 0.0 && value > 0.0)) {
                throw std::runtime_error("invalid positive power-table row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(value));
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("power table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k > 0.0)) return 0.0;
        const double x = std::log(k);
        std::size_t right = 1;
        if (x >= log_k_.back()) {
            right = log_k_.size() - 1;
        } else if (x > log_k_.front()) {
            right = static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(), log_k_.end(), x)
                - log_k_.begin());
        }
        const std::size_t left = right - 1;
        const double fraction =
            (x - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_value_[left]
            + fraction * (log_value_[right] - log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("tabulated power has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

class TabulatedTransfer final : public PowerSpectrum {
public:
    explicit TabulatedTransfer(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open local-PNG table: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double power = 0.0;
            double ignored = 0.0;
            double transfer = 0.0;
            if (!(row >> k >> power >> ignored >> transfer)
                || !(k > 0.0 && transfer > 0.0)) {
                throw std::runtime_error("invalid local-PNG transfer row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(transfer));
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("local-PNG table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k > 0.0)) return 0.0;
        const double x = std::log(k);
        std::size_t right = 1;
        if (x >= log_k_.back()) {
            right = log_k_.size() - 1;
        } else if (x > log_k_.front()) {
            right = static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(), log_k_.end(), x)
                - log_k_.begin());
        }
        const std::size_t left = right - 1;
        const double fraction =
            (x - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_value_[left]
            + fraction * (log_value_[right] - log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("tabulated transfer has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

int parse_int(const char* text, const std::string& name, int minimum) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size() || value < minimum) {
        throw std::invalid_argument("invalid integer for " + name);
    }
    return value;
}

double parse_double(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid number for " + name);
    }
    return value;
}

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open edge file: " + path);
    }
    std::vector<double> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        double value = 0.0;
        if (!(row >> value)) {
            throw std::runtime_error("invalid edge row");
        }
        result.push_back(value);
    }
    if (result.size() != 16) {
        throw std::runtime_error("edge file must contain 16 values");
    }
    for (std::size_t index = 1; index < result.size(); ++index) {
        if (!(result[index] > result[index - 1])) {
            throw std::runtime_error("edges must be strictly increasing");
        }
    }
    return result;
}

void write_json_string(const std::string& value) {
    std::cout << '"';
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            std::cout << '\\' << static_cast<char>(character);
        } else if (character < 0x20) {
            const auto flags = std::cout.flags();
            const char fill = std::cout.fill();
            std::cout << "\\u" << std::hex << std::setw(4)
                      << std::setfill('0') << static_cast<int>(character);
            std::cout.flags(flags);
            std::cout.fill(fill);
        } else {
            std::cout << static_cast<char>(character);
        }
    }
    std::cout << '"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout << '{';
    bool first = true;
    for (const auto& term : polynomial.terms()) {
        if (!first) std::cout << ',';
        first = false;
        write_json_string(term.first.canonical_string());
        std::cout << ':' << term.second;
    }
    std::cout << '}';
}

struct LeadingBshotShellShapes {
    double ashot=0.0;
    eft::SparsePolynomial pre;
    eft::SparsePolynomial post;
};

LeadingBshotShellShapes leading_bshot_shell_shapes(
    const PowerSpectrum& power,
    const shell::ShellBin& bin,
    const eft::FieldKernelProvider& provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const eft::HaarOrientedLatticeShellRuleConfig& config) {
    const eft::HaarOrientedLatticeShellRule rule =
        eft::make_haar_oriented_lattice_shell_rule(bin, config);
    if (rule.nodes.empty()) {
        throw std::runtime_error("leading Bshot shell has no projection nodes");
    }
    LeadingBshotShellShapes result;
    const eft::SparsePolynomial b1_squared =
        eft::SparsePolynomial::variable(eft::ParameterId::B1)
        * eft::SparsePolynomial::variable(eft::ParameterId::B1);
    for (const shell::ShellNode& node : rule.nodes) {
        result.ashot+=node.weight;
        double power_sum = 0.0;
        for (const eft::Vec3& wavevector : node.closed_vectors) {
            power_sum += power(marisa_b_halo_v1::norm(wavevector));
        }
        result.pre += node.weight * power_sum * b1_squared;
        result.post += node.weight
                       * eft::reconstructed_leading_bshot_residual_generating_shape(
                           power,
                           node.closed_vectors,
                           provider,
                           reconstruction,
                           false);
    }
    return result;
}

void write_bin(
    int index,
    const eft::EftShellTemplates& value,
    const LeadingBshotShellShapes& leading_bshot) {
    std::cout << std::setprecision(17)
              << "{\"record\":\"bin\",\"index\":" << index
              << ",\"edges\":[" << value.bin.k1_lower << ','
              << value.bin.k1_upper << ',' << value.bin.k2_lower << ','
              << value.bin.k2_upper << ']'
              << ",\"shell_nodes\":" << value.shell_nodes
              << ",\"invariant_shell_nodes\":" << value.invariant_shell_nodes
              << ",\"orientation_nodes\":" << value.orientation_nodes
              << ",\"total_loop_nodes\":" << value.total_loop_nodes
              << ",\"zero_external_leg_pairs\":" << value.zero_external_leg_pairs
              << ",\"closing_zero_pairs\":" << value.closing_zero_pairs
              << ",\"valid_pair_fraction\":"
              << value.exact_lattice_valid_pair_fraction
              << ",\"tree\":";
    write_polynomial(value.diagrams.tree);
    std::cout << ",\"B222\":";
    write_polynomial(value.diagrams.B222);
    std::cout << ",\"B321I\":";
    write_polynomial(value.diagrams.B321I);
    std::cout << ",\"B321II\":";
    write_polynomial(value.diagrams.B321II);
    std::cout << ",\"B411\":";
    write_polynomial(value.diagrams.B411);
    std::cout << ",\"B321II_bare\":";
    write_polynomial(value.diagrams.B321II_bare);
    std::cout << ",\"B411_bare\":";
    write_polynomial(value.diagrams.B411_bare);
    std::cout << ",\"B321II_uv_subtraction\":";
    write_polynomial(value.diagrams.B321II_uv_subtraction);
    std::cout << ",\"B411_uv_subtraction\":";
    write_polynomial(value.diagrams.B411_uv_subtraction);
    std::cout << ",\"B321II_uv_restoration\":";
    write_polynomial(value.diagrams.B321II_uv_restoration);
    std::cout << ",\"B411_uv_restoration\":";
    write_polynomial(value.diagrams.B411_uv_restoration);
    std::cout << ",\"Ashot_residual\":";
    write_polynomial(
        eft::SparsePolynomial::constant(leading_bshot.ashot));
    std::cout << ",\"Bshot_residual\":";
    write_polynomial(leading_bshot.post);
    std::cout << ",\"pre_Bshot_residual\":";
    write_polynomial(leading_bshot.pre);
    std::cout << "}\n";
    std::cout.flush();
}

void write_tagged_pair(
    const std::string& name,
    const eft::SparsePolynomial& tagged) {
    std::cout << ',';
    write_json_string(name+"_zero");
    std::cout << ':';
    write_polynomial(
        extract_marker_power_and_restore_b1_cubed(tagged,0));
    std::cout << ',';
    write_json_string(name+"_direction");
    std::cout << ':';
    write_polynomial(
        extract_marker_power_and_restore_b1_cubed(tagged,1));
}

void write_analytic_bin(
    int index,
    const eft::EftShellTemplates& value,
    const LeadingBshotShellShapes& leading_bshot) {
    std::cout << std::setprecision(17)
              << "{\"record\":\"bin\",\"index\":" << index
              << ",\"edges\":[" << value.bin.k1_lower << ','
              << value.bin.k1_upper << ',' << value.bin.k2_lower << ','
              << value.bin.k2_upper << ']'
              << ",\"shell_nodes\":" << value.shell_nodes
              << ",\"invariant_shell_nodes\":" << value.invariant_shell_nodes
              << ",\"orientation_nodes\":" << value.orientation_nodes
              << ",\"total_loop_nodes\":" << value.total_loop_nodes
              << ",\"zero_external_leg_pairs\":" << value.zero_external_leg_pairs
              << ",\"closing_zero_pairs\":" << value.closing_zero_pairs
              << ",\"valid_pair_fraction\":"
              << value.exact_lattice_valid_pair_fraction;
    write_tagged_pair("tree",value.diagrams.tree);
    write_tagged_pair("B222",value.diagrams.B222);
    write_tagged_pair("B321I",value.diagrams.B321I);
    write_tagged_pair("B321II",value.diagrams.B321II);
    write_tagged_pair("B411",value.diagrams.B411);
    write_tagged_pair(
        "B321II_bare",value.diagrams.B321II_bare);
    write_tagged_pair(
        "B411_bare",value.diagrams.B411_bare);
    write_tagged_pair(
        "B321II_uv_subtraction",
        value.diagrams.B321II_uv_subtraction);
    write_tagged_pair(
        "B411_uv_subtraction",
        value.diagrams.B411_uv_subtraction);
    write_tagged_pair(
        "B321II_uv_restoration",
        value.diagrams.B321II_uv_restoration);
    write_tagged_pair(
        "B411_uv_restoration",
        value.diagrams.B411_uv_restoration);
    std::cout << ",\"Ashot_residual_zero\":";
    write_polynomial(
        eft::SparsePolynomial::constant(leading_bshot.ashot));
    std::cout << ",\"Ashot_residual_direction\":";
    write_polynomial(eft::SparsePolynomial());
    std::cout << ",\"Bshot_residual_zero\":";
    write_polynomial(leading_bshot.post);
    std::cout << ",\"Bshot_residual_direction\":";
    write_polynomial(eft::SparsePolynomial());
    std::cout << ",\"pre_Bshot_residual_zero\":";
    write_polynomial(leading_bshot.pre);
    std::cout << "}\n";
    std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr
                << "usage: marisa_b_v0p9_oracle_matter_driver "
                << "POWER_TABLE EDGE_FILE PNG_TABLE START_BIN STOP_BIN "
                << "FNL_REC_OR_ANALYTIC\n";
            return 2;
        }
        const std::string power_path = argv[1];
        const std::string edge_path = argv[2];
        const std::string png_path = argv[3];
        const int start = parse_int(argv[4], "start bin", 0);
        const int stop = parse_int(argv[5], "stop bin", 1);
        const bool analytic_direction=std::string(argv[6])=="analytic";
        const double fnl_rec=analytic_direction
            ?0.0:parse_double(argv[6], "fNL_rec");
        if (start < 0 || stop <= start || stop > 120) {
            throw std::invalid_argument("invalid bin range");
        }

        eft::ReconstructedShellIntegrationConfig integration;
        integration.shell.exact.radial_order = 2;
        integration.shell.exact.angular_order = 8;
        integration.shell.exact.fft_box_size = kBoxSize;
        integration.shell.exact.fft_mesh_size = kMeshSize;
        integration.shell.n_alpha = 2;
        integration.shell.n_cos_beta = 2;
        integration.shell.n_gamma = 2;
        integration.loop.n_radial = 8;
        integration.loop.n_mu = 32;
        integration.loop.n_phi = 24;
        integration.loop.qmin = 0.0001;
        integration.loop.qmax = 6.4;
        integration.loop.restore_uv_tail = true;
        integration.loop.uv_tail_kmax = 30.0;
        integration.loop.ir_safe = true;
        integration.loop.uv_subtract = true;
        integration.loop.exploit_phi_reflection = false;
        integration.loop.angular_rule = eft::DirectAngularRule::AntipodalFibonacci;
        integration.include_reconstructed_stochastic = false;
        integration.include_reconstructed_fixed_poisson = false;
        integration.include_bare_fixed_poisson_one_loop = false;

        eft::StochasticIntegrationConfig stochastic;
        stochastic.qmin = integration.loop.qmin;
        stochastic.qmax = integration.loop.qmax;
        stochastic.n_radial = integration.loop.n_radial;
        stochastic.n_mu = integration.loop.n_mu;
        stochastic.n_phi = integration.loop.n_phi;
        stochastic.restore_p13_uv_tail = true;
        stochastic.uv_tail_kmax = integration.loop.uv_tail_kmax;

        const TabulatedPower power(power_path);
        const TabulatedTransfer transfer(png_path);
        const std::vector<double> edges = read_edges(edge_path);
        marisa_b_halo_v1::ReconstructionConfig reconstruction;
        reconstruction.enabled = true;
        reconstruction.smoothing_radius = kSmoothingRadius;
        reconstruction.bias_recon = kHaloReconstructionBias;
        reconstruction.cell_size = kCellSize;
        const marisa_b_halo_v1::ReconstructionConfig fixed_reconstruction=
            reconstruction;
        if (!analytic_direction) {
            reconstruction.local_png_bias_transfer = &transfer;
            reconstruction.local_png_bias_amplitude =
                fnl_rec * kHaloReconstructionPhiBias;
            reconstruction.local_png_bias_kmin = kReconstructionMinimumK;
        }

        const LinearBiasMatterKernelProvider matter;
        const TaggedOracleMatterKernelProvider tagged(
            matter,fixed_reconstruction,transfer);
        std::cout << std::setprecision(17)
                  << "{\"record\":\"header\","
                  << "\"schema\":\""
                  << (analytic_direction
                      ?"marisa-b-v0p9-oracle-pure-b1-analytic-v1"
                      :"marisa-b-v0p9-oracle-pure-b1-finite-v2")
                  << "\","
                  << "\"model\":\"pure-linear-bias matter SPT tree plus one-loop and selected stochastic\","
                  << "\"production_candidate\":false,"
                  << "\"bin_range\":[" << start << ',' << stop << "],"
                  << "\"radial_stop\":" << kRadialStop << ','
                  << "\"fnl_rec\":" << fnl_rec << ','
                  << "\"analytic_direction\":"
                  << (analytic_direction?"true":"false") << ','
                  << "\"auxiliary_marker\":\"gamma31\","
                  << "\"gamma31_is_auxiliary_marker\":"
                  << (analytic_direction?"true":"false") << ','
                  << "\"per_kernel_common_b1_factored\":"
                  << (analytic_direction?"true":"false") << ','
                  << "\"b_rec_h\":" << reconstruction.bias_recon << ','
                  << "\"bphi_rec\":" << kHaloReconstructionPhiBias << ','
                  << "\"kmin\":" << kReconstructionMinimumK << ','
                  << "\"shell_orders\":[2,8,2,2,2],"
                  << "\"loop_orders\":[8,32,24],"
                  << "\"pure_b1_polynomial\":true,"
                  << "\"selected_stochastic\":[\"Ashot_residual\",\"Bshot_residual\"],"
                  << "\"q_range\":[0.0001,6.4],"
                  << "\"uv_tail_kmax\":30,"
                  << "\"source_hashes\":{"
                  << "\"linear_power\":\"" << file_sha256(power_path) << "\","
                  << "\"edge_file\":\"" << file_sha256(edge_path) << "\","
                  << "\"png_table\":\"" << file_sha256(png_path) << "\","
                  << "\"driver_executable\":\"" << file_sha256(argv[0]) << "\"}"
                  << "}\n";
        std::cout.flush();

        int flat = 0;
        for (int first = 0; first < 15; ++first) {
            for (int second = first; second < 15; ++second, ++flat) {
                if (flat < start || flat >= stop) continue;
                if (first >= kRadialStop || second >= kRadialStop) continue;
                const shell::ShellBin bin{
                    edges[static_cast<std::size_t>(first)],
                    edges[static_cast<std::size_t>(first + 1)],
                    edges[static_cast<std::size_t>(second)],
                    edges[static_cast<std::size_t>(second + 1)]};
                const eft::EftShellTemplates value =analytic_direction
                    ?compute_tagged_oracle_shell_templates(
                        power,bin,tagged,integration)
                    :eft::compute_reconstructed_eft_shell_templates(
                        power, power, bin, matter, reconstruction,
                        integration, stochastic, 0.30);
                const LeadingBshotShellShapes leading_bshot =
                    leading_bshot_shell_shapes(
                        power,
                        bin,
                        matter,
                        reconstruction,
                        integration.shell);
                if (analytic_direction) {
                    write_analytic_bin(flat,value,leading_bshot);
                } else {
                    write_bin(flat, value, leading_bshot);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0.9 oracle matter driver failed: " << error.what() << '\n';
        return 1;
    }
}
