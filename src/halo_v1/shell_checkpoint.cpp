#include "shell_average.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"

#ifndef MARISA_B_GIT_COMMIT
#define MARISA_B_GIT_COMMIT "diagnostic"
#endif

#ifndef MARISA_B_TREE_SHA256
#define MARISA_B_TREE_SHA256 "diagnostic"
#endif

#ifndef MARISA_B_BUILD_KIND
#define MARISA_B_BUILD_KIND "diagnostic"
#endif

namespace hv1 = marisa_b_halo_v1;
namespace sv1 = marisa_b_shell_v1;

namespace {

class TabulatedPower final : public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table: " + path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double p = 0.0;
            if (row >> k >> p) {
                if (!(k > 0.0 && p > 0.0)) {
                    throw std::runtime_error("power table must contain positive k and P(k)");
                }
                log_k_.push_back(std::log(k));
                log_p_.push_back(std::log(p));
            }
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("power table has fewer than two rows");
        }
    }

    real Evaluate(real k) const override {
        const double log_k = std::log(std::max<double>(k, std::exp(log_k_.front())));
        if (log_k <= log_k_.front()) return interpolate(log_k, 0, 1);
        if (log_k >= log_k_.back()) {
            return interpolate(log_k, log_k_.size() - 2, log_k_.size() - 1);
        }
        const auto upper = std::upper_bound(log_k_.begin(), log_k_.end(), log_k);
        const std::size_t right = static_cast<std::size_t>(upper - log_k_.begin());
        return interpolate(log_k, right - 1, right);
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no Cosmology object");
    }

private:
    double interpolate(double log_k, std::size_t left, std::size_t right) const {
        const double fraction =
            (log_k - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_p_[left] + fraction * (log_p_[right] - log_p_[left]));
    }

    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

int parse_int(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const std::string value(text);
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < 1 || parsed > 1000000) {
        throw std::invalid_argument("invalid positive integer for " + name);
    }
    return static_cast<int>(parsed);
}

double parse_double(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const std::string value(text);
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid finite number for " + name);
    }
    return parsed;
}

void write_polynomial(const hv1::Polynomial& polynomial) {
    const std::vector<hv1::PolynomialTerm> terms = polynomial.terms();
    std::cout << "[";
    for (std::size_t index = 0; index < terms.size(); ++index) {
        if (index != 0) std::cout << ", ";
        const hv1::PolynomialTerm& term = terms[index];
        std::cout << "{\"powers\": ["
                  << static_cast<int>(term.powers.b1) << ", "
                  << static_cast<int>(term.powers.b2) << ", "
                  << static_cast<int>(term.powers.bK2)
                  << "], \"coefficient\": " << term.coefficient << "}";
    }
    std::cout << "]";
}

void write_result(
    const std::string& space,
    const sv1::ShellComponentValues& value,
    const hv1::IntegrationConfig& loop,
    const sv1::ShellQuadratureConfig& shell,
    const hv1::BiasPoint& bias,
    const sv1::ShellComponentTemplates* templates,
    double elapsed_seconds) {
    std::cout << std::setprecision(17);
    std::cout << "{\n"
              << "  \"build_kind\": \"" MARISA_B_BUILD_KIND "\",\n"
              << "  \"git_commit\": \"" MARISA_B_GIT_COMMIT "\",\n"
              << "  \"tree_sha256\": \"" MARISA_B_TREE_SHA256 "\",\n"
              << "  \"integration_product\": \"bias_templates\",\n"
              << "  \"space\": \"" << space << "\",\n"
              << "  \"bin\": [" << value.bin.k1_lower << ", "
              << value.bin.k1_upper << ", " << value.bin.k2_lower << ", "
              << value.bin.k2_upper << "],\n"
              << "  \"bias\": [" << bias.b1 << ", " << bias.b2 << ", "
              << bias.bK2 << "],\n"
              << "  \"loop_quadrature\": [" << loop.n_radial << ", "
              << loop.n_mu << ", " << loop.n_phi << "],\n"
              << "  \"q_range\": [" << loop.qmin << ", " << loop.qmax << "],\n"
              << "  \"shell_quadrature\": [" << shell.n_radial << ", "
              << shell.n_internal_mu << ", " << shell.n_alpha << ", "
              << shell.n_cos_beta << ", " << shell.n_gamma << "],\n"
              << "  \"shell_radial_measure\": \""
              << (shell.radial_measure == sv1::ShellRadialMeasure::FftLattice
                      ? "fft_lattice" : "continuum_volume")
              << "\",\n"
              << "  \"shell_fft_geometry\": [" << shell.fft_box_size << ", "
              << shell.fft_mesh_size << "],\n"
              << "  \"grid_orientation_averaged\": "
              << (value.grid_orientation_averaged ? "true" : "false") << ",\n"
              << "  \"shell_nodes\": " << value.shell_nodes << ",\n"
              << "  \"total_loop_nodes\": " << value.total_loop_nodes << ",\n"
              << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
              << "  \"components\": {\n"
              << "    \"tree\": " << value.tree << ",\n"
              << "    \"B222\": " << value.B222 << ",\n"
              << "    \"B321I\": " << value.B321I << ",\n"
              << "    \"B321II\": " << value.B321II << ",\n"
              << "    \"B411\": " << value.B411 << ",\n"
              << "    \"one_loop\": " << value.one_loop << ",\n"
              << "    \"total\": " << value.total << ",\n"
              << "    \"stochastic_alpha3_raw\": "
              << value.stochastic_alpha3_raw << ",\n"
              << "    \"stochastic_alpha4_raw\": "
              << value.stochastic_alpha4_raw << "\n"
              << "  }";
    if (templates != nullptr) {
        std::cout << ",\n  \"polynomial_templates\": {\n"
                  << "    \"tree\": ";
        write_polynomial(templates->tree);
        std::cout << ",\n    \"B222\": ";
        write_polynomial(templates->B222);
        std::cout << ",\n    \"B321I\": ";
        write_polynomial(templates->B321I);
        std::cout << ",\n    \"B321II\": ";
        write_polynomial(templates->B321II);
        std::cout << ",\n    \"B411\": ";
        write_polynomial(templates->B411);
        std::cout << ",\n    \"one_loop\": ";
        write_polynomial(templates->one_loop);
        std::cout << ",\n    \"total\": ";
        write_polynomial(templates->total);
        std::cout << ",\n    \"stochastic_alpha3_raw\": ";
        write_polynomial(templates->stochastic_alpha3_raw);
        std::cout << ",\n    \"stochastic_alpha4_raw\": "
                  << templates->stochastic_alpha4_raw << "\n  }";
    }
    std::cout << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool emit_templates =
            argc == 21 && std::string(argv[20]) == "--emit-templates";
        if (argc != 20 && !emit_templates) {
            std::cerr
                << "usage: shell_checkpoint POWER_TABLE pre|post "
                << "K1_LO K1_HI K2_LO K2_HI "
                << "SHELL_NRAD SHELL_NMU NALPHA NBETA NGAMMA "
                << "LOOP_NRAD LOOP_NMU LOOP_NPHI QMIN QMAX B1 B2 BK2 "
                << "[--emit-templates]\n";
            return 2;
        }
        const std::string space(argv[2]);
        if (space != "pre" && space != "post") {
            throw std::invalid_argument("space must be pre or post");
        }
        const sv1::ShellBin bin{
            parse_double(argv[3], "k1 lower"),
            parse_double(argv[4], "k1 upper"),
            parse_double(argv[5], "k2 lower"),
            parse_double(argv[6], "k2 upper")};
        sv1::ShellQuadratureConfig shell;
        shell.n_radial = parse_int(argv[7], "shell radial order");
        shell.n_internal_mu = parse_int(argv[8], "shell mu order");
        shell.n_alpha = parse_int(argv[9], "alpha order");
        shell.n_cos_beta = parse_int(argv[10], "cos beta order");
        shell.n_gamma = parse_int(argv[11], "gamma order");
        shell.average_grid_orientation = true;
        shell.radial_measure = sv1::ShellRadialMeasure::FftLattice;
        shell.fft_box_size = 1000.0;
        shell.fft_mesh_size = 256;

        hv1::IntegrationConfig loop;
        loop.n_radial = parse_int(argv[12], "loop radial order");
        loop.n_mu = parse_int(argv[13], "loop mu order");
        loop.n_phi = parse_int(argv[14], "loop phi order");
        loop.qmin = parse_double(argv[15], "qmin");
        loop.qmax = parse_double(argv[16], "qmax");
        loop.p13_epsrel = 1.0e-6;
        loop.p13_epsabs = 1.0e-12;
        loop.reconstruction.enabled = space == "post";
        const hv1::BiasPoint bias{
            parse_double(argv[17], "b1"),
            parse_double(argv[18], "b2"),
            parse_double(argv[19], "bK2")};

        TabulatedPower power(argv[1]);
        const auto started = std::chrono::steady_clock::now();
        const sv1::ShellComponentTemplates templates =
            sv1::compute_shell_templates(power, bin, loop, shell);
        const sv1::ShellComponentValues value =
            sv1::evaluate_shell_templates(templates, bias);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        write_result(
            space, value, loop, shell, bias,
            emit_templates ? &templates : nullptr, elapsed);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shell checkpoint failed: " << error.what() << "\n";
        return 1;
    }
}
