#ifndef MARISA_B_SHELL_AVERAGE_H
#define MARISA_B_SHELL_AVERAGE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "halo_v1.h"

class PowerSpectrum;

namespace marisa_b_shell_v1 {

struct ShellBin {
    double k1_lower = 0.0;
    double k1_upper = 0.0;
    double k2_lower = 0.0;
    double k2_upper = 0.0;
};

enum class ShellRadialMeasure {
    ContinuumVolume,
    FftLattice,
};

enum class ShellFftLatticeBinning {
    Float32CartesianRadius,
    EstimatorModeCountConstrainedIntegerRadius,
};

struct ShellQuadratureConfig {
    int n_radial = 3;
    int n_internal_mu = 12;
    int n_alpha = 4;
    int n_cos_beta = 3;
    int n_gamma = 4;
    bool average_grid_orientation = true;
    ShellRadialMeasure radial_measure = ShellRadialMeasure::ContinuumVolume;
    double fft_box_size = 1000.0;
    int fft_mesh_size = 256;
    bool cap_fft_lattice_radial_order_to_support = false;
    ShellFftLatticeBinning fft_lattice_binning =
        ShellFftLatticeBinning::Float32CartesianRadius;
    std::uint64_t fft_lattice_expected_mode_count = 0;
};

struct ShellNode {
    std::array<marisa_b_halo_v1::Vec3, 3> closed_vectors;
    double k1 = 0.0;
    double k2 = 0.0;
    double internal_mu = 0.0;
    double alpha = 0.0;
    double cos_beta = 1.0;
    double gamma = 0.0;
    double weight = 0.0;
};

struct RadialShellNode {
    double k = 0.0;
    double weight = 0.0;
};

struct RadialShellRule {
    std::vector<RadialShellNode> nodes;
    int requested_radial_order = 0;
    std::size_t actual_radial_order = 0;
    std::uint64_t lattice_mode_count = 0;
    std::size_t lattice_unique_radius_count = 0;
    std::int64_t lattice_boundary_mode_adjustment = 0;
    bool fft_lattice_measure = false;
};

struct ShellComponentTemplates {
    ShellBin bin;
    marisa_b_halo_v1::Polynomial tree;
    marisa_b_halo_v1::Polynomial B222;
    marisa_b_halo_v1::Polynomial B321I;
    marisa_b_halo_v1::Polynomial B321II;
    marisa_b_halo_v1::Polynomial B411;
    marisa_b_halo_v1::Polynomial one_loop;
    marisa_b_halo_v1::Polynomial total;
    marisa_b_halo_v1::Polynomial stochastic_alpha3_raw;
    double stochastic_alpha4_raw = 1.0;
    std::size_t shell_nodes = 0;
    std::uint64_t total_loop_nodes = 0;
    bool grid_orientation_averaged = false;
};

struct ShellComponentValues {
    ShellBin bin;
    double tree = 0.0;
    double B222 = 0.0;
    double B321I = 0.0;
    double B321II = 0.0;
    double B411 = 0.0;
    double one_loop = 0.0;
    double total = 0.0;
    double stochastic_alpha3_raw = 0.0;
    double stochastic_alpha4_raw = 1.0;
    std::size_t shell_nodes = 0;
    std::uint64_t total_loop_nodes = 0;
    bool grid_orientation_averaged = false;
};

std::vector<ShellNode> make_shell_nodes(
    const ShellBin& bin,
    const ShellQuadratureConfig& config);

std::vector<RadialShellNode> make_radial_shell_nodes(
    double lower,
    double upper,
    const ShellQuadratureConfig& config);

RadialShellRule make_radial_shell_rule(
    double lower,
    double upper,
    const ShellQuadratureConfig& config);

ShellComponentTemplates compute_shell_templates(
    const PowerSpectrum& linear_power,
    const ShellBin& bin,
    const marisa_b_halo_v1::IntegrationConfig& loop_config,
    const ShellQuadratureConfig& shell_config);

ShellComponentValues compute_shell_direct(
    const PowerSpectrum& linear_power,
    const ShellBin& bin,
    const marisa_b_halo_v1::IntegrationConfig& loop_config,
    const ShellQuadratureConfig& shell_config,
    const marisa_b_halo_v1::BiasPoint& bias);

ShellComponentValues evaluate_shell_templates(
    const ShellComponentTemplates& templates,
    const marisa_b_halo_v1::BiasPoint& bias);

}  // namespace marisa_b_shell_v1

#endif  // MARISA_B_SHELL_AVERAGE_H
