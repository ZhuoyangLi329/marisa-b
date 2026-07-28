#ifndef MARISA_B_EFT_V2_LATTICE_SHELL_RULE_H
#define MARISA_B_EFT_V2_LATTICE_SHELL_RULE_H

#include <cstdint>
#include <vector>

#include "shell_average.h"

namespace marisa_b_eft_v2 {

struct ExactLatticeShellRuleConfig {
    int radial_order = 4;
    int angular_order = 80;
    double fft_box_size = 1000.0;
    int fft_mesh_size = 256;
};

struct ExactLatticeShellDiagnostics {
    std::uint64_t first_full_modes = 0;
    std::uint64_t second_full_modes = 0;
    std::uint64_t first_nonzero_modes = 0;
    std::uint64_t second_nonzero_modes = 0;
    std::uint64_t zero_external_leg_pairs = 0;
    std::uint64_t closing_zero_pairs = 0;
    double valid_pair_fraction = 0.0;
    double total_weight = 0.0;
    double total_variation = 0.0;
    double maximum_absolute_weight = 0.0;
};

struct ExactLatticeShellRule {
    marisa_b_shell_v1::ShellBin bin;
    ExactLatticeShellRuleConfig config;
    std::vector<marisa_b_shell_v1::ShellNode> nodes;
    ExactLatticeShellDiagnostics diagnostics;
};

ExactLatticeShellRule make_exact_lattice_shell_rule(
    const marisa_b_shell_v1::ShellBin& bin,
    const ExactLatticeShellRuleConfig& config = {});

/*
 * The exact lattice rule fixes every representative triangle in the x-z
 * plane.  This is sufficient only for rotationally invariant kernels.  A
 * reconstructed field with a Cartesian CIC shift window is instead tied to
 * the FFT axes.  The rule below preserves the exact lattice k1-k2-mu measure
 * (including zero-mode atoms in the estimator denominator), then averages
 * each representative over normalized Haar measure on SO(3).
 *
 * This is deliberately named "Haar-oriented", rather than "exact lattice":
 * the radial-angle marginal is exact to the configured polynomial order,
 * while the conditional orientation distribution is a controlled cubature.
 * Production use must be checked against direct cubic-lattice enumeration
 * for inexpensive tree/counterterm shapes.
 */
struct HaarOrientedLatticeShellRuleConfig {
    ExactLatticeShellRuleConfig exact;
    int n_alpha = 4;
    int n_cos_beta = 3;
    int n_gamma = 4;
};

struct HaarOrientedLatticeShellDiagnostics {
    ExactLatticeShellDiagnostics exact;
    std::size_t invariant_nodes = 0;
    std::size_t orientation_nodes = 0;
    double total_weight = 0.0;
    double total_variation = 0.0;
    double maximum_absolute_weight = 0.0;
};

struct HaarOrientedLatticeShellRule {
    marisa_b_shell_v1::ShellBin bin;
    HaarOrientedLatticeShellRuleConfig config;
    std::vector<marisa_b_shell_v1::ShellNode> nodes;
    HaarOrientedLatticeShellDiagnostics diagnostics;
};

HaarOrientedLatticeShellRule make_haar_oriented_lattice_shell_rule(
    const marisa_b_shell_v1::ShellBin& bin,
    const HaarOrientedLatticeShellRuleConfig& config = {});

struct MultilevelLatticeShellRuleConfig {
    ExactLatticeShellRuleConfig exact;
    int interpolation_order = 20;
};

struct MultilevelLatticeShellDiagnostics {
    ExactLatticeShellDiagnostics exact;
    double total_weight = 0.0;
    double total_variation = 0.0;
    double maximum_absolute_weight = 0.0;
    double maximum_sampled_lebesgue = 0.0;
};

struct MultilevelLatticeShellRule {
    marisa_b_shell_v1::ShellBin bin;
    MultilevelLatticeShellRuleConfig config;
    std::vector<marisa_b_shell_v1::ShellNode> nodes;
    MultilevelLatticeShellDiagnostics diagnostics;
};

MultilevelLatticeShellRule make_multilevel_lattice_shell_rule(
    const marisa_b_shell_v1::ShellBin& bin,
    const MultilevelLatticeShellRuleConfig& config = {});

}  // namespace marisa_b_eft_v2

#endif  // MARISA_B_EFT_V2_LATTICE_SHELL_RULE_H
