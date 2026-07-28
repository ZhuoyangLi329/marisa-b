#ifndef MARISA_B_EFT_V2_SHELL_PROJECTOR_H
#define MARISA_B_EFT_V2_SHELL_PROJECTOR_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "counterterms.h"
#include "direct_evaluator.h"
#include "fftlog_dr_oracle.h"
#include "lattice_shell_rule.h"
#include "poisson_reconstruction.h"
#include "shell_average.h"
#include "stochastic.h"
#include "tracer_power.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

struct EftShellTemplates {
    marisa_b_shell_v1::ShellBin bin;
    DiagramTemplates diagrams;
    CountertermTemplates counterterms;
    StochasticTemplates stochastic;
    StochasticTemplates stochastic_density_only;
    StochasticTemplates stochastic_noisy_shift;
    ReconstructedFixedPoissonTemplates fixed_poisson;
    std::size_t shell_nodes=0;
    std::uint64_t total_loop_nodes=0;
    std::uint64_t cached_p13_nodes=0;
    bool fft_lattice_radial_measure=false;
    bool hybrid_factorized_analytic_tadpoles=false;
    bool fftlog_analytic_convolutions=false;
    bool exact_joint_lattice_measure=false;
    bool exact_k1_k2_mu_lattice_measure=false;
    bool conditional_cubic_orientation_exact=false;
    bool haar_oriented_cic_projection=false;
    bool reconstructed_counterterms=false;
    bool reconstructed_stochastic=false;
    bool reconstructed_stochastic_decomposed=false;
    bool reconstructed_fixed_poisson=false;
    bool multilevel_external_projection=false;
    std::size_t cheap_shell_nodes=0;
    std::size_t expensive_shell_nodes=0;
    std::size_t invariant_shell_nodes=0;
    std::size_t orientation_nodes=0;
    int exact_lattice_radial_order=0;
    int exact_lattice_angular_order=0;
    int expensive_lattice_radial_order=0;
    int expensive_k3_interpolation_order=0;
    std::uint64_t zero_external_leg_pairs=0;
    std::uint64_t closing_zero_pairs=0;
    double exact_lattice_valid_pair_fraction=0.0;
    double exact_lattice_total_variation=0.0;
    double expensive_total_variation=0.0;
    double expensive_maximum_sampled_lebesgue=0.0;
};

struct EftShellValues {
    marisa_b_shell_v1::ShellBin bin;
    double tree=0.0;
    double B222=0.0;
    double B321I=0.0;
    double B321II=0.0;
    double B411=0.0;
    double renormalized_loop=0.0;
    double counterterm=0.0;
    double stochastic=0.0;
    double total=0.0;
    std::array<double,kCountertermCount> counterterm_shapes={};
    std::array<double,kStochasticCount> stochastic_shapes={};
};

EftShellTemplates compute_eft_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& loop_config,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    const StochasticIntegrationConfig& stochastic_config={},
    double k_nl=0.30);

struct ReconstructedShellIntegrationConfig {
    DirectIntegrationConfig loop;
    HaarOrientedLatticeShellRuleConfig shell;
    /*
     * Propagate the four renormalized mixed-stochastic directions separately
     * through the density and noisy-displacement blocks.  The returned
     * templates expose density-only, incremental noisy-shift, and their tied
     * sum, permitting nested zero/tied/released robustness branches.
     */
    bool include_reconstructed_stochastic=false;
    /*
     * Add the estimator-matched local-Poisson K0 mean baseline through the
     * same P_L^3 counting as the deterministic one-loop calculation.  This
     * is separate from the released stochastic EFT directions.
     */
    bool include_reconstructed_fixed_poisson=false;
    /*
     * The bare P_L^3 fixed-Poisson contractions require a separate
     * renormalization and convergence gate.  Production keeps this false
     * until that gate passes; the exact leading P_L/nbar term remains on.
     */
    bool include_bare_fixed_poisson_one_loop=false;
};

EftShellTemplates compute_reconstructed_eft_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const ReconstructedShellIntegrationConfig& integration,
    const StochasticIntegrationConfig& stochastic_config={},
    double k_nl=0.30);

struct HybridShellIntegrationConfig {
    DirectIntegrationConfig convolution;
    TracerPowerIntegrationConfig factorized_p13;
    FftlogAnalyticOneLoopConfig analytic;
    bool analytic_convolutions=false;
    bool exact_lattice_multilevel=false;
    int expensive_lattice_radial_order=3;
    int expensive_k3_interpolation_order=20;
};

EftShellTemplates compute_eft_shell_templates_hybrid(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& provider,
    const FftlogDrOracle& b411_oracle,
    const HybridShellIntegrationConfig& loop_config,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    const StochasticIntegrationConfig& stochastic_config={},
    double k_nl=0.30);

EftShellValues evaluate_eft_shell_templates(
    const EftShellTemplates& templates,
    const std::array<double,kParameterCount>& parameters,
    double number_density);

}  // namespace marisa_b_eft_v2

#endif
