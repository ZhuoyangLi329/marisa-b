#ifndef MARISA_B_EFT_V2_FFTLOG_DR_ORACLE_H
#define MARISA_B_EFT_V2_FFTLOG_DR_ORACLE_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "direct_evaluator.h"
#include "tracer_power.h"
#include "PowerSpectrum.h"

namespace marisa_b_eft_v2 {

struct FftlogConfig {
    double kmin=1.0e-5;
    double kmax=100.0;
    int frequency_count=256;
    int reconstruction_grid_size=8192;
    double bias_nu=std::numeric_limits<double>::quiet_NaN();
    // Analytic DR contractions require only the complex power-law modes.
    // Disable the positive log-interpolated reconstruction when scanning
    // topology-specific FFTLog biases that are not intended as a standalone
    // power-spectrum approximation.
    bool reconstruct_interpolated_power=true;
    // Philcox/OneLoopBispectrum samples N points including both k-range
    // endpoints, so the Fourier period is N/(N-1) log(kmax/kmin).  Keep the
    // historical endpoint-excluded convention as the default for the
    // standalone interpolated-power oracle.
    bool endpoint_inclusive_sampling=false;
};

struct FftlogDrMasterConfig {
    double relative_tolerance=1.0e-12;
    int maximum_outer_terms=256;
    int maximum_hypergeometric_terms=4096;
};

struct FftlogDrMasterResult {
    std::complex<double> value;
    int outer_terms=0;
    int maximum_hypergeometric_terms_used=0;
    double relative_tail=std::numeric_limits<double>::infinity();
};

struct FftlogMode {
    std::complex<double> coefficient;
    std::complex<double> exponent;
    double pivot_k=1.0;
};

struct FftlogLaurentConfig {
    int minimum_power=-2;
    int maximum_power=2;
    int oversampling_factor=3;
    int validation_samples=96;
    double singular_value_tolerance=1.0e-12;
    double coefficient_prune_relative=1.0e-10;
};

struct FftlogLaurentTerm {
    int chart=0;
    std::array<int,3> powers={{0,0,0}};
    SparsePolynomial coefficient;
};

struct FftlogLaurentReduction {
    std::vector<FftlogLaurentTerm> terms;
    int basis_terms=0;
    int training_samples=0;
    int validation_samples=0;
    int numerical_rank=0;
    double condition_number=std::numeric_limits<double>::infinity();
    int refit_rank=0;
    double refit_condition_number=std::numeric_limits<double>::infinity();
    double maximum_validation_absolute_error=std::numeric_limits<double>::infinity();
    double maximum_validation_relative_error=std::numeric_limits<double>::infinity();
};

struct FftlogAnalyticDiagramResult {
    SparsePolynomial value;
    FftlogLaurentReduction reduction;
    std::vector<FftlogLaurentReduction> route_reductions;
    int fftlog_modes_used=0;
    std::size_t master_integral_evaluations=0;
    double maximum_imaginary_to_real=0.0;
};

struct FftlogRegulatorConfig {
    // All regulated master integrals are evaluated at two strictly positive
    // epsilon values.  The production value is the first-order Richardson
    // extrapolation to epsilon=0; no singular master is called exactly on a
    // Gamma-function pole.
    double coarse_epsilon=1.0e-8;
    double refinement_ratio=10.0;
};

struct FftlogRegulatorDiagnostic {
    double coarse_epsilon=0.0;
    double fine_epsilon=0.0;
    double refinement_ratio=0.0;
    double maximum_absolute_coarse_to_fine=0.0;
    double maximum_relative_coarse_to_fine=0.0;
    double maximum_absolute_extrapolation_correction=0.0;
    double maximum_relative_extrapolation_correction=0.0;
};

struct FftlogUvRestorationConfig {
    bool enabled=true;
    double moment_kmin=1.0e-4;
    double moment_kmax=80.0;
    int moment_quadrature_order=256;
    int angular_n_mu=16;
    int angular_n_phi=16;
    UvSubtractionConfig asymptotic;
};

inline constexpr std::uint64_t kB222MatterContourMask=
    std::uint64_t{1}<<0;
inline constexpr std::uint64_t kB222ConstantContourMask=
    std::uint64_t{1}<<3;
inline constexpr std::uint64_t kB222AllContourMask=
    (std::uint64_t{1}<<10)-1;
inline constexpr std::uint64_t kB321IMatterContourMask=
    std::uint64_t{1}<<0;
inline constexpr std::uint64_t kB321ITidalContourMask=
    std::uint64_t{1}<<5;
inline constexpr std::uint64_t kB321ISoftContourMask=
    kB321IMatterContourMask|kB321ITidalContourMask;
inline constexpr std::uint64_t kB321ILocalQuadraticContourMask=
    std::uint64_t{1}<<1;
inline constexpr std::uint64_t kB321IAllContourMask=
    (std::uint64_t{1}<<18)-1;
inline constexpr std::uint64_t kB321ICompositeContourMask=
    kB321IAllContourMask
    &~(kB321ISoftContourMask|kB321ILocalQuadraticContourMask);

// A finite FFTLog expansion has to be evaluated on a contour lying in the
// fundamental strip of the complete operator sector.  The public reference
// uses one bias per topology; that contour gives the wrong sign for positive
// constant-kernel convolutions and is measurably unstable for the local
// quadratic-density sector.  The B321I classes follow the soft structure of
// the exact kernels: matter/tidal-soft, local b1^2*b2, and the remaining
// composite operators.  These masks split only exact bias monomials, so their
// sum is the original B222/B321I topology without a fitted correction.
struct FftlogContourSectorConfig {
    bool enabled=false;
    bool endpoint_inclusive_sampling=true;
    double B222_default_bias=-0.6;
    double B222_constant_bias=-1.9;
    double B321I_soft_bias=-0.3;
    // A convergence-certified interior point of the (-1,-1/2)
    // fundamental strip.  Avoid the Gamma poles at both boundaries.
    double B321I_local_quadratic_bias=-0.66;
    double B321I_composite_bias=-1.9;
    double B411_bias=-0.3;
};

struct FftlogAnalyticOneLoopConfig {
    FftlogLaurentConfig B222_laurent;
    FftlogLaurentConfig B321I_laurent;
    FftlogLaurentConfig B321II_laurent;
    FftlogDrMasterConfig master;
    FftlogRegulatorConfig regulator;
    FftlogUvRestorationConfig uv_restoration;
    FftlogContourSectorConfig contour_sectors;
    TracerPowerIntegrationConfig B321II_factorized_p13;
    double mode_coefficient_relative_tolerance=1.0e-12;

    FftlogAnalyticOneLoopConfig() {
        // The cubic kernel in B321I has one extra Laurent order compared
        // with B222 and the tadpole routes.
        B321I_laurent.minimum_power=-3;
        B321I_laurent.maximum_power=3;
        B321II_factorized_p13.qmax=30.0;
        B321II_factorized_p13.n_radial=32;
        B321II_factorized_p13.n_mu=64;
    }
};

struct FftlogAnalyticRegulatedDiagramResult {
    FftlogAnalyticDiagramResult extrapolated;
    SparsePolynomial coarse_value;
    SparsePolynomial fine_value;
    FftlogRegulatorDiagnostic regulator;
};

struct FftlogAnalyticConvolutionResult {
    FftlogAnalyticDiagramResult B222;
    FftlogAnalyticRegulatedDiagramResult B321I;
    FftlogAnalyticRegulatedDiagramResult B411;
    std::size_t master_integral_evaluations=0;
};

struct FftlogAnalyticOneLoopResult {
    DiagramTemplates templates;
    FftlogAnalyticDiagramResult B222;
    FftlogAnalyticRegulatedDiagramResult B321I;
    // The flattened three-propagator continuation is retained as a
    // diagnostic.  The physical real-space B321II template uses the
    // factorized renormalized P13 construction below.
    FftlogAnalyticRegulatedDiagramResult B321II;
    FactorizedB321IITemplates B321II_factorized;
    FftlogAnalyticRegulatedDiagramResult B411;
    UvSubtractionComponents uv_subleading_coefficients;
    double linear_power_uv_moment=0.0;
    std::size_t master_integral_evaluations=0;
};

// Analytically continued three-propagator integral of Simonovic et al.
// (arXiv:1708.08130, eqs. 3.6 and 3.32), including d^3q/(2*pi)^3.
// The series representation is selected for the canonical physical domain
// 0 < x <= y <= 1.
FftlogDrMasterResult fftlog_dr_master_integral(
    std::complex<double> nu1,
    std::complex<double> nu2,
    std::complex<double> nu3,
    double x,
    double y,
    FftlogDrMasterConfig config={});

// Analytically continued two-propagator massless integral I(nu1,nu2)
// of Simonovic et al. eq. (2.6), including d^3q/(2*pi)^3.
std::complex<double> fftlog_dr_two_propagator_integral(
    std::complex<double> nu1,
    std::complex<double> nu2);

// Recover the fixed-triangle B222 kernel as a finite Laurent expansion in
// q^2, |k_short-q|^2 and |k_middle+q|^2, all normalized by k_long^2.
// Independent holdout samples make this a checked reduction rather than an
// assumed exponent table.
FftlogLaurentReduction fftlog_dr_reduce_b222_kernel(
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    FftlogLaurentConfig config={});

// Route-resolved B321I reduction.  The two indices follow DiagramAssembler:
// one identifies the external P(k_i), the other P(|k_j-q|), with i != j.
FftlogLaurentReduction fftlog_dr_reduce_b321i_route_kernel(
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    int external_power_index,
    int shifted_power_index,
    FftlogLaurentConfig config={});

FftlogLaurentReduction fftlog_dr_reduce_b321ii_route_kernel(
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    int left_external_index,
    int right_external_index,
    int tadpole_external_index,
    FftlogLaurentConfig config={});

// Finite terms that the public OneLoopBispectrum FFTLog construction adds
// after analytically continuing its scale-free master integrals.  The moment
// is (2*pi^2)^-1 integral_0^infinity P11(q)dq.  These functions intentionally
// expose the two topology corrections separately for scheme-conversion tests.
SparsePolynomial fftlog_b321ii_exact_uv_restoration(
    const PowerSpectrum& source,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    double normalized_linear_power_moment);

SparsePolynomial fftlog_b411_exact_uv_restoration(
    const PowerSpectrum& source,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    double normalized_linear_power_moment);

double fftlog_normalized_linear_power_uv_moment(
    const PowerSpectrum& source,
    FftlogUvRestorationConfig config={});

class FftlogPowerSpectrum final:public PowerSpectrum {
public:
    FftlogPowerSpectrum(const PowerSpectrum& source,FftlogConfig config);
    real Evaluate(real k) const override;
    const Cosmology& GetCosmology() const override;

    const FftlogConfig& config() const noexcept { return config_; }
    double actual_bias_nu() const noexcept { return actual_bias_nu_; }
    const std::vector<std::complex<double>>& coefficients() const noexcept { return coefficients_; }
    std::vector<FftlogMode> modes() const;
    std::string metadata_json() const;

private:
    const PowerSpectrum& source_;
    FftlogConfig config_;
    double actual_bias_nu_=0.0;
    double log_kmin_=0.0;
    double log_span_=0.0;
    double log_period_=0.0;
    std::vector<std::complex<double>> coefficients_;
    std::vector<double> dense_log_power_;
};

class FftlogDrOracle {
public:
    FftlogDrOracle(const PowerSpectrum& source,FftlogConfig fftlog);
    DiagramTemplates evaluate(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        DirectIntegrationConfig integration) const;
    FftlogAnalyticDiagramResult evaluate_b222_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogLaurentConfig laurent={},
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12) const;
    FftlogAnalyticDiagramResult evaluate_b222_public_table_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12,
        std::uint64_t operator_mask=~std::uint64_t{0}) const;
    FftlogAnalyticDiagramResult evaluate_b321i_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogLaurentConfig laurent={},
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12,
        double analytic_regulator_epsilon=1.0e-8) const;
    FftlogAnalyticDiagramResult evaluate_b321i_public_table_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12,
        double analytic_regulator_epsilon=1.0e-8,
        std::uint64_t operator_mask=~std::uint64_t{0}) const;
    FftlogAnalyticDiagramResult evaluate_b321ii_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogLaurentConfig laurent={},
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12,
        double analytic_regulator_epsilon=1.0e-8) const;
    FftlogAnalyticDiagramResult evaluate_b411_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogLaurentConfig laurent={},
        FftlogDrMasterConfig master={},
        double mode_coefficient_relative_tolerance=1.0e-12,
        double analytic_regulator_epsilon=1.0e-8) const;
    FftlogAnalyticConvolutionResult evaluate_convolutions_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogAnalyticOneLoopConfig config={}) const;
    FftlogAnalyticOneLoopResult evaluate_analytic(
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider,
        FftlogAnalyticOneLoopConfig config={}) const;
    const FftlogPowerSpectrum& power() const noexcept { return power_; }
    std::string metadata_json(const DirectIntegrationConfig& integration) const;
    std::string analytic_metadata_json(
        FftlogAnalyticOneLoopConfig config={}) const;

private:
    const PowerSpectrum& source_;
    FftlogPowerSpectrum power_;
};

}  // namespace marisa_b_eft_v2

#endif
