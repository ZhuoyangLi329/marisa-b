#ifndef MARISA_B_EFT_V2_STOCHASTIC_H
#define MARISA_B_EFT_V2_STOCHASTIC_H

#include <array>
#include <cstddef>

#include "counterterms.h"
#include "field_kernel_provider.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

constexpr std::size_t kStochasticCount=12;

struct MixedStochasticPowerTemplates {
    std::array<SparsePolynomial,4> value;
    std::array<SparsePolynomial,4> P22;
    std::array<SparsePolynomial,4> P13;
    std::array<SparsePolynomial,4> P22_bare;
    std::array<SparsePolynomial,4> P13_bare;
    std::array<SparsePolynomial,4> P22_zero_lag_subtraction;
    std::array<SparsePolynomial,4> P13_bias_subtraction;
    std::array<SparsePolynomial,4> P13_uv_tail_restoration;
    std::size_t integration_nodes=0;

    const SparsePolynomial& operator[](std::size_t index) const {
        return value.at(index);
    }
    SparsePolynomial& operator[](std::size_t index) {
        return value.at(index);
    }
};

struct StochasticIntegrationConfig {
    double qmin=1.0e-4;
    double qmax=1.0;
    int n_radial=12;
    int n_mu=16;
    // The pre-reconstruction implementation integrates azimuth analytically.
    // Reconstructed Cartesian CIC kernels require an explicit full 0..2pi
    // azimuthal rule.
    int n_phi=16;
    double mu_ren=0.30;
    double asymptotic_factor=64.0;
    bool restore_p13_uv_tail=true;
    double uv_tail_kmax=80.0;
    int uv_tail_quadrature_order=128;
};

struct StochasticTemplates {
    std::array<SparsePolynomial,kStochasticCount> pure;
    std::array<SparsePolynomial,kStochasticCount> mixed_tree;
    std::array<SparsePolynomial,kStochasticCount> mixed_one_loop;
    std::array<SparsePolynomial,kStochasticCount> derivative_mixed;
    std::array<SparsePolynomial,kStochasticCount> raw;
    SparsePolynomial bshot_bnabla2_cross;

    std::array<double,kStochasticCount> linear_shapes(
        const std::array<double,kParameterCount>& parameter_values,
        double number_density) const;
    double evaluate(
        const std::array<double,kParameterCount>& parameter_values,
        double number_density) const;
};

enum class ReconstructedStochasticMap {
    DensityOnly,
    TiedDensityAndShift
};

struct ReconstructedStochasticTemplates {
    /*
     * ``density_only`` propagates tracer stochasticity through the observed
     * density while keeping the reconstruction displacement noise-free.
     * ``noisy_shift`` is the incremental response of the stochastic
     * displacement source, and ``tied`` is their exact sum for the same halo
     * noise realization.  Keeping all three makes the zero/tied/released
     * inference branches nested and auditable.
     */
    StochasticTemplates density_only;
    StochasticTemplates noisy_shift;
    StochasticTemplates tied;
};

const std::array<ParameterId,kStochasticCount>& stochastic_parameter_ids();

MixedStochasticPowerTemplates mixed_stochastic_cross_power(
    const PowerSpectrum& linear_power,
    double k,
    const StochasticIntegrationConfig& integration={});

MixedStochasticPowerTemplates
reconstructed_mixed_stochastic_cross_power(
    const PowerSpectrum& linear_power,
    const Vec3& external,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& integration={},
    ReconstructedStochasticMap map=
        ReconstructedStochasticMap::TiedDensityAndShift);

StochasticTemplates stochastic_bispectrum(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    double k_nl=0.30,
    const StochasticIntegrationConfig& integration={},
    const std::array<MixedStochasticPowerTemplates,3>* precomputed_cross_power=nullptr,
    const PowerSpectrum* leading_tree_power=nullptr);

StochasticTemplates reconstructed_stochastic_bispectrum(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_nl=0.30,
    const StochasticIntegrationConfig& integration={},
    const PowerSpectrum* leading_tree_power=nullptr);

ReconstructedStochasticTemplates
reconstructed_stochastic_bispectrum_decomposition(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_nl=0.30,
    const StochasticIntegrationConfig& integration={},
    const PowerSpectrum* leading_tree_power=nullptr);

/*
 * Generating shape for the leading estimator-subtracted Bshot residual after
 * standard reconstruction.  ``stochastic_shift=false`` propagates the
 * residual only through the observed density, while ``true`` also inserts it
 * into the reconstruction displacement.
 *
 * This primitive varies the deterministic K1 provider in both factors of the
 * mixed P/nbar contraction.  Consequently, for
 *
 *   K1(f)=b1+f q,
 *
 * it generates H(f)=(b1+f q)^2 sum(P).  This is the strict tied
 * alpha3=alpha3PNG limit of Moradinezhad Dizgah et al. Eq. (2.65).  The
 * production model keeps alpha3 and alpha3PNG independent and therefore
 * needs the three shared shapes
 *
 *   G=b1^2 sum(P), L=b1 q sum(P), Q=q^2 sum(P),
 *
 * extracted as G=H(0), L=[H(+1)-H(-1)]/4 and
 * Q=[H(+1)+H(-1)]/2-H(0).  Eq. (2.65) then uses L once in the alpha3 branch
 * and once in the alpha3PNG branch.  Keeping ``generating`` in the API name
 * prevents a future caller from silently identifying either individual L
 * branch with the full strict-tied derivative [H(+1)-H(-1)]/2.
 */
SparsePolynomial reconstructed_leading_bshot_residual_generating_shape(
    const PowerSpectrum& tree_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    bool stochastic_shift);

double analytic_poisson_bispectrum(
    const std::array<double,3>& tracer_tree_power,
    double number_density);

}  // namespace marisa_b_eft_v2

#endif
