#ifndef MARISA_B_EFT_V2_DR_STOCHASTIC_ROLES_H
#define MARISA_B_EFT_V2_DR_STOCHASTIC_ROLES_H

#include <array>

#include "field_kernel_provider.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

constexpr std::size_t kSignedDrRoleCount=5;

enum class SignedDrRole : std::size_t {
    DDD=0,
    DDR=1,
    DRR=2,
    RRR=3,
    TiedF=4
};

struct SignedDrRoleTemplates {
    std::array<SparsePolynomial,kSignedDrRoleCount> value;

    const SparsePolynomial& operator[](SignedDrRole role) const {
        return value.at(static_cast<std::size_t>(role));
    }
    SparsePolynomial& operator[](SignedDrRole role) {
        return value.at(static_cast<std::size_t>(role));
    }
};

enum class RandomCatalogConvention {
    Continuum,
    FiniteIndependentPoisson
};

struct RandomCatalogConfig {
    RandomCatalogConvention convention=RandomCatalogConvention::Continuum;
    double random_to_halo_density=30.0;
};

/*
 * Fixed Poisson is parameter-free.  Index p multiplies n_species^{-p}; zero
 * is unused.  Raw signed D/R roles use the same zero-shot algebra as the
 * measurement role vectors.  ``legacy_theory_filter_complement`` is only the
 * complement implied by the historical topology filter in
 * reconstructed_fixed_poisson_bispectrum(); it has not been shown to equal
 * the jaxpower ``full - ordered_sum_no_shot`` measurement channel.  Thus
 * raw role sum + legacy_theory_filter_complement equals
 * ``legacy_theory_retained`` for each inverse-density power, but all three
 * fixed-Poisson objects remain diagnostic and frozen-model inactive.
 */
struct DrFixedPoissonSpeciesTemplates {
    std::array<SignedDrRoleTemplates,3> raw_by_inverse_density;
    std::array<SparsePolynomial,3> legacy_theory_filter_complement;
    std::array<SparsePolynomial,3> legacy_theory_retained;
};

struct DrLeadingStochasticPoissonTemplates {
    // The only two stochastic nuisance directions in this API.
    SignedDrRoleTemplates Ashot_residual;
    SignedDrRoleTemplates Bshot_residual;

    // Independent Poisson species; there is no halo-random cross cumulant.
    DrFixedPoissonSpeciesTemplates halo_poisson;
    DrFixedPoissonSpeciesTemplates random_poisson;
    RandomCatalogConfig random_catalog;
};

struct DrLeadingStochasticPngResponses {
    // Ashot is a pure constant residual and has no leading PNG response.
    SignedDrRoleTemplates Ashot_residual;
    // Frozen ``legacy_accepted_pre`` single-alpha3 branch: one half of the
    // strict generating-shape product derivative.
    SignedDrRoleTemplates Bshot_residual;
    // Parameter-free diagnostic fixed-Poisson response.  It is not part of
    // the frozen Gaussian or linear-PNG production model.
    DrFixedPoissonSpeciesTemplates halo_poisson;
    DrFixedPoissonSpeciesTemplates random_poisson;
};

/*
 * Fully tree-level, ordered external-field contraction.  Ashot and Bshot are
 * global coordinates shared by all four signed roles.  Finite randoms add a
 * fixed Poisson species only; they never introduce a random stochastic
 * nuisance or a response gain.
 */
DrLeadingStochasticPoissonTemplates
dr_leading_stochastic_poisson_tree(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const RandomCatalogConfig& random_catalog={});

DrLeadingStochasticPngResponses
dr_leading_stochastic_png_response(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const FieldKernelProvider& local_png_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const RandomCatalogConfig& random_catalog={});

/*
 * Independent PNG stochastic operator [epsilon_Psi Psi] for Delta=2,
 * normalized to a unit <epsilon_0 epsilon_Psi> amplitude and
 * Psi(k)=(k/k_pivot)^2 Phi(k).  This is not the deterministic bPsi response
 * of Bshot_residual; both directions are required and are kept separate.
 */
SignedDrRoleTemplates dr_delta2_epsilon_psi_stochastic_tree(
    const PowerSpectrum& linear_power,
    const PowerSpectrum& transfer_m,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_pivot=0.05);

SparsePolynomial signed_dr_role_sum(
    const SignedDrRoleTemplates& roles);

std::array<SparsePolynomial,3>
fixed_poisson_in_halo_density_units(
    const DrFixedPoissonSpeciesTemplates& halo,
    const DrFixedPoissonSpeciesTemplates& random,
    double random_to_halo_density);

}  // namespace marisa_b_eft_v2

#endif
