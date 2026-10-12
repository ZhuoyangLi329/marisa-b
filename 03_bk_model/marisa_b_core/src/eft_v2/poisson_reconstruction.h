#ifndef MARISA_B_EFT_V2_POISSON_RECONSTRUCTION_H
#define MARISA_B_EFT_V2_POISSON_RECONSTRUCTION_H

#include <array>
#include <cstddef>

#include "field_kernel_provider.h"
#include "stochastic.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

/*
 * Fixed local-Poisson contribution that remains after the measured
 * bispectrum estimator has removed all contractions in which two or three
 * external output-particle density factors refer to the same catalog
 * particle.  The surviving terms contain at least one internal
 * reconstruction-shift noise leg.
 *
 * Index s of ``by_inverse_number_density`` multiplies nbar^{-s}; index zero is
 * unused.  ``tree`` contains total stochastic/matter power order two and
 * ``one_loop`` order three.  This separation is retained in production
 * metadata so that no released stochastic nuisance can silently masquerade
 * as the fixed Poisson baseline.
 */
struct FixedPoissonOrderTemplates {
    std::array<SparsePolynomial,4>
        by_inverse_number_density;
    std::size_t generated_topologies=0;
    std::size_t estimator_allowed_topologies=0;
    std::size_t integration_nodes=0;
};

struct ReconstructedFixedPoissonTemplates {
    FixedPoissonOrderTemplates tree;
    FixedPoissonOrderTemplates one_loop;

    std::array<SparsePolynomial,4> total_by_inverse_number_density() const;
    double evaluate(
        const std::array<double,kParameterCount>& parameter_values,
        double number_density) const;
};

enum class PoissonIntensityMode {
    /*
     * Only the constant K0 part of each local Poisson cumulant.  This mode is
     * retained as a sharp combinatoric oracle.
     */
    UnitK0,
    /*
     * Each cumulant is proportional to 1+delta_h at its coincident point.
     * The delta_h factor is expanded with the same deterministic tracer
     * kernels as the observed catalog.
     */
    ConditionalTracer
};

ReconstructedFixedPoissonTemplates
reconstructed_fixed_poisson_bispectrum(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& integration={},
    ReconstructedStochasticMap map=
        ReconstructedStochasticMap::TiedDensityAndShift,
    PoissonIntensityMode intensity_mode=
        PoissonIntensityMode::ConditionalTracer,
    bool include_bare_one_loop=false);

}  // namespace marisa_b_eft_v2

#endif
