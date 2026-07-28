#ifndef MARISA_B_EFT_V2_TRACER_POWER_H
#define MARISA_B_EFT_V2_TRACER_POWER_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "field_kernel_provider.h"
#include "shell_average.h"
#include "uv_subtraction.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

struct TracerPowerIntegrationConfig {
    double qmin=1.0e-4;
    double qmax=10.0;
    int n_radial=8;
    int n_mu=16;
    UvSubtractionConfig uv;
    bool restore_p13_uv_tail=true;
    double uv_tail_kmax=80.0;
    int uv_tail_quadrature_order=128;
};

struct TracerPowerTemplates {
    double k=0.0;
    std::size_t shell_nodes=0;
    std::uint64_t shell_mode_count=0;
    std::size_t shell_unique_radius_count=0;
    std::int64_t shell_boundary_mode_adjustment=0;
    SparsePolynomial tree;
    SparsePolynomial P22;
    SparsePolynomial P13;
    SparsePolynomial P22_bare;
    SparsePolynomial P13_bare;
    SparsePolynomial P22_stochastic_subtraction;
    SparsePolynomial P13_bias_subtraction;
    SparsePolynomial P13_uv_tail_restoration;
    SparsePolynomial one_loop;
    SparsePolynomial spt;
    SparsePolynomial counterterm_bnabla2_delta;
    double stochastic_pshot=1.0;
    double stochastic_a0=0.0;
    std::size_t integration_nodes=0;
};

struct FactorizedB321IITemplates {
    SparsePolynomial value;
    SparsePolynomial bare;
    SparsePolynomial bias_subtraction;
    SparsePolynomial uv_tail_restoration;
    std::array<SparsePolynomial,3> P13_legs;
    std::array<SparsePolynomial,3> P13_bare_legs;
    std::array<SparsePolynomial,3> P13_bias_subtraction_legs;
    std::array<SparsePolynomial,3> P13_uv_tail_restoration_legs;
    std::size_t integration_nodes=0;
};

struct RenormalizedTracerP13Template {
    double k=0.0;
    SparsePolynomial value;
    SparsePolynomial bare;
    SparsePolynomial bias_subtraction;
    SparsePolynomial uv_tail_restoration;
    std::size_t integration_nodes=0;
};

TracerPowerTemplates tracer_power_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    double k,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration={},
    double k_nl=0.30);

TracerPowerTemplates tracer_power_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    double lower,
    double upper,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell,
    double k_nl=0.30);

RenormalizedTracerP13Template renormalized_tracer_p13_template(
    const PowerSpectrum& linear_power,
    double k,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration={});

FactorizedB321IITemplates assemble_factorized_b321ii_templates(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const std::array<RenormalizedTracerP13Template,3>& p13_legs);

// In real space K1=b1 is momentum independent, so the six B321II
// permutations factorize into three renormalized tracer P13 legs.  This is
// the same special P13 construction used by the public pre-reconstruction
// one-loop implementation and avoids a flattened three-propagator master.
FactorizedB321IITemplates factorized_b321ii_templates(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration={});

}  // namespace marisa_b_eft_v2

#endif
