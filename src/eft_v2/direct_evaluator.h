#ifndef MARISA_B_EFT_V2_DIRECT_EVALUATOR_H
#define MARISA_B_EFT_V2_DIRECT_EVALUATOR_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "ir_safe_integrands.h"
#include "uv_subtraction.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

enum class DirectAngularRule {
    TensorGaussLegendre,
    AntipodalFibonacci,
};

const char* direct_angular_rule_name(DirectAngularRule rule) noexcept;

using DirectDiagramMask=std::uint32_t;
inline constexpr DirectDiagramMask kDirectB222=
    DirectDiagramMask{1}<<0;
inline constexpr DirectDiagramMask kDirectB321I=
    DirectDiagramMask{1}<<1;
inline constexpr DirectDiagramMask kDirectB321II=
    DirectDiagramMask{1}<<2;
inline constexpr DirectDiagramMask kDirectB411=
    DirectDiagramMask{1}<<3;
inline constexpr DirectDiagramMask kDirectAllDiagrams=
    kDirectB222|kDirectB321I|kDirectB321II|kDirectB411;

struct DirectIntegrationConfig {
    double qmin=1.0e-4;
    double qmax=1.0;
    int n_radial=12;
    int n_mu=12;
    int n_phi=12;
    bool ir_safe=true;
    bool uv_subtract=false;
    // Restore the finite q^-2 tadpole tail above qmax after the leading
    // renormalized zero-lag piece has been subtracted.  Unlike the
    // pre-reconstruction FFTLog tables, this numerical route works for a
    // reconstructed provider with external-vector-dependent CIC windows.
    bool restore_uv_tail=false;
    double uv_tail_kmax=80.0;
    int uv_tail_quadrature_order=128;
    int uv_tail_n_mu=24;
    int uv_tail_n_phi=32;
    // Real-space kernels are invariant under reflection through the external
    // triangle plane.  When enabled, n_phi Gauss--Legendre nodes cover
    // [0,pi] and their weights are doubled instead of evaluating the
    // reflected copies on [pi,2pi].
    bool exploit_phi_reflection=false;
    // The mapped IR-safe integrand has angular Voronoi boundaries.  The
    // antipodal Fibonacci rule is available as a deterministic,
    // equal-weight cross-check of the tensor Gauss--Legendre rule.
    DirectAngularRule angular_rule=DirectAngularRule::TensorGaussLegendre;
    // Production hybrids may replace the tadpole topologies with their
    // factorized/analytic forms while retaining the same direct convolution
    // evaluator for B222 and B321I.
    DirectDiagramMask diagram_mask=kDirectAllDiagrams;
    UvSubtractionConfig uv;
};

struct DiagramTemplates {
    std::array<Vec3,3> external;
    SparsePolynomial tree;
    SparsePolynomial B222;
    SparsePolynomial B321I;
    SparsePolynomial B321II;
    SparsePolynomial B411;
    SparsePolynomial B321II_bare;
    SparsePolynomial B411_bare;
    SparsePolynomial B321II_uv_subtraction;
    SparsePolynomial B411_uv_subtraction;
    SparsePolynomial uv_subtraction_total;
    SparsePolynomial B321II_uv_restoration;
    SparsePolynomial B411_uv_restoration;
    SparsePolynomial uv_restoration_total;
    SparsePolynomial one_loop;
    SparsePolynomial total;
    std::size_t integration_nodes=0;
    bool ir_safe=false;
};

DiagramTemplates evaluate_direct(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& config);

}  // namespace marisa_b_eft_v2

#endif
