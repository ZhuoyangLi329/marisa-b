#ifndef MARISA_B_EFT_V2_FIELD_KERNEL_PROVIDER_H
#define MARISA_B_EFT_V2_FIELD_KERNEL_PROVIDER_H

#include <string>
#include <string_view>
#include <vector>

#include "bias_operators.h"

namespace marisa_b_eft_v2 {

class FieldKernelProvider {
public:
    virtual ~FieldKernelProvider() = default;
    virtual SparsePolynomial deterministic(const std::vector<Vec3>& momenta) const = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual bool supports_triangle_plane_reflection() const noexcept {
        return true;
    }
};

/*
 * A marked field contains one distinguished stochastic leg and zero or more
 * ordinary matter legs.  The stochastic momentum is kept separate because
 * reconstruction may place the marked block either in the observed-density
 * factor or in one of the displacement-source factors.  Treating this leg as
 * an ordinary interchangeable perturbative leg gives incorrect factorials.
 */
class MarkedFieldKernelProvider {
public:
    virtual ~MarkedFieldKernelProvider() = default;
    virtual SparsePolynomial marked(
        const std::vector<Vec3>& matter_momenta,
        const Vec3& stochastic_momentum) const = 0;
    virtual std::string_view name() const noexcept = 0;
};

class EftBiasKernelProvider final : public FieldKernelProvider {
public:
    SparsePolynomial deterministic(const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return "eft_v2_cobra_bias"; }
};

class MatterKernelProvider final : public FieldKernelProvider {
public:
    SparsePolynomial deterministic(const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return "matter_spt"; }
};

class V1CompatibilityKernelProvider final : public FieldKernelProvider {
public:
    SparsePolynomial deterministic(const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return "halo_v1_compatibility"; }
};

struct FieldKernelVariation {
    SparsePolynomial value;
    SparsePolynomial direction;
};

/*
 * Push one linear field-kernel variation through reconstruction.  The
 * direction is inserted once, in either the density block or one of the
 * displacement-source blocks.  This is the field-level product rule needed
 * by reconstructed EFT counterterms and by later response operators.
 */
FieldKernelVariation reconstructed_field_kernel_variation(
    const FieldKernelProvider& base,
    const FieldKernelProvider& direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta);

/*
 * Generalized product rule for renormalized sectors whose density insertion
 * is represented in a reduced operator basis, while the reconstruction shift
 * is still sourced by the full observed tracer.  This distinction is needed
 * by the mixed-stochastic sector: its pre-reconstruction renormalized density
 * kernel has four observable directions, but noise also enters the full
 * displacement source.
 */
FieldKernelVariation reconstructed_field_kernel_variation_with_shift(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_direction,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta);

/*
 * Push a single distinguished stochastic leg through standard
 * reconstruction.  Only the n ordinary matter legs are partitioned
 * symmetrically.  If the marked block contains m matter legs, and the
 * unmarked blocks have sizes m_B, its weight is
 *
 *     m! prod_B(m_B!) / n!.
 *
 * Equivalently, when implemented with labelled (n+1)-leg set partitions this
 * is the usual product(|B|!)/(n+1)! multiplied by
 * (n+1)/|B_marked|.  The separate density/shift marked providers permit
 * noise-free-shift, tied-noise, and released-noise closures without changing
 * the deterministic reconstruction map.
 */
SparsePolynomial reconstructed_marked_field_kernel(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const Vec3& stochastic_momentum);

enum class StochasticBlockRole {
    Density,
    Shift
};

struct RoleResolvedMarkedFieldTerm {
    SparsePolynomial value;
    /*
     * One entry per input stochastic momentum, in the same order.  Keeping
     * this provenance is essential for matching an estimator that removes
     * contractions between two external output-particle density factors but
     * does not remove contractions internal to the reconstruction shift.
     */
    std::vector<StochasticBlockRole> stochastic_roles;
};

std::vector<RoleResolvedMarkedFieldTerm>
reconstructed_multimarked_field_terms(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta);

/*
 * Generalization to several stochastic legs.  A pre-reconstruction density
 * or displacement-source block may contain at most one stochastic leg, as
 * appropriate for X=delta_h+epsilon.  Reconstruction itself may nevertheless
 * generate several such blocks.  Matter and stochastic labels are
 * symmetrized separately; this is the field-level object required by
 * P/nbar^2 and higher Poisson-cumulant terms.
 */
SparsePolynomial reconstructed_multimarked_field_kernel(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta);

/*
 * Apply the standard-reconstruction set-partition map to any deterministic
 * field-kernel provider.  Keeping the wrapped provider explicit is essential:
 * the production EFT-v2 provider carries the complete 11-operator sparse
 * polynomial, while the matter and halo-v1 providers are independent
 * regression oracles.
 *
 * The reconstruction displacement is sourced by the same wrapped tracer
 * field as the density block.  Thus every non-density partition block
 * contributes R_k(k_B) K_|B|(B), exactly as in the established halo-v1
 * derivation.  The Cartesian CIC window remains inside ReconstructionConfig
 * and is evaluated on the actual block vector.
 */
class ReconstructedFieldKernelProvider final : public FieldKernelProvider {
public:
    ReconstructedFieldKernelProvider(
        const FieldKernelProvider& base,
        marisa_b_halo_v1::ReconstructionConfig reconstruction);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }
    bool supports_triangle_plane_reflection() const noexcept override {
        // A Cartesian CIC window is tied to the mesh axes.  Reflection in an
        // arbitrary external-triangle plane is therefore not a symmetry.
        return false;
    }

    const FieldKernelProvider& base() const noexcept { return base_; }
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction() const noexcept {
        return reconstruction_;
    }

private:
    const FieldKernelProvider& base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    std::string name_;
};

}  // namespace marisa_b_eft_v2

#endif
