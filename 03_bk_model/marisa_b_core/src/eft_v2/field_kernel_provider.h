#ifndef MARISA_B_EFT_V2_FIELD_KERNEL_PROVIDER_H
#define MARISA_B_EFT_V2_FIELD_KERNEL_PROVIDER_H

#include <string>
#include <string_view>
#include <vector>

#include "bias_operators.h"

class PowerSpectrum;

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

enum class LocalPngTreeDirectionSector {
    ExternalBphi,
    LocalIc,
    BphiAdvection,
    Bphidelta,
    Total
};

/*
 * Pre-reconstruction halo local-PNG K1/K2 direction in the EFT-v2
 * (b1,b2,gamma2) coordinate system.  The two universality offsets are kept
 * separate because current p=1.12 analyses rescale bphi while deliberately
 * retaining the p=1 bphidelta relation.
 */
class LocalPngHaloTreeDirectionProvider final : public FieldKernelProvider {
public:
    LocalPngHaloTreeDirectionProvider(
        const PowerSpectrum& transfer,
        double p_bphi,
        double p_bphidelta,
        LocalPngTreeDirectionSector sector=
            LocalPngTreeDirectionSector::Total);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }

private:
    const PowerSpectrum& transfer_;
    double p_bphi_;
    double p_bphidelta_;
    LocalPngTreeDirectionSector sector_;
    std::string name_;
};

enum class Delta2PngTreeDirectionSector {
    ExternalBPsi,
    BPsiAdvection,
    BPsiDelta,
    BPsiS,
    Total
};

/*
 * Delta=2 halo-PNG K1/K2 direction used by EQ/OR-LSS tree and standard
 * reconstruction product rules.  The provider carries independently
 * calibrated numeric response coefficients and the explicit convention
 *
 *   Psi(k)=(k/k_pivot)^2 Phi(k).
 *
 * It contains no primordial B111 term; initial non-Gaussianity is contracted
 * separately.  Keeping that separation prevents double counting when the
 * same direction is used by deterministic and stochastic D/R role algebra.
 */
class Delta2PngHaloTreeDirectionProvider final : public FieldKernelProvider {
public:
    Delta2PngHaloTreeDirectionProvider(
        const PowerSpectrum& transfer,
        double bPsi,
        double bPsiDelta,
        double bPsiS,
        double k_pivot=0.05,
        Delta2PngTreeDirectionSector sector=
            Delta2PngTreeDirectionSector::Total);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }

private:
    const PowerSpectrum& transfer_;
    double bPsi_=0.0;
    double bPsiDelta_=0.0;
    double bPsiS_=0.0;
    double k_pivot_=0.05;
    Delta2PngTreeDirectionSector sector_;
    std::string name_;
};

struct FieldKernelVariation {
    SparsePolynomial value;
    SparsePolynomial direction;
};

/*
 * Keep the two physical response legs separate.  ``density_direction`` is an
 * insertion on the observed/output density block, while ``shift_direction``
 * is an insertion on one displacement-source block.  Their sum must equal
 * the legacy tied product-rule direction exactly; storing both is what lets
 * a caller propagate independently measured output and source PNG transfer
 * coefficients without introducing a bispectrum response gain.
 */
struct RoleResolvedFieldKernelVariation {
    SparsePolynomial value;
    SparsePolynomial density_direction;
    SparsePolynomial shift_direction;
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
 * Role-resolved form of reconstructed_field_kernel_variation_with_shift().
 * The base providers and the two response providers are all explicit.  This
 * function is backward-compatible by construction:
 *
 *   tied.direction = density_direction + shift_direction.
 */
RoleResolvedFieldKernelVariation
reconstructed_field_kernel_role_variation(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_response,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_response,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta);

/*
 * Keep the two estimator fields behind F=D-R explicit.  ``shifted_random``
 * contains only displacement blocks, while ``displaced_data`` adds those
 * pure-shift partitions back to the existing reconstructed overdensity F.
 * Both functions propagate one arbitrary field-kernel direction analytically.
 * They are additive diagnostics: the historical reconstructed-field provider
 * and every tied production call remain unchanged.
 */
FieldKernelVariation shifted_random_field_kernel_variation(
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta);

FieldKernelVariation displaced_data_field_kernel_variation(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_direction,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta);

/*
 * Differentiate only the momentum-dependent local-PNG reconstruction
 * denominator while keeping the pre-reconstruction tracer kernels fixed.
 * The direction is normalized to
 *
 *   d / d(local_png_bias_amplitude)
 *
 * at zero amplitude, with local_png_bias_amplitude=fNL_rec*bphi_rec.
 */
FieldKernelVariation
reconstructed_field_kernel_local_png_denominator_variation(
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& fixed_reconstruction,
    const PowerSpectrum& transfer,
    double kmin,
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

enum class EstimatorFieldRole {
    DisplacedData,
    ShiftedRandom
};

/*
 * Additive marked-field form of the explicit D/R estimator fields.  The
 * three marked providers distinguish noise carried by the data-catalog
 * output density, by the finite random-catalog output density, and by the
 * common reconstruction displacement source.  This separation is required
 * to keep halo and random Poisson species independent.
 *
 * For DisplacedData the returned terms are F plus the pure-shift R forest;
 * for ShiftedRandom they are the pure-shift forest plus the optional random
 * output-density mark.  Consequently, with random_density_marked=0,
 * D-R reproduces reconstructed_multimarked_field_terms() term by term after
 * summation.  No historical tied call is redirected through this API.
 */
std::vector<RoleResolvedMarkedFieldTerm>
estimator_field_multimarked_terms(
    EstimatorFieldRole field,
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& data_density_marked,
    const MarkedFieldKernelProvider& random_density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta);

SparsePolynomial estimator_field_marked_kernel(
    EstimatorFieldRole field,
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& data_density_marked,
    const MarkedFieldKernelProvider& random_density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const Vec3& stochastic_momentum);

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
 * The one-provider constructor retains the production tied map, where the
 * reconstruction displacement is sourced by the same wrapped tracer field
 * as the density block.  The two-provider constructor makes those roles
 * explicit: every density partition uses density_base and every non-density
 * partition block uses shift_base.  This is required by frozen-source and
 * source-only response experiments, without changing the tied API.
 * The Cartesian CIC window remains inside ReconstructionConfig and is
 * evaluated on the actual block vector.
 */
class ReconstructedFieldKernelProvider final : public FieldKernelProvider {
public:
    ReconstructedFieldKernelProvider(
        const FieldKernelProvider& base,
        marisa_b_halo_v1::ReconstructionConfig reconstruction);
    ReconstructedFieldKernelProvider(
        const FieldKernelProvider& density_base,
        const FieldKernelProvider& shift_base,
        marisa_b_halo_v1::ReconstructionConfig reconstruction);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }
    bool supports_triangle_plane_reflection() const noexcept override {
        // A Cartesian CIC window is tied to the mesh axes.  Reflection in an
        // arbitrary external-triangle plane is therefore not a symmetry.
        return false;
    }

    // base() is retained as a compatibility alias for the observed-density
    // provider used by the continuous zero-external-momentum limit.
    const FieldKernelProvider& base() const noexcept { return density_base_; }
    const FieldKernelProvider& density_base() const noexcept {
        return density_base_;
    }
    const FieldKernelProvider& shift_base() const noexcept {
        return shift_base_;
    }
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction() const noexcept {
        return reconstruction_;
    }

private:
    const FieldKernelProvider& density_base_;
    const FieldKernelProvider& shift_base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    std::string name_;
};

class ShiftedRandomFieldKernelProvider final : public FieldKernelProvider {
public:
    ShiftedRandomFieldKernelProvider(
        const FieldKernelProvider& shift_base,
        marisa_b_halo_v1::ReconstructionConfig reconstruction);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }
    bool supports_triangle_plane_reflection() const noexcept override {
        return false;
    }

private:
    const FieldKernelProvider& shift_base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    std::string name_;
};

class DisplacedDataFieldKernelProvider final : public FieldKernelProvider {
public:
    DisplacedDataFieldKernelProvider(
        const FieldKernelProvider& density_base,
        const FieldKernelProvider& shift_base,
        marisa_b_halo_v1::ReconstructionConfig reconstruction);

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override;
    std::string_view name() const noexcept override { return name_; }
    bool supports_triangle_plane_reflection() const noexcept override {
        return false;
    }

private:
    const FieldKernelProvider& density_base_;
    const FieldKernelProvider& shift_base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    std::string name_;
};

}  // namespace marisa_b_eft_v2

#endif
