#ifndef MARISA_B_EFT_V2_MODEL_CONFIG_H
#define MARISA_B_EFT_V2_MODEL_CONFIG_H

#include <cstdint>
#include <string_view>

#include "bias_operators.h"

namespace marisa_b_eft_v2 {

enum class BiasTier : std::uint8_t {
    Full,
    Coevolution,
};

enum class ReconStage : std::uint8_t {
    Pre,
    Post,
};

enum class PngOrder : std::uint8_t {
    Gaussian,
    Linear,
    Quadratic,
};

struct ModelConfig {
    BiasTier bias_tier = BiasTier::Full;
    ReconStage recon_stage = ReconStage::Pre;
    PngOrder png_order = PngOrder::Gaussian;
};

/* Current production selection; full remains an explicit cross-check tier. */
ModelConfig production_model_config(ReconStage stage);

std::string_view bias_tier_name(BiasTier value);
std::string_view recon_stage_name(ReconStage value);
std::string_view png_order_name(PngOrder value);

BiasTier parse_bias_tier(std::string_view value);
ReconStage parse_recon_stage(std::string_view value);
PngOrder parse_png_order(std::string_view value);

/*
 * Resolve the eleven deterministic bias coefficients for a model tier.
 *
 * Full returns the input exactly.  Coevolution applies Eggemeier et al.
 * (2021), Eqs. (20)--(24), with the five non-local Lagrangian coefficients
 * set to zero, matching the current production inference convention.
 */
BiasValues resolve_bias_values(
    BiasTier tier,
    const BiasValues& input);
BiasValues resolve_bias_values(
    const ModelConfig& config,
    const BiasValues& input);

}  // namespace marisa_b_eft_v2

#endif
