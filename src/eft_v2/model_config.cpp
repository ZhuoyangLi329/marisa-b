#include "model_config.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace marisa_b_eft_v2 {
namespace {

constexpr std::size_t index(ParameterId id) {
    return static_cast<std::size_t>(id);
}

std::invalid_argument unknown_value(
    const char* kind,
    std::string_view value) {
    return std::invalid_argument(
        std::string("unknown EFT-v2 ") + kind + ": "
        + std::string(value));
}

}  // namespace

ModelConfig production_model_config(ReconStage stage) {
    ModelConfig config;
    config.bias_tier = BiasTier::Coevolution;
    config.recon_stage = stage;
    config.png_order = PngOrder::Quadratic;
    return config;
}

std::string_view bias_tier_name(BiasTier value) {
    switch (value) {
        case BiasTier::Full:
            return "full";
        case BiasTier::Coevolution:
            return "coevolution";
    }
    throw std::invalid_argument("invalid EFT-v2 BiasTier");
}

std::string_view recon_stage_name(ReconStage value) {
    switch (value) {
        case ReconStage::Pre:
            return "pre";
        case ReconStage::Post:
            return "post";
    }
    throw std::invalid_argument("invalid EFT-v2 ReconStage");
}

std::string_view png_order_name(PngOrder value) {
    switch (value) {
        case PngOrder::Gaussian:
            return "gaussian";
        case PngOrder::Linear:
            return "linear";
        case PngOrder::Quadratic:
            return "quadratic";
    }
    throw std::invalid_argument("invalid EFT-v2 PngOrder");
}

BiasTier parse_bias_tier(std::string_view value) {
    if (value == "full") return BiasTier::Full;
    if (value == "coevolution") return BiasTier::Coevolution;
    throw unknown_value("bias tier", value);
}

ReconStage parse_recon_stage(std::string_view value) {
    if (value == "pre") return ReconStage::Pre;
    if (value == "post") return ReconStage::Post;
    throw unknown_value("reconstruction stage", value);
}

PngOrder parse_png_order(std::string_view value) {
    if (value == "gaussian") return PngOrder::Gaussian;
    if (value == "linear") return PngOrder::Linear;
    if (value == "quadratic") return PngOrder::Quadratic;
    throw unknown_value("PNG order", value);
}

BiasValues resolve_bias_values(
    BiasTier tier,
    const BiasValues& input) {
    if (tier == BiasTier::Full) return input;
    if (tier != BiasTier::Coevolution) {
        throw std::invalid_argument("invalid EFT-v2 BiasTier");
    }

    BiasValues result = input;
    const double b1 = input[index(ParameterId::B1)];
    const double b2 = input[index(ParameterId::B2)];
    const double gamma2 = input[index(ParameterId::Gamma2)];
    const double gamma2x = input[index(ParameterId::Gamma2x)];
    const double gamma3 = input[index(ParameterId::Gamma3)];

    const double gamma21 =
        (2.0 / 21.0) * (b1 - 1.0)
        + (6.0 / 7.0) * gamma2;
    result[index(ParameterId::Gamma21)] = gamma21;
    result[index(ParameterId::Gamma21x)] =
        (2.0 / 21.0) * b2
        + (6.0 / 7.0) * gamma2x;
    result[index(ParameterId::Gamma211)] =
        (5.0 / 77.0) * (b1 - 1.0)
        + (15.0 / 14.0) * gamma2
        - (9.0 / 7.0) * gamma3
        + gamma21;
    result[index(ParameterId::Gamma22)] =
        -(6.0 / 539.0) * (b1 - 1.0)
        - (9.0 / 49.0) * gamma2;
    result[index(ParameterId::Gamma31)] =
        -(4.0 / 11.0) * (b1 - 1.0)
        - 6.0 * gamma2;
    return result;
}

BiasValues resolve_bias_values(
    const ModelConfig& config,
    const BiasValues& input) {
    return resolve_bias_values(config.bias_tier, input);
}

}  // namespace marisa_b_eft_v2
