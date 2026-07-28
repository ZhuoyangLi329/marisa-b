#include "model_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace eft = marisa_b_eft_v2;

namespace {

int checks = 0;

constexpr std::size_t index(eft::ParameterId id) {
    return static_cast<std::size_t>(id);
}

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,
    double expected,
    const std::string& message) {
    ++checks;
    const double scale =
        std::max({1.0, std::fabs(actual), std::fabs(expected)});
    if (std::fabs(actual - expected) > 1.0e-14 * scale) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual)
            + ", expected=" + std::to_string(expected));
    }
}

template<typename Function>
void require_throws(Function function, const std::string& message) {
    ++checks;
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_enum_round_trips() {
    for (const eft::BiasTier value : {
             eft::BiasTier::Full,
             eft::BiasTier::Coevolution}) {
        require(
            eft::parse_bias_tier(eft::bias_tier_name(value)) == value,
            "bias-tier string round trip");
    }
    for (const eft::ReconStage value : {
             eft::ReconStage::Pre,
             eft::ReconStage::Post}) {
        require(
            eft::parse_recon_stage(eft::recon_stage_name(value)) == value,
            "reconstruction-stage string round trip");
    }
    for (const eft::PngOrder value : {
             eft::PngOrder::Gaussian,
             eft::PngOrder::Linear,
             eft::PngOrder::Quadratic}) {
        require(
            eft::parse_png_order(eft::png_order_name(value)) == value,
            "PNG-order string round trip");
    }
    require_throws(
        [] { (void)eft::parse_bias_tier("uplift"); },
        "unknown bias tier rejected");
    require_throws(
        [] { (void)eft::parse_recon_stage("during"); },
        "unknown reconstruction stage rejected");
    require_throws(
        [] { (void)eft::parse_png_order("cubic"); },
        "unknown PNG order rejected");
}

eft::BiasValues test_bias_values() {
    eft::BiasValues values = {};
    values[index(eft::ParameterId::B1)] = 2.45;
    values[index(eft::ParameterId::B2)] = -0.73;
    values[index(eft::ParameterId::Gamma2)] = 0.19;
    values[index(eft::ParameterId::B3)] = 0.41;
    values[index(eft::ParameterId::Gamma2x)] = -0.27;
    values[index(eft::ParameterId::Gamma3)] = 0.13;
    values[index(eft::ParameterId::Gamma21)] = 11.0;
    values[index(eft::ParameterId::Gamma21x)] = 12.0;
    values[index(eft::ParameterId::Gamma211)] = 13.0;
    values[index(eft::ParameterId::Gamma22)] = 14.0;
    values[index(eft::ParameterId::Gamma31)] = 15.0;
    return values;
}

void test_full_identity() {
    const eft::BiasValues input = test_bias_values();
    require(
        eft::resolve_bias_values(eft::BiasTier::Full, input) == input,
        "full tier is an exact identity");

    eft::ModelConfig config;
    config.bias_tier = eft::BiasTier::Full;
    config.recon_stage = eft::ReconStage::Post;
    config.png_order = eft::PngOrder::Quadratic;
    require(
        eft::resolve_bias_values(config, input) == input,
        "non-bias model fields do not alter full bias values");
}

void test_production_selection() {
    for (const eft::ReconStage stage : {
             eft::ReconStage::Pre,
             eft::ReconStage::Post}) {
        const eft::ModelConfig config =
            eft::production_model_config(stage);
        require(
            config.bias_tier == eft::BiasTier::Coevolution,
            "production bias tier is coevolution");
        require(
            config.recon_stage == stage,
            "production reconstruction stage is explicit");
        require(
            config.png_order == eft::PngOrder::Quadratic,
            "production PNG order is quadratic");
    }
}

void test_five_coevolution_relations() {
    const eft::BiasValues input = test_bias_values();
    const eft::BiasValues actual =
        eft::resolve_bias_values(eft::BiasTier::Coevolution, input);

    const double b1 = input[index(eft::ParameterId::B1)];
    const double b2 = input[index(eft::ParameterId::B2)];
    const double gamma2 = input[index(eft::ParameterId::Gamma2)];
    const double gamma2x = input[index(eft::ParameterId::Gamma2x)];
    const double gamma3 = input[index(eft::ParameterId::Gamma3)];
    const double gamma21 =
        (2.0 / 21.0) * (b1 - 1.0)
        + (6.0 / 7.0) * gamma2;

    require_close(
        actual[index(eft::ParameterId::Gamma21)],
        gamma21,
        "gamma21 coevolution relation");
    require_close(
        actual[index(eft::ParameterId::Gamma21x)],
        (2.0 / 21.0) * b2 + (6.0 / 7.0) * gamma2x,
        "gamma21x coevolution relation");
    require_close(
        actual[index(eft::ParameterId::Gamma211)],
        (5.0 / 77.0) * (b1 - 1.0)
            + (15.0 / 14.0) * gamma2
            - (9.0 / 7.0) * gamma3
            + gamma21,
        "gamma211 coevolution relation");
    require_close(
        actual[index(eft::ParameterId::Gamma22)],
        -(6.0 / 539.0) * (b1 - 1.0)
            - (9.0 / 49.0) * gamma2,
        "gamma22 coevolution relation");
    require_close(
        actual[index(eft::ParameterId::Gamma31)],
        -(4.0 / 11.0) * (b1 - 1.0) - 6.0 * gamma2,
        "gamma31 coevolution relation");

    for (const eft::ParameterId id : {
             eft::ParameterId::B1,
             eft::ParameterId::B2,
             eft::ParameterId::Gamma2,
             eft::ParameterId::B3,
             eft::ParameterId::Gamma2x,
             eft::ParameterId::Gamma3}) {
        require(
            actual[index(id)] == input[index(id)],
            "independent coevolution input remains unchanged");
    }
}

}  // namespace

int main() {
    try {
        test_enum_round_trips();
        test_full_identity();
        test_production_selection();
        test_five_coevolution_relations();
        std::cout
            << "EFT-v2 model-config checks passed: "
            << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "EFT-v2 model-config test failed after "
            << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
