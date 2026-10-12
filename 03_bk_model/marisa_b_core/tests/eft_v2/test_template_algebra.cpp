#include "template_algebra.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo_v1.h"

namespace eft = marisa_b_eft_v2;
namespace hv1 = marisa_b_halo_v1;

namespace {

int checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
    ++checks;
    const double scale = std::max({1.0, std::fabs(actual), std::fabs(expected)});
    if (std::fabs(actual - expected) > tolerance * scale) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual)
                                 + ", expected=" + std::to_string(expected));
    }
}

template<typename Function>
void require_throws(Function function, const std::string& message) {
    ++checks;
    try { function(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

hv1::Polynomial v1_power(hv1::Polynomial base, int exponent) {
    hv1::Polynomial result = hv1::Polynomial::constant(1.0);
    for (int index = 0; index < exponent; ++index) result = result * base;
    return result;
}

void test_parameter_registry() {
    require(eft::parameter_table().size() == 28, "parameter count");
    for (std::size_t index = 0; index < eft::parameter_table().size(); ++index) {
        const auto& item = eft::parameter_table()[index];
        require(static_cast<std::size_t>(item.id) == index, "stable parameter enum ordering");
        require(eft::parameter_id(item.name) == item.id, "parameter name round trip");
    }
    require(eft::registry_sha256().size() == 64, "registry SHA-256 length");
    require(
        eft::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223"
                                  "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 standard vector");
    require_throws([] { (void)eft::parameter_id("not_a_parameter"); }, "unknown parameter rejection");
}

void test_sparse_algebra_random() {
    const eft::SparsePolynomial b1 = eft::SparsePolynomial::variable(eft::ParameterId::B1);
    const eft::SparsePolynomial b2 = eft::SparsePolynomial::variable(eft::ParameterId::B2);
    const eft::SparsePolynomial g2 = eft::SparsePolynomial::variable(eft::ParameterId::Gamma2);
    const eft::SparsePolynomial b3 = eft::SparsePolynomial::variable(eft::ParameterId::B3);
    const eft::SparsePolynomial expression =
        (2.0 * b1 - 3.0 * b2 + 0.25 * g2) * (b1 + b2) * (1.5 * b1 - 0.3 * b3)
        + 7.0 * b1 * g2 - 0.125 * b3;

    std::mt19937_64 random(120722);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int sample = 0; sample < 1000; ++sample) {
        std::array<double, eft::kParameterCount> values = {};
        for (double& value : values) value = distribution(random);
        const double x1 = values[static_cast<std::size_t>(eft::ParameterId::B1)];
        const double x2 = values[static_cast<std::size_t>(eft::ParameterId::B2)];
        const double xg = values[static_cast<std::size_t>(eft::ParameterId::Gamma2)];
        const double x3 = values[static_cast<std::size_t>(eft::ParameterId::B3)];
        const long double reference =
            (2.0L*x1-3.0L*x2+0.25L*xg)*(x1+x2)*(1.5L*x1-0.3L*x3)
            + 7.0L*x1*xg - 0.125L*x3;
        require_close(expression.evaluate(values), static_cast<double>(reference), 1.0e-13,
                      "random sparse polynomial reference");
    }
    const std::string serialized = expression.serialize();
    const eft::SparsePolynomial restored = eft::SparsePolynomial::deserialize(serialized);
    require(restored.serialize() == serialized, "polynomial serialization round trip");

    require_throws(
        [] { (void)eft::MonomialKey({
            {eft::ParameterId::B1,1}, {eft::ParameterId::B1,1}}); },
        "repeated monomial ID rejection");
    require_throws(
        [] { (void)eft::SparsePolynomial::variable(eft::ParameterId::BNabla2Delta); },
        "linear nuisance excluded from polynomial");
    require_throws(
        [&b1] { (void)(b1*b1*b1*b1*b1*b1*b1); },
        "over-degree rejection");
}

void test_dual_template() {
    using Dual = eft::DualTemplate<double>;
    const double x = 1.7;
    const double y = -0.4;
    const Dual dx{x, 1.0};
    const Dual dy{y, 0.0};
    const Dual result = (dx * dx + 3.0 * dx * dy) * dx;
    require_close(result.value, (x*x+3*x*y)*x, 1.0e-14, "dual value");
    require_close(result.direction, 3*x*x+6*x*y, 1.0e-14, "dual product rule");
}

eft::TemplateKey make_key(int index) {
    return eft::TemplateKey{
        eft::MonomialKey::variable(static_cast<eft::ParameterId>(index % 11)),
        eft::TemplateMetadata{
            index % 2 == 0 ? "B222" : "B411",
            "operator-" + std::to_string(index),
            "one-loop",
            "DR/MS"}};
}

void test_template_registry() {
    std::vector<eft::TemplateKey> keys;
    for (int index = 0; index < 31; ++index) keys.push_back(make_key(index));
    eft::TemplateRegistry forward;
    for (const auto& key : keys) forward.add(key);
    forward.freeze();
    std::reverse(keys.begin(), keys.end());
    eft::TemplateRegistry reverse;
    for (const auto& key : keys) reverse.add(key);
    reverse.freeze();
    require(forward.canonical_string() == reverse.canonical_string(), "generation-order invariance");
    require(forward.stable_sha256() == reverse.stable_sha256(), "registry hash invariance");
    require(forward.stable_sha256().size() == 64, "template SHA-256 length");
    for (std::size_t index = 0; index < forward.size(); ++index) {
        require(forward.index_of(forward.at(index)) == index, "registry index lookup");
    }
    require_throws([&forward] { forward.add(make_key(99)); }, "frozen registry mutation rejection");
    eft::TemplateRegistry duplicate;
    duplicate.add(make_key(1));
    duplicate.add(make_key(1));
    require_throws([&duplicate] { duplicate.freeze(); }, "duplicate template rejection");
}

void test_linear_nuisance() {
    eft::LinearNuisanceBlock block;
    block.add(eft::ParameterId::BNabla2Delta, {1.0, 2.0, 3.0});
    block.add(eft::ParameterId::AshotResidual, {-1.0, 0.5, 4.0});
    std::array<double, eft::kParameterCount> values = {};
    values[static_cast<std::size_t>(eft::ParameterId::BNabla2Delta)] = 2.0;
    values[static_cast<std::size_t>(eft::ParameterId::AshotResidual)] = -3.0;
    const std::vector<double> result = block.evaluate(values);
    require_close(result[0], 5.0, 1.0e-14, "linear nuisance row 0");
    require_close(result[1], 2.5, 1.0e-14, "linear nuisance row 1");
    require_close(result[2], -6.0, 1.0e-14, "linear nuisance row 2");
    require_throws(
        [&block] { block.add(eft::ParameterId::A3Mixed, {1.0}); },
        "linear nuisance row mismatch rejection");
}

void test_v1_compatibility() {
    const hv1::Polynomial b1 = hv1::Polynomial::variable_b1();
    const hv1::Polynomial b2 = hv1::Polynomial::variable_b2();
    const hv1::Polynomial bk = hv1::Polynomial::variable_bK2();
    hv1::Polynomial all;
    int coefficient = 1;
    for (int i = 0; i <= 6; ++i) {
        for (int j = 0; j <= 6-i; ++j) {
            for (int k = 0; k <= 6-i-j; ++k) {
                all += static_cast<double>(coefficient++)
                       * v1_power(b1,i) * v1_power(b2,j) * v1_power(bk,k);
            }
        }
    }
    require(all.terms().size() == 84, "complete v1 84-term polynomial");
    const eft::SparsePolynomial adapted = eft::adapt_v1_polynomial(all);
    std::mt19937_64 random(84);
    std::uniform_real_distribution<double> distribution(-1.2, 1.2);
    for (int sample = 0; sample < 1000; ++sample) {
        const hv1::BiasPoint old{distribution(random),distribution(random),distribution(random)};
        std::array<double, eft::kParameterCount> values = {};
        values[static_cast<std::size_t>(eft::ParameterId::B1)] = old.b1;
        values[static_cast<std::size_t>(eft::ParameterId::Gamma2)] = old.bK2;
        values[static_cast<std::size_t>(eft::ParameterId::B2)] = old.b2 + 4.0*old.bK2/3.0;
        long double reference = 0.0L;
        for (const hv1::PolynomialTerm& term : all.terms()) {
            long double value = term.coefficient;
            for (int power = 0; power < term.powers.b1; ++power) value *= old.b1;
            for (int power = 0; power < term.powers.b2; ++power) value *= old.b2;
            for (int power = 0; power < term.powers.bK2; ++power) value *= old.bK2;
            reference += value;
        }
        require_close(adapted.evaluate(values), static_cast<double>(reference), 1.0e-13,
                      "v1 84-term convention adapter");
    }
}

}  // namespace

int main() {
    try {
        test_parameter_registry();
        test_sparse_algebra_random();
        test_dual_template();
        test_template_registry();
        test_linear_nuisance();
        test_v1_compatibility();
        std::cout << "EFT-v2 template algebra checks passed: " << checks << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "EFT-v2 template algebra test failed: " << error.what() << "\n";
        return 1;
    }
}
