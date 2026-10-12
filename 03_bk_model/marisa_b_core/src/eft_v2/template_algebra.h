#ifndef MARISA_B_EFT_V2_TEMPLATE_ALGEBRA_H
#define MARISA_B_EFT_V2_TEMPLATE_ALGEBRA_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "parameter_registry.h"

namespace marisa_b_halo_v1 { class Polynomial; }

namespace marisa_b_eft_v2 {

struct MonomialFactor {
    ParameterId id;
    std::uint8_t power;
};

class MonomialKey {
public:
    MonomialKey() = default;
    explicit MonomialKey(std::vector<MonomialFactor> factors);
    static MonomialKey variable(ParameterId id);

    const std::vector<MonomialFactor>& factors() const noexcept { return factors_; }
    int total_degree() const noexcept;
    std::string canonical_string() const;

    friend bool operator<(const MonomialKey& left, const MonomialKey& right) noexcept;
    friend bool operator==(const MonomialKey& left, const MonomialKey& right) noexcept;
    friend MonomialKey multiply(const MonomialKey& left, const MonomialKey& right);

private:
    std::vector<MonomialFactor> factors_;
};

MonomialKey multiply(const MonomialKey& left, const MonomialKey& right);

class SparsePolynomial {
public:
    using Terms = std::map<MonomialKey, double>;

    SparsePolynomial() = default;
    static SparsePolynomial constant(double value);
    static SparsePolynomial variable(ParameterId id);
    static SparsePolynomial deserialize(const std::string& serialized);

    const Terms& terms() const noexcept { return terms_; }
    bool empty() const noexcept { return terms_.empty(); }
    double coefficient(const MonomialKey& key) const;
    double evaluate(const std::array<double, kParameterCount>& values) const;
    std::string serialize() const;

    SparsePolynomial& operator+=(const SparsePolynomial& other);
    SparsePolynomial& operator-=(const SparsePolynomial& other);
    SparsePolynomial& operator*=(double scalar);

    friend SparsePolynomial operator+(SparsePolynomial left, const SparsePolynomial& right);
    friend SparsePolynomial operator-(SparsePolynomial left, const SparsePolynomial& right);
    friend SparsePolynomial operator*(const SparsePolynomial& left, const SparsePolynomial& right);
    friend SparsePolynomial operator*(SparsePolynomial value, double scalar);
    friend SparsePolynomial operator*(double scalar, SparsePolynomial value);

private:
    void add_term(const MonomialKey& key, double coefficient);
    Terms terms_;
};

SparsePolynomial operator+(SparsePolynomial left, const SparsePolynomial& right);
SparsePolynomial operator-(SparsePolynomial left, const SparsePolynomial& right);
SparsePolynomial operator*(const SparsePolynomial& left, const SparsePolynomial& right);
SparsePolynomial operator*(SparsePolynomial value, double scalar);
SparsePolynomial operator*(double scalar, SparsePolynomial value);

template<typename T>
struct DualTemplate {
    T value{};
    T direction{};
};

template<typename T>
DualTemplate<T> operator+(DualTemplate<T> left, const DualTemplate<T>& right) {
    left.value += right.value;
    left.direction += right.direction;
    return left;
}

template<typename T>
DualTemplate<T> operator*(const DualTemplate<T>& left, const DualTemplate<T>& right) {
    return DualTemplate<T>{
        left.value * right.value,
        left.direction * right.value + left.value * right.direction};
}

template<typename T>
DualTemplate<T> operator*(DualTemplate<T> value, double scalar) {
    value.value *= scalar;
    value.direction *= scalar;
    return value;
}

template<typename T>
DualTemplate<T> operator*(double scalar, DualTemplate<T> value) {
    return value * scalar;
}

struct TemplateMetadata {
    std::string diagram;
    std::string operator_origin;
    std::string power_counting;
    std::string scheme;
    std::string canonical_string() const;
};

struct TemplateKey {
    MonomialKey monomial;
    TemplateMetadata metadata;
    std::string canonical_string() const;
};

class TemplateRegistry {
public:
    void add(TemplateKey key);
    void freeze();
    bool frozen() const noexcept { return frozen_; }
    std::size_t size() const noexcept { return entries_.size(); }
    const TemplateKey& at(std::size_t index) const;
    std::size_t index_of(const TemplateKey& key) const;
    std::string canonical_string() const;
    std::string stable_sha256() const;

private:
    bool frozen_ = false;
    std::vector<TemplateKey> entries_;
};

class LinearNuisanceBlock {
public:
    void add(ParameterId id, std::vector<double> shape);
    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return ids_.size(); }
    const std::vector<ParameterId>& ids() const noexcept { return ids_; }
    const std::vector<std::vector<double>>& shapes() const noexcept { return shapes_; }
    std::vector<double> evaluate(const std::array<double, kParameterCount>& values) const;

private:
    std::size_t rows_ = 0;
    std::vector<ParameterId> ids_;
    std::vector<std::vector<double>> shapes_;
};

SparsePolynomial adapt_v1_polynomial(const marisa_b_halo_v1::Polynomial& input);

}  // namespace marisa_b_eft_v2

#endif
