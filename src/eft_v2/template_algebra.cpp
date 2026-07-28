#include "template_algebra.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include "halo_v1.h"

namespace marisa_b_eft_v2 {
namespace {

double integer_power(double base, int exponent) {
    double result = 1.0;
    for (int index = 0; index < exponent; ++index) result *= base;
    return result;
}

double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    double result = 1.0;
    for (int index = 1; index <= k; ++index) {
        result *= static_cast<double>(n - k + index) / static_cast<double>(index);
    }
    return result;
}

}  // namespace

MonomialKey::MonomialKey(std::vector<MonomialFactor> factors)
    : factors_(std::move(factors)) {
    ParameterId previous = ParameterId::Count;
    bool first = true;
    int degree = 0;
    for (const MonomialFactor& factor : factors_) {
        if (!is_bias_parameter(factor.id)) {
            throw std::invalid_argument("only deterministic bias parameters may enter a monomial");
        }
        if (factor.power == 0) throw std::invalid_argument("monomial factor has zero power");
        if (!first && static_cast<int>(factor.id) <= static_cast<int>(previous)) {
            throw std::invalid_argument("monomial IDs must be strictly ordered and non-repeated");
        }
        first = false;
        previous = factor.id;
        degree += factor.power;
    }
    if (degree > 6) throw std::overflow_error("EFT-v2 monomial degree exceeds six");
}

MonomialKey MonomialKey::variable(ParameterId id) {
    return MonomialKey({MonomialFactor{id, 1}});
}

int MonomialKey::total_degree() const noexcept {
    int result = 0;
    for (const MonomialFactor& factor : factors_) result += factor.power;
    return result;
}

std::string MonomialKey::canonical_string() const {
    if (factors_.empty()) return "1";
    std::ostringstream output;
    for (std::size_t index = 0; index < factors_.size(); ++index) {
        if (index != 0) output << '*';
        output << parameter_info(factors_[index].id).name << '^'
               << static_cast<int>(factors_[index].power);
    }
    return output.str();
}

bool operator<(const MonomialKey& left, const MonomialKey& right) noexcept {
    const std::size_t common = std::min(left.factors_.size(), right.factors_.size());
    for (std::size_t index = 0; index < common; ++index) {
        if (left.factors_[index].id != right.factors_[index].id) {
            return static_cast<int>(left.factors_[index].id)
                   < static_cast<int>(right.factors_[index].id);
        }
        if (left.factors_[index].power != right.factors_[index].power) {
            return left.factors_[index].power < right.factors_[index].power;
        }
    }
    return left.factors_.size() < right.factors_.size();
}

bool operator==(const MonomialKey& left, const MonomialKey& right) noexcept {
    if (left.factors_.size() != right.factors_.size()) return false;
    for (std::size_t index = 0; index < left.factors_.size(); ++index) {
        if (left.factors_[index].id != right.factors_[index].id
            || left.factors_[index].power != right.factors_[index].power) return false;
    }
    return true;
}

MonomialKey multiply(const MonomialKey& left, const MonomialKey& right) {
    std::vector<MonomialFactor> result;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < left.factors_.size() || j < right.factors_.size()) {
        if (j == right.factors_.size()
            || (i < left.factors_.size()
                && static_cast<int>(left.factors_[i].id) < static_cast<int>(right.factors_[j].id))) {
            result.push_back(left.factors_[i++]);
        } else if (i == left.factors_.size()
                   || static_cast<int>(right.factors_[j].id) < static_cast<int>(left.factors_[i].id)) {
            result.push_back(right.factors_[j++]);
        } else {
            const int power = left.factors_[i].power + right.factors_[j].power;
            if (power > 255) throw std::overflow_error("monomial factor power overflow");
            result.push_back(MonomialFactor{left.factors_[i].id, static_cast<std::uint8_t>(power)});
            ++i;
            ++j;
        }
    }
    return MonomialKey(std::move(result));
}

SparsePolynomial SparsePolynomial::constant(double value) {
    SparsePolynomial result;
    result.add_term(MonomialKey(), value);
    return result;
}

SparsePolynomial SparsePolynomial::variable(ParameterId id) {
    SparsePolynomial result;
    result.add_term(MonomialKey::variable(id), 1.0);
    return result;
}

void SparsePolynomial::add_term(const MonomialKey& key, double coefficient) {
    if (!std::isfinite(coefficient)) throw std::invalid_argument("non-finite polynomial coefficient");
    if (coefficient == 0.0) return;
    const double updated = terms_[key] + coefficient;
    if (updated == 0.0) terms_.erase(key);
    else terms_[key] = updated;
}

double SparsePolynomial::coefficient(const MonomialKey& key) const {
    const auto match = terms_.find(key);
    return match == terms_.end() ? 0.0 : match->second;
}

double SparsePolynomial::evaluate(const std::array<double, kParameterCount>& values) const {
    long double result = 0.0L;
    for (const auto& term : terms_) {
        long double value = term.second;
        for (const MonomialFactor& factor : term.first.factors()) {
            const double parameter = values[static_cast<std::size_t>(factor.id)];
            if (!std::isfinite(parameter)) throw std::invalid_argument("non-finite parameter value");
            long double factor_value = 1.0L;
            for (int index = 0; index < factor.power; ++index) factor_value *= parameter;
            value *= factor_value;
        }
        result += value;
    }
    return static_cast<double>(result);
}

std::string SparsePolynomial::serialize() const {
    std::ostringstream output;
    output << std::setprecision(17);
    for (const auto& term : terms_) output << term.first.canonical_string() << '=' << term.second << '\n';
    return output.str();
}

SparsePolynomial SparsePolynomial::deserialize(const std::string& serialized) {
    SparsePolynomial result;
    std::istringstream input(serialized);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos) throw std::invalid_argument("invalid polynomial serialization");
        const double coefficient = std::stod(line.substr(equal + 1));
        std::vector<MonomialFactor> factors;
        const std::string key = line.substr(0, equal);
        if (key != "1") {
            std::size_t start = 0;
            while (start < key.size()) {
                const std::size_t end = key.find('*', start);
                const std::string factor = key.substr(start, end == std::string::npos ? end : end - start);
                const std::size_t caret = factor.find('^');
                if (caret == std::string::npos) throw std::invalid_argument("invalid monomial serialization");
                const int power = std::stoi(factor.substr(caret + 1));
                if (power < 1 || power > 255) throw std::invalid_argument("invalid serialized power");
                factors.push_back(MonomialFactor{
                    parameter_id(factor.substr(0, caret)), static_cast<std::uint8_t>(power)});
                if (end == std::string::npos) break;
                start = end + 1;
            }
        }
        result.add_term(MonomialKey(std::move(factors)), coefficient);
    }
    return result;
}

SparsePolynomial& SparsePolynomial::operator+=(const SparsePolynomial& other) {
    for (const auto& term : other.terms_) add_term(term.first, term.second);
    return *this;
}

SparsePolynomial& SparsePolynomial::operator-=(const SparsePolynomial& other) {
    for (const auto& term : other.terms_) add_term(term.first, -term.second);
    return *this;
}

SparsePolynomial& SparsePolynomial::operator*=(double scalar) {
    if (!std::isfinite(scalar)) throw std::invalid_argument("non-finite polynomial scale");
    if (scalar == 0.0) { terms_.clear(); return *this; }
    for (auto& term : terms_) term.second *= scalar;
    return *this;
}

SparsePolynomial operator+(SparsePolynomial left, const SparsePolynomial& right) { return left += right; }
SparsePolynomial operator-(SparsePolynomial left, const SparsePolynomial& right) { return left -= right; }

SparsePolynomial operator*(const SparsePolynomial& left, const SparsePolynomial& right) {
    SparsePolynomial result;
    for (const auto& left_term : left.terms_) {
        for (const auto& right_term : right.terms_) {
            result.add_term(
                multiply(left_term.first, right_term.first),
                left_term.second * right_term.second);
        }
    }
    return result;
}

SparsePolynomial operator*(SparsePolynomial value, double scalar) { return value *= scalar; }
SparsePolynomial operator*(double scalar, SparsePolynomial value) { return value *= scalar; }

std::string TemplateMetadata::canonical_string() const {
    return diagram + "\x1f" + operator_origin + "\x1f" + power_counting + "\x1f" + scheme;
}

std::string TemplateKey::canonical_string() const {
    return metadata.canonical_string() + "\x1f" + monomial.canonical_string();
}

void TemplateRegistry::add(TemplateKey key) {
    if (frozen_) throw std::logic_error("cannot add to a frozen TemplateRegistry");
    entries_.push_back(std::move(key));
}

void TemplateRegistry::freeze() {
    if (frozen_) throw std::logic_error("TemplateRegistry is already frozen");
    std::sort(entries_.begin(), entries_.end(), [](const TemplateKey& left, const TemplateKey& right) {
        return left.canonical_string() < right.canonical_string();
    });
    for (std::size_t index = 1; index < entries_.size(); ++index) {
        if (entries_[index - 1].canonical_string() == entries_[index].canonical_string()) {
            throw std::invalid_argument("duplicate template key");
        }
    }
    frozen_ = true;
}

const TemplateKey& TemplateRegistry::at(std::size_t index) const {
    if (!frozen_) throw std::logic_error("TemplateRegistry must be frozen before access");
    return entries_.at(index);
}

std::size_t TemplateRegistry::index_of(const TemplateKey& key) const {
    if (!frozen_) throw std::logic_error("TemplateRegistry must be frozen before lookup");
    const std::string target = key.canonical_string();
    const auto match = std::lower_bound(entries_.begin(), entries_.end(), target,
        [](const TemplateKey& item, const std::string& value) {
            return item.canonical_string() < value;
        });
    if (match == entries_.end() || match->canonical_string() != target) {
        throw std::out_of_range("template is absent from registry");
    }
    return static_cast<std::size_t>(match - entries_.begin());
}

std::string TemplateRegistry::canonical_string() const {
    if (!frozen_) throw std::logic_error("TemplateRegistry must be frozen before serialization");
    std::ostringstream output;
    output << "parameter-registry=" << registry_sha256() << '\n';
    for (const TemplateKey& item : entries_) output << item.canonical_string() << '\n';
    return output.str();
}

std::string TemplateRegistry::stable_sha256() const { return sha256_hex(canonical_string()); }

void LinearNuisanceBlock::add(ParameterId id, std::vector<double> shape) {
    if (is_bias_parameter(id)) throw std::invalid_argument("bias parameter cannot enter linear nuisance block");
    if (std::find(ids_.begin(), ids_.end(), id) != ids_.end()) {
        throw std::invalid_argument("duplicate linear nuisance parameter");
    }
    if (ids_.empty()) rows_ = shape.size();
    if (shape.size() != rows_) throw std::invalid_argument("linear nuisance shape row mismatch");
    if (!std::all_of(shape.begin(), shape.end(), [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("linear nuisance shape contains non-finite value");
    }
    ids_.push_back(id);
    shapes_.push_back(std::move(shape));
}

std::vector<double> LinearNuisanceBlock::evaluate(
    const std::array<double, kParameterCount>& values) const {
    std::vector<double> result(rows_, 0.0);
    for (std::size_t column = 0; column < ids_.size(); ++column) {
        const double coefficient = values[static_cast<std::size_t>(ids_[column])];
        for (std::size_t row = 0; row < rows_; ++row) result[row] += coefficient * shapes_[column][row];
    }
    return result;
}

SparsePolynomial adapt_v1_polynomial(const marisa_b_halo_v1::Polynomial& input) {
    const SparsePolynomial b1 = SparsePolynomial::variable(ParameterId::B1);
    const SparsePolynomial b2 = SparsePolynomial::variable(ParameterId::B2);
    const SparsePolynomial gamma2 = SparsePolynomial::variable(ParameterId::Gamma2);
    SparsePolynomial result;
    for (const marisa_b_halo_v1::PolynomialTerm& term : input.terms()) {
        SparsePolynomial expanded = SparsePolynomial::constant(term.coefficient);
        for (int count = 0; count < term.powers.b1; ++count) expanded = expanded * b1;
        // b2_v1 = b2_COBRA - 4 gamma2 / 3.
        SparsePolynomial b2_power;
        for (int j = 0; j <= term.powers.b2; ++j) {
            SparsePolynomial piece = SparsePolynomial::constant(
                binomial(term.powers.b2, j) * integer_power(-4.0 / 3.0, j));
            for (int count = 0; count < term.powers.b2 - j; ++count) piece = piece * b2;
            for (int count = 0; count < j; ++count) piece = piece * gamma2;
            b2_power += piece;
        }
        expanded = expanded * b2_power;
        for (int count = 0; count < term.powers.bK2; ++count) expanded = expanded * gamma2;
        result += expanded;
    }
    return result;
}

}  // namespace marisa_b_eft_v2
