#include "halo_v1.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "marisa_b_native.h"
#include "shell_average.h"

#ifndef MARISA_B_GIT_COMMIT
#define MARISA_B_GIT_COMMIT "diagnostic"
#endif

#ifndef MARISA_B_TREE_SHA256
#define MARISA_B_TREE_SHA256 "diagnostic"
#endif

#ifndef MARISA_B_BUILD_KIND
#define MARISA_B_BUILD_KIND "diagnostic"
#endif

namespace hv1 = marisa_b_halo_v1;
namespace sv1 = marisa_b_shell_v1;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr char kBuildIdentity[] =
    "program=marisa_b_halo_v1_test;git=" MARISA_B_GIT_COMMIT
    ";tree_sha256=" MARISA_B_TREE_SHA256 ";build_kind=" MARISA_B_BUILD_KIND;
int checks = 0;

bool report_enabled() {
    return std::getenv("HALO_V1_REPORT") != nullptr;
}

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

double relative_difference(double left, double right, double floor = 1.0e-14) {
    return std::fabs(left - right) / std::max({std::fabs(left), std::fabs(right), floor});
}

void require_close(
    double actual,
    double expected,
    double relative_tolerance,
    double absolute_tolerance,
    const std::string& message) {
    ++checks;
    const double error = std::fabs(actual - expected);
    if (error > absolute_tolerance
        && error > relative_tolerance * std::max(std::fabs(actual), std::fabs(expected))) {
        std::ostringstream details;
        details.precision(17);
        details << message << ": actual=" << actual << ", expected=" << expected
                << ", absolute_error=" << error;
        throw std::runtime_error(details.str());
    }
}

void require_polynomial_close(
    const hv1::Polynomial& actual,
    const hv1::Polynomial& expected,
    double tolerance,
    const std::string& message) {
    for (int b1 = 0; b1 <= 6; ++b1) {
        for (int b2 = 0; b2 <= 6; ++b2) {
            for (int bK2 = 0; bK2 <= 6; ++bK2) {
                if (b1 + b2 + bK2 > 6) continue;
                require_close(
                    actual.coefficient(b1, b2, bK2),
                    expected.coefficient(b1, b2, bK2),
                    tolerance,
                    tolerance,
                    message);
            }
        }
    }
}

std::vector<hv1::TadpoleTerm> sorted_tadpoles(const hv1::KernelTemplate& kernel) {
    std::vector<hv1::TadpoleTerm> result = kernel.matter_f3_tadpoles;
    std::sort(
        result.begin(), result.end(),
        [](const hv1::TadpoleTerm& left, const hv1::TadpoleTerm& right) {
            return left.external_k < right.external_k;
        });
    return result;
}

void require_kernel_close(
    const hv1::KernelTemplate& actual,
    const hv1::KernelTemplate& expected,
    double tolerance,
    const std::string& message) {
    require_polynomial_close(actual.regular, expected.regular, tolerance, message + " regular");
    const std::vector<hv1::TadpoleTerm> actual_tadpoles = sorted_tadpoles(actual);
    const std::vector<hv1::TadpoleTerm> expected_tadpoles = sorted_tadpoles(expected);
    require(actual_tadpoles.size() == expected_tadpoles.size(), message + " tadpole count");
    for (std::size_t index = 0; index < actual_tadpoles.size(); ++index) {
        require_close(
            actual_tadpoles[index].external_k,
            expected_tadpoles[index].external_k,
            tolerance,
            tolerance,
            message + " tadpole momentum");
        require_polynomial_close(
            actual_tadpoles[index].coefficient,
            expected_tadpoles[index].coefficient,
            tolerance,
            message + " tadpole coefficient");
    }
}

hv1::KernelTemplate scale_test_kernel(hv1::KernelTemplate kernel, double scalar) {
    kernel.regular *= scalar;
    for (hv1::TadpoleTerm& term : kernel.matter_f3_tadpoles) {
        term.coefficient *= scalar;
    }
    return kernel;
}

hv1::KernelTemplate add_test_kernels(
    hv1::KernelTemplate left,
    const hv1::KernelTemplate& right) {
    left.regular += right.regular;
    for (const hv1::TadpoleTerm& addition : right.matter_f3_tadpoles) {
        auto match = std::find_if(
            left.matter_f3_tadpoles.begin(),
            left.matter_f3_tadpoles.end(),
            [&addition](const hv1::TadpoleTerm& term) {
                return std::fabs(term.external_k - addition.external_k) <= 1.0e-12;
            });
        if (match == left.matter_f3_tadpoles.end()) {
            left.matter_f3_tadpoles.push_back(addition);
        } else {
            match->coefficient += addition.coefficient;
        }
    }
    return left;
}

hv1::KernelTemplate multiply_test_kernels(
    const hv1::KernelTemplate& left,
    const hv1::KernelTemplate& right) {
    if (!left.matter_f3_tadpoles.empty() && !right.matter_f3_tadpoles.empty()) {
        throw std::logic_error("test kernel product contains two F3 tadpoles");
    }
    hv1::KernelTemplate result;
    result.regular = left.regular * right.regular;
    for (const hv1::TadpoleTerm& term : left.matter_f3_tadpoles) {
        result.matter_f3_tadpoles.push_back(
            hv1::TadpoleTerm{term.external_k, term.coefficient * right.regular});
    }
    for (const hv1::TadpoleTerm& term : right.matter_f3_tadpoles) {
        result.matter_f3_tadpoles.push_back(
            hv1::TadpoleTerm{term.external_k, term.coefficient * left.regular});
    }
    return result;
}

double cosine(const hv1::Vec3& left, const hv1::Vec3& right) {
    return hv1::dot(left, right) / (hv1::norm(left) * hv1::norm(right));
}

double f2(const hv1::Vec3& left, const hv1::Vec3& right) {
    return marisa_b_native::F2eds(
        hv1::norm(left), hv1::norm(right), cosine(left, right));
}

double f3(const hv1::Vec3& first, const hv1::Vec3& second, const hv1::Vec3& third) {
    return marisa_b_native::F3eds(
        hv1::norm(first),
        hv1::norm(second),
        hv1::norm(third),
        cosine(second, third),
        cosine(first, second),
        cosine(first, third));
}

hv1::Vec3 sum(const std::vector<hv1::Vec3>& vectors) {
    hv1::Vec3 result;
    for (const hv1::Vec3& vector : vectors) result = hv1::add(result, vector);
    return result;
}

class TabulatedPower final : public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table: " + path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double p = 0.0;
            if (row >> k >> p) {
                if (!(k > 0.0 && p > 0.0)) {
                    throw std::runtime_error("power table must contain positive k and P(k)");
                }
                log_k_.push_back(std::log(k));
                log_p_.push_back(std::log(p));
            }
        }
        if (log_k_.size() < 2) throw std::runtime_error("power table has fewer than two rows");
    }

    real Evaluate(real k) const override {
        const double log_k = std::log(std::max<double>(k, std::exp(log_k_.front())));
        if (log_k <= log_k_.front()) return extrapolate(log_k, 0, 1);
        if (log_k >= log_k_.back()) {
            return extrapolate(log_k, log_k_.size() - 2, log_k_.size() - 1);
        }
        const auto upper = std::upper_bound(log_k_.begin(), log_k_.end(), log_k);
        const std::size_t right = static_cast<std::size_t>(upper - log_k_.begin());
        const std::size_t left = right - 1;
        return extrapolate(log_k, left, right);
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no Cosmology object");
    }

private:
    double extrapolate(double log_k, std::size_t left, std::size_t right) const {
        const double fraction = (log_k - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(log_p_[left] + fraction * (log_p_[right] - log_p_[left]));
    }

    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

void test_polynomial() {
    const hv1::Polynomial b1 = hv1::Polynomial::variable_b1();
    const hv1::Polynomial b2 = hv1::Polynomial::variable_b2();
    const hv1::Polynomial bK2 = hv1::Polynomial::variable_bK2();
    const hv1::Polynomial polynomial = (2.0 * b1 - 3.0 * b2 + 0.5 * bK2)
                                       * (b1 + b2);
    const hv1::BiasPoint point{2.1, -0.7, 0.3};
    require_close(
        polynomial.evaluate(point),
        (2.0 * point.b1 - 3.0 * point.b2 + 0.5 * point.bK2)
            * (point.b1 + point.b2),
        1.0e-14,
        1.0e-14,
        "polynomial evaluation");
    require_close(polynomial.coefficient(2, 0, 0), 2.0, 0.0, 0.0, "b1^2 coefficient");
    require_close(polynomial.coefficient(0, 2, 0), -3.0, 0.0, 0.0, "b2^2 coefficient");

    hv1::Polynomial complete = hv1::Polynomial::constant(1.0);
    const hv1::Polynomial affine = hv1::Polynomial::constant(1.0) + b1 + b2 + bK2;
    for (int order = 0; order < 6; ++order) complete = complete * affine;
    require(complete.terms().size() == 84, "degree-six polynomial must contain 84 terms");
    require_close(
        complete.coefficient(2, 2, 2),
        90.0,
        0.0,
        0.0,
        "degree-six multinomial coefficient");
    require((complete - complete).empty(), "polynomial cancellation must remove every term");
    bool rejected_degree_seven = false;
    try {
        static_cast<void>(complete * b1);
    } catch (const std::overflow_error&) {
        rejected_degree_seven = true;
    }
    require(rejected_degree_seven, "degree-seven bias polynomial must be rejected");
}

void test_bias_kernels() {
    const hv1::Vec3 a{0.031, -0.014, 0.071};
    const hv1::Vec3 b{-0.052, 0.023, 0.044};
    const hv1::Vec3 c{0.017, 0.063, -0.028};
    const hv1::KernelTemplate k1 = hv1::pre_reconstruction_kernel({a});
    require_close(k1.regular.coefficient(1, 0, 0), 1.0, 0.0, 0.0, "K1 b1");
    require(k1.matter_f3_tadpoles.empty(), "K1 cannot have a tadpole");

    const hv1::KernelTemplate k2 = hv1::pre_reconstruction_kernel({a, b});
    require_close(k2.regular.coefficient(1, 0, 0), f2(a, b), 1.0e-13, 1.0e-13, "K2 F2");
    require_close(k2.regular.coefficient(0, 1, 0), 0.5, 1.0e-14, 1.0e-14, "K2 D2");
    require_close(
        k2.regular.coefficient(0, 0, 1),
        hv1::tidal_s2(a, b),
        1.0e-13,
        1.0e-13,
        "K2 T2");

    const hv1::KernelTemplate k3 = hv1::pre_reconstruction_kernel({a, b, c});
    const double d3 = (f2(a, b) + f2(a, c) + f2(b, c)) / 3.0;
    const double t3 = 2.0 / 3.0
                      * (hv1::tidal_s2(a, hv1::add(b, c)) * f2(b, c)
                         + hv1::tidal_s2(b, hv1::add(a, c)) * f2(a, c)
                         + hv1::tidal_s2(c, hv1::add(a, b)) * f2(a, b));
    require_close(k3.regular.coefficient(1, 0, 0), f3(a, b, c), 1.0e-12, 1.0e-12, "K3 F3");
    require_close(k3.regular.coefficient(0, 1, 0), d3, 1.0e-12, 1.0e-12, "K3 D3");
    require_close(k3.regular.coefficient(0, 0, 1), t3, 1.0e-12, 1.0e-12, "K3 T3");
    const std::array<hv1::Vec3, 3> vectors3 = {a, b, c};
    std::array<int, 3> permutation3 = {0, 1, 2};
    do {
        require_kernel_close(
            hv1::pre_reconstruction_kernel({
                vectors3[permutation3[0]],
                vectors3[permutation3[1]],
                vectors3[permutation3[2]]}),
            k3,
            2.0e-12,
            "K3 permutation symmetry");
    } while (std::next_permutation(permutation3.begin(), permutation3.end()));

    const hv1::Vec3 q{0.043, -0.027, 0.081};
    const hv1::Vec3 minus_q = hv1::negate(q);
    const hv1::KernelTemplate k4 = hv1::pre_reconstruction_kernel({a, b, q, minus_q});
    const double d4_regular = 0.25 * (f3(a, b, minus_q) + f3(a, b, q))
                              + 1.0 / 6.0
                                    * (f2(a, b) * f2(q, minus_q)
                                       + f2(a, q) * f2(b, minus_q)
                                       + f2(a, minus_q) * f2(b, q));
    const double t4_regular =
        0.5
            * (hv1::tidal_s2(q, sum({a, b, minus_q})) * f3(a, b, minus_q)
               + hv1::tidal_s2(minus_q, sum({a, b, q})) * f3(a, b, q))
        + 1.0 / 3.0
              * (hv1::tidal_s2(hv1::add(a, b), hv1::add(q, minus_q))
                     * f2(a, b) * f2(q, minus_q)
                 + hv1::tidal_s2(hv1::add(a, q), hv1::add(b, minus_q))
                       * f2(a, q) * f2(b, minus_q)
                 + hv1::tidal_s2(hv1::add(a, minus_q), hv1::add(b, q))
                       * f2(a, minus_q) * f2(b, q));
    require_close(
        k4.regular.coefficient(0, 1, 0), d4_regular, 2.0e-12, 2.0e-12, "K4 D4 regular");
    require_close(
        k4.regular.coefficient(0, 0, 1), t4_regular, 2.0e-12, 2.0e-12, "K4 T4 regular");
    const std::vector<hv1::TadpoleTerm> tadpoles = sorted_tadpoles(k4);
    require(tadpoles.size() == 2, "K4 must expose two matter F3 tadpoles");
    const double external_tidal = hv1::tidal_s2(a, b);
    for (const hv1::TadpoleTerm& tadpole : tadpoles) {
        require_close(tadpole.coefficient.coefficient(0, 1, 0), 0.25, 1.0e-13, 1.0e-13, "K4 tadpole D4");
        require_close(
            tadpole.coefficient.coefficient(0, 0, 1),
            0.5 * external_tidal,
            1.0e-12,
            1.0e-12,
            "K4 tadpole T4");
    }

    std::array<hv1::Vec3, 4> vectors = {a, b, q, minus_q};
    std::array<int, 4> permutation = {0, 1, 2, 3};
    const hv1::KernelTemplate reference = k4;
    do {
        require_kernel_close(
            hv1::pre_reconstruction_kernel({
                vectors[permutation[0]], vectors[permutation[1]],
                vectors[permutation[2]], vectors[permutation[3]]}),
            reference,
            2.0e-11,
            "K4 permutation symmetry");
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

void test_reconstruction_kernels() {
    const hv1::Vec3 a{0.026, 0.011, 0.067};
    const hv1::Vec3 b{-0.039, 0.045, 0.031};
    const hv1::Vec3 c{0.054, -0.022, -0.019};
    hv1::ReconstructionConfig config;
    config.enabled = true;
    config.smoothing_radius = 15.0;
    config.bias_recon = 2.7340475186190334;
    config.cell_size = 8.0;

    const hv1::Vec3 output2 = hv1::add(a, b);
    const hv1::Polynomial k1a = hv1::pre_reconstruction_kernel({a}).regular;
    const hv1::Polynomial k1b = hv1::pre_reconstruction_kernel({b}).regular;
    const hv1::Polynomial expected2 = hv1::pre_reconstruction_kernel({a, b}).regular
        + 0.5
              * (hv1::reconstruction_shift_factor(output2, a, config)
                 + hv1::reconstruction_shift_factor(output2, b, config))
              * k1a * k1b;
    require_polynomial_close(
        hv1::reconstructed_kernel({a, b}, config).regular,
        expected2,
        2.0e-13,
        "reconstructed K2 explicit formula");

    const std::array<hv1::Vec3, 3> vectors = {a, b, c};
    const hv1::Vec3 output3 = sum({a, b, c});
    hv1::Polynomial expected3 = hv1::pre_reconstruction_kernel({a, b, c}).regular;
    for (int singleton = 0; singleton < 3; ++singleton) {
        const int first = (singleton + 1) % 3;
        const int second = (singleton + 2) % 3;
        const hv1::Vec3 pair_sum = hv1::add(vectors[first], vectors[second]);
        const hv1::Polynomial k1 = hv1::pre_reconstruction_kernel({vectors[singleton]}).regular;
        const hv1::Polynomial k2 = hv1::pre_reconstruction_kernel(
            {vectors[first], vectors[second]}).regular;
        expected3 += 1.0 / 3.0
                     * (hv1::reconstruction_shift_factor(output3, vectors[singleton], config)
                        + hv1::reconstruction_shift_factor(output3, pair_sum, config))
                     * k1 * k2;
    }
    const double singleton_partition =
        hv1::reconstruction_shift_factor(output3, a, config)
            * hv1::reconstruction_shift_factor(output3, b, config)
        + hv1::reconstruction_shift_factor(output3, a, config)
              * hv1::reconstruction_shift_factor(output3, c, config)
        + hv1::reconstruction_shift_factor(output3, b, config)
              * hv1::reconstruction_shift_factor(output3, c, config);
    expected3 += 1.0 / 6.0 * singleton_partition
                 * hv1::pre_reconstruction_kernel({a}).regular
                 * hv1::pre_reconstruction_kernel({b}).regular
                 * hv1::pre_reconstruction_kernel({c}).regular;
    require_polynomial_close(
        hv1::reconstructed_kernel({a, b, c}, config).regular,
        expected3,
        3.0e-12,
        "reconstructed K3 explicit formula");

    const hv1::Vec3 q{0.041, 0.028, -0.073};
    const std::array<hv1::Vec3, 4> vectors4 = {a, b, q, hv1::negate(q)};
    const hv1::Vec3 output4 = sum({vectors4[0], vectors4[1], vectors4[2], vectors4[3]});
    hv1::KernelTemplate expected4 = hv1::pre_reconstruction_kernel(
        {vectors4[0], vectors4[1], vectors4[2], vectors4[3]});

    for (int singleton = 0; singleton < 4; ++singleton) {
        std::vector<hv1::Vec3> triple;
        for (int index = 0; index < 4; ++index) {
            if (index != singleton) triple.push_back(vectors4[index]);
        }
        const double shift_sum =
            hv1::reconstruction_shift_factor(output4, vectors4[singleton], config)
            + hv1::reconstruction_shift_factor(output4, sum(triple), config);
        expected4 = add_test_kernels(
            std::move(expected4),
            scale_test_kernel(
                multiply_test_kernels(
                    hv1::pre_reconstruction_kernel(triple),
                    hv1::pre_reconstruction_kernel({vectors4[singleton]})),
                0.25 * shift_sum));
    }

    for (int second = 1; second < 4; ++second) {
        const std::array<int, 2> pair = {0, second};
        std::array<int, 2> complement = {};
        int cursor = 0;
        for (int index = 0; index < 4; ++index) {
            if (index != pair[0] && index != pair[1]) complement[cursor++] = index;
        }
        const hv1::Vec3 pair_sum = hv1::add(vectors4[pair[0]], vectors4[pair[1]]);
        const hv1::Vec3 complement_sum =
            hv1::add(vectors4[complement[0]], vectors4[complement[1]]);
        if (hv1::norm(pair_sum) <= 1.0e-12 || hv1::norm(complement_sum) <= 1.0e-12) {
            continue;
        }
        const double shift_sum = hv1::reconstruction_shift_factor(output4, pair_sum, config)
                                 + hv1::reconstruction_shift_factor(
                                       output4, complement_sum, config);
        expected4 = add_test_kernels(
            std::move(expected4),
            scale_test_kernel(
                multiply_test_kernels(
                    hv1::pre_reconstruction_kernel(
                        {vectors4[pair[0]], vectors4[pair[1]]}),
                    hv1::pre_reconstruction_kernel(
                        {vectors4[complement[0]], vectors4[complement[1]]})),
                shift_sum / 6.0));
    }

    for (int left = 0; left < 4; ++left) {
        for (int right = left + 1; right < 4; ++right) {
            std::array<int, 2> singletons = {};
            int cursor = 0;
            for (int index = 0; index < 4; ++index) {
                if (index != left && index != right) singletons[cursor++] = index;
            }
            const hv1::Vec3 pair_sum = hv1::add(vectors4[left], vectors4[right]);
            if (hv1::norm(pair_sum) <= 1.0e-12) continue;
            const double shift_pair =
                hv1::reconstruction_shift_factor(output4, pair_sum, config);
            const double shift_first = hv1::reconstruction_shift_factor(
                output4, vectors4[singletons[0]], config);
            const double shift_second = hv1::reconstruction_shift_factor(
                output4, vectors4[singletons[1]], config);
            hv1::KernelTemplate term = multiply_test_kernels(
                hv1::pre_reconstruction_kernel({vectors4[left], vectors4[right]}),
                hv1::pre_reconstruction_kernel({vectors4[singletons[0]]}));
            term = multiply_test_kernels(
                term,
                hv1::pre_reconstruction_kernel({vectors4[singletons[1]]}));
            expected4 = add_test_kernels(
                std::move(expected4),
                scale_test_kernel(
                    std::move(term),
                    (shift_first * shift_second
                     + shift_pair * (shift_first + shift_second))
                        / 12.0));
        }
    }

    hv1::KernelTemplate four_singletons = hv1::pre_reconstruction_kernel({vectors4[0]});
    std::array<double, 4> singleton_shifts = {};
    for (int index = 0; index < 4; ++index) {
        if (index > 0) {
            four_singletons = multiply_test_kernels(
                four_singletons,
                hv1::pre_reconstruction_kernel({vectors4[index]}));
        }
        singleton_shifts[index] =
            hv1::reconstruction_shift_factor(output4, vectors4[index], config);
    }
    double triple_shift_sum = 0.0;
    for (int density = 0; density < 4; ++density) {
        double product = 1.0;
        for (int index = 0; index < 4; ++index) {
            if (index != density) product *= singleton_shifts[index];
        }
        triple_shift_sum += product;
    }
    expected4 = add_test_kernels(
        std::move(expected4),
        scale_test_kernel(std::move(four_singletons), triple_shift_sum / 24.0));
    require_kernel_close(
        hv1::reconstructed_kernel(
            {vectors4[0], vectors4[1], vectors4[2], vectors4[3]}, config),
        expected4,
        5.0e-11,
        "reconstructed K4 explicit partition formula");
    const hv1::KernelTemplate reconstructed4 = hv1::reconstructed_kernel(
        {vectors4[0], vectors4[1], vectors4[2], vectors4[3]}, config);
    std::array<int, 4> permutation4 = {0, 1, 2, 3};
    do {
        require_kernel_close(
            hv1::reconstructed_kernel({
                vectors4[permutation4[0]],
                vectors4[permutation4[1]],
                vectors4[permutation4[2]],
                vectors4[permutation4[3]]}, config),
            reconstructed4,
            8.0e-11,
            "reconstructed K4 permutation symmetry");
    } while (std::next_permutation(permutation4.begin(), permutation4.end()));

    const hv1::Vec3 window_vector{0.037, -0.052, 0.081};
    const auto sinc = [](double x) { return std::sin(x) / x; };
    const double expected_window =
        std::exp(-0.5 * hv1::dot(window_vector, window_vector) * 15.0 * 15.0)
        * std::pow(sinc(window_vector.x * 4.0), 4)
        * std::pow(sinc(window_vector.y * 4.0), 4)
        * std::pow(sinc(window_vector.z * 4.0), 4);
    require_close(
        hv1::reconstruction_window(window_vector, config),
        expected_window,
        1.0e-14,
        1.0e-14,
        "Gaussian times CIC-paint/read window");

    hv1::ReconstructionConfig infinite = config;
    infinite.smoothing_radius = 1.0e5;
    for (const std::vector<hv1::Vec3>& arguments : std::vector<std::vector<hv1::Vec3>>{
             {a}, {a, b}, {a, b, c}, {a, b, q, hv1::negate(q)}}) {
        require_kernel_close(
            hv1::reconstructed_kernel(arguments, infinite),
            hv1::pre_reconstruction_kernel(arguments),
            2.0e-12,
            "R to infinity reconstruction limit");
    }
}

void test_canonical_triangle() {
    std::array<double, 3> sides = {0.047, 0.063, 0.081};
    std::array<int, 3> permutation = {0, 1, 2};
    const hv1::CanonicalTriangle reference = hv1::canonicalize_triangle(
        hv1::Triangle{sides[0], sides[1], sides[2]});
    do {
        const hv1::CanonicalTriangle actual = hv1::canonicalize_triangle(hv1::Triangle{
            sides[permutation[0]], sides[permutation[1]], sides[permutation[2]]});
        require_close(actual.sides.k1, reference.sides.k1, 0.0, 0.0, "canonical side 1");
        require_close(actual.sides.k2, reference.sides.k2, 0.0, 0.0, "canonical side 2");
        require_close(actual.sides.k3, reference.sides.k3, 0.0, 0.0, "canonical side 3");
        require_close(hv1::norm(sum({actual.k1, actual.k2, actual.k3})), 0.0, 0.0, 1.0e-15, "triangle closure");
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

std::array<double, 8> component_array(const hv1::ComponentValues& result) {
    return {
        result.tree, result.B222, result.B321I, result.B321II,
        result.B411, result.one_loop, result.total, result.stochastic_alpha3_raw};
}

hv1::Vec3 rotate_z(const hv1::Vec3& vector, double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return hv1::Vec3{
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
        vector.z};
}

hv1::Vec3 rotate_y(const hv1::Vec3& vector, double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return hv1::Vec3{
        cosine * vector.x + sine * vector.z,
        vector.y,
        -sine * vector.x + cosine * vector.z};
}

hv1::Vec3 rotate_zyz(
    const hv1::Vec3& vector,
    double alpha,
    double beta,
    double gamma) {
    return rotate_z(rotate_y(rotate_z(vector, gamma), beta), alpha);
}

void test_oriented_core(const PowerSpectrum& power) {
    const hv1::Triangle sides{0.047, 0.063, 0.081};
    const hv1::CanonicalTriangle canonical = hv1::canonicalize_triangle(sides);
    const std::array<hv1::Vec3, 3> vectors = {
        canonical.k1, canonical.k2, canonical.k3};
    std::array<hv1::Vec3, 3> rotated = {};
    for (std::size_t index = 0; index < vectors.size(); ++index) {
        rotated[index] = rotate_zyz(vectors[index], 0.419, 0.731, 1.127);
    }

    hv1::IntegrationConfig config;
    config.qmin = 1.0e-3;
    config.qmax = 0.5;
    config.n_radial = 3;
    config.n_mu = 3;
    config.n_phi = 4;
    config.p13_epsrel = 1.0e-7;
    const hv1::BiasPoint bias{2.7340475186190334, -0.4, -0.3};

    const hv1::ComponentValues from_sides =
        hv1::compute_direct(power, sides, config, bias);
    const hv1::ComponentValues from_vectors =
        hv1::compute_direct_vectors(power, vectors, config, bias);
    const std::array<double, 8> side_values = component_array(from_sides);
    const std::array<double, 8> vector_values = component_array(from_vectors);
    for (std::size_t component = 0; component < side_values.size(); ++component) {
        require_close(
            vector_values[component], side_values[component], 2.0e-13, 1.0e-7,
            "canonical vector core agrees with side-length core");
    }

    const hv1::ComponentValues rotated_pre =
        hv1::compute_direct_vectors(power, rotated, config, bias);
    const std::array<double, 8> rotated_pre_values = component_array(rotated_pre);
    for (std::size_t component = 0; component < side_values.size(); ++component) {
        require_close(
            rotated_pre_values[component], side_values[component], 2.0e-11, 1.0e-5,
            "pre-reconstruction core is rotationally invariant");
    }

    config.reconstruction.enabled = true;
    config.reconstruction.cell_size = 0.0;
    const std::array<double, 8> isotropic_reference = component_array(
        hv1::compute_direct_vectors(power, vectors, config, bias));
    const std::array<double, 8> isotropic_rotated = component_array(
        hv1::compute_direct_vectors(power, rotated, config, bias));
    for (std::size_t component = 0; component < isotropic_reference.size(); ++component) {
        require_close(
            isotropic_rotated[component], isotropic_reference[component],
            3.0e-11, 1.0e-5,
            "Gaussian-only reconstructed core is rotationally invariant");
    }

    config.reconstruction.cell_size = 8.0;
    const std::array<double, 8> cic_reference = component_array(
        hv1::compute_direct_vectors(power, vectors, config, bias));
    const std::array<double, 8> cic_rotated = component_array(
        hv1::compute_direct_vectors(power, rotated, config, bias));
    bool orientation_dependence_detected = false;
    for (std::size_t component = 0; component < 7; ++component) {
        orientation_dependence_detected |=
            relative_difference(cic_reference[component], cic_rotated[component], 1.0)
            > 1.0e-8;
    }
    require(
        orientation_dependence_detected,
        "Cartesian CIC window must retain grid-orientation dependence before B000 projection");

    std::array<hv1::Vec3, 3> open = vectors;
    open[2].x += 1.0e-4;
    bool rejected = false;
    try {
        static_cast<void>(hv1::canonicalize_closed_vectors(open));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "open external vectors must be rejected");
}

void test_shell_geometry() {
    const sv1::ShellBin bin{0.02, 0.04, 0.08, 0.10};
    sv1::ShellQuadratureConfig config;
    config.n_radial = 4;
    config.n_internal_mu = 5;
    config.n_alpha = 6;
    config.n_cos_beta = 4;
    config.n_gamma = 6;
    config.average_grid_orientation = true;
    const std::vector<sv1::ShellNode> nodes = sv1::make_shell_nodes(bin, config);
    require(
        nodes.size() == static_cast<std::size_t>(4 * 4 * 5 * 6 * 4 * 6),
        "shell node count");

    double weight_sum = 0.0;
    double k1_cubed = 0.0;
    double k2_cubed = 0.0;
    double mu = 0.0;
    double mu_squared = 0.0;
    double direction_x2 = 0.0;
    double direction_y2 = 0.0;
    double direction_z2 = 0.0;
    double direction_x4 = 0.0;
    double direction_x2y2 = 0.0;
    double maximum_closure = 0.0;
    for (const sv1::ShellNode& node : nodes) {
        require(node.weight > 0.0, "shell weights are positive");
        weight_sum += node.weight;
        k1_cubed += node.weight * node.k1 * node.k1 * node.k1;
        k2_cubed += node.weight * node.k2 * node.k2 * node.k2;
        mu += node.weight * node.internal_mu;
        mu_squared += node.weight * node.internal_mu * node.internal_mu;
        const hv1::Vec3 unit{
            node.closed_vectors[0].x / node.k1,
            node.closed_vectors[0].y / node.k1,
            node.closed_vectors[0].z / node.k1};
        direction_x2 += node.weight * unit.x * unit.x;
        direction_y2 += node.weight * unit.y * unit.y;
        direction_z2 += node.weight * unit.z * unit.z;
        direction_x4 += node.weight * std::pow(unit.x, 4);
        direction_x2y2 += node.weight * unit.x * unit.x * unit.y * unit.y;
        maximum_closure = std::max(
            maximum_closure,
            hv1::norm(hv1::add(
                hv1::add(node.closed_vectors[0], node.closed_vectors[1]),
                node.closed_vectors[2])));
    }
    require_close(weight_sum, 1.0, 0.0, 2.0e-13, "normalized shell measure");
    require_close(
        k1_cubed,
        0.5 * (std::pow(bin.k1_lower, 3) + std::pow(bin.k1_upper, 3)),
        2.0e-13,
        1.0e-15,
        "k1 cubed radial-volume moment");
    require_close(
        k2_cubed,
        0.5 * (std::pow(bin.k2_lower, 3) + std::pow(bin.k2_upper, 3)),
        2.0e-13,
        1.0e-15,
        "k2 cubed radial-volume moment");
    require_close(mu, 0.0, 0.0, 2.0e-15, "B000 internal-mu monopole mean");
    require_close(mu_squared, 1.0 / 3.0, 2.0e-13, 2.0e-15, "B000 mu squared");
    require_close(direction_x2, 1.0 / 3.0, 2.0e-13, 2.0e-15, "Haar x squared");
    require_close(direction_y2, 1.0 / 3.0, 2.0e-13, 2.0e-15, "Haar y squared");
    require_close(direction_z2, 1.0 / 3.0, 2.0e-13, 2.0e-15, "Haar z squared");
    require_close(direction_x4, 1.0 / 5.0, 2.0e-13, 2.0e-15, "Haar x fourth");
    require_close(
        direction_x2y2, 1.0 / 15.0, 2.0e-13, 2.0e-15,
        "Haar x squared y squared");
    require(maximum_closure < 2.0e-16, "every shell node closes its triangle");
}

void test_fft_lattice_shell_geometry() {
    const sv1::ShellBin bin{
        0.02035714285714286, 0.04107142857142857,
        0.12392857142857142, 0.14464285714285713};
    sv1::ShellQuadratureConfig config;
    config.n_radial = 1;
    config.n_internal_mu = 1;
    config.average_grid_orientation = false;
    config.radial_measure = sv1::ShellRadialMeasure::FftLattice;
    config.fft_box_size = 1000.0;
    config.fft_mesh_size = 256;
    const std::vector<sv1::ShellNode> nodes = sv1::make_shell_nodes(bin, config);
    require(nodes.size() == 1, "one-node FFT-lattice shell rule");
    require_close(nodes[0].weight, 1.0, 0.0, 2.0e-15, "FFT-lattice shell weight");
    require_close(
        nodes[0].k1, 0.03315890968675348, 0.0, 2.0e-15,
        "JAXPower shell-1 mean radius");
    require_close(
        nodes[0].k2, 0.13475574612451321, 0.0, 2.0e-15,
        "JAXPower shell-6 mean radius");
}

std::array<double, 8> shell_component_array(const sv1::ShellComponentValues& result) {
    return {
        result.tree, result.B222, result.B321I, result.B321II,
        result.B411, result.one_loop, result.total, result.stochastic_alpha3_raw};
}

void test_shell_recomposition(const PowerSpectrum& power) {
    const sv1::ShellBin bin{0.0410714285714286, 0.0617857142857143,
                            0.0825, 0.103214285714286};
    hv1::IntegrationConfig loop;
    loop.qmin = 1.0e-3;
    loop.qmax = 0.4;
    loop.n_radial = 2;
    loop.n_mu = 2;
    loop.n_phi = 3;
    loop.p13_epsrel = 1.0e-7;
    sv1::ShellQuadratureConfig shell;
    shell.n_radial = 2;
    shell.n_internal_mu = 3;
    shell.n_alpha = 2;
    shell.n_cos_beta = 2;
    shell.n_gamma = 2;
    const hv1::BiasPoint bias{2.4, -0.6, -0.25};

    const sv1::ShellComponentTemplates pre_templates =
        sv1::compute_shell_templates(power, bin, loop, shell);
    const sv1::ShellComponentValues pre_from_templates =
        sv1::evaluate_shell_templates(pre_templates, bias);
    const sv1::ShellComponentValues pre_direct =
        sv1::compute_shell_direct(power, bin, loop, shell, bias);
    require(!pre_direct.grid_orientation_averaged, "pre shell skips redundant grid rotations");
    const std::array<double, 8> pre_left = shell_component_array(pre_from_templates);
    const std::array<double, 8> pre_right = shell_component_array(pre_direct);
    for (std::size_t component = 0; component < pre_left.size(); ++component) {
        require_close(
            pre_left[component], pre_right[component], 3.0e-11, 1.0e-6,
            "pre shell template recomposition");
    }

    loop.reconstruction.enabled = true;
    const sv1::ShellComponentTemplates post_templates =
        sv1::compute_shell_templates(power, bin, loop, shell);
    const sv1::ShellComponentValues post_from_templates =
        sv1::evaluate_shell_templates(post_templates, bias);
    const sv1::ShellComponentValues post_direct =
        sv1::compute_shell_direct(power, bin, loop, shell, bias);
    require(post_direct.grid_orientation_averaged, "post shell averages Cartesian CIC orientation");
    const std::array<double, 8> post_left = shell_component_array(post_from_templates);
    const std::array<double, 8> post_right = shell_component_array(post_direct);
    for (std::size_t component = 0; component < post_left.size(); ++component) {
        require_close(
            post_left[component], post_right[component], 4.0e-11, 1.0e-6,
            "post shell template recomposition");
    }
}

void test_template_recomposition(const PowerSpectrum& power) {
    hv1::IntegrationConfig config;
    config.qmin = 1.0e-3;
    config.qmax = 2.0;
    config.n_radial = 5;
    config.n_mu = 4;
    config.n_phi = 5;
    config.p13_epsrel = 1.0e-7;
    config.reconstruction.enabled = true;
    const hv1::Triangle triangle{0.047, 0.063, 0.081};
    const hv1::ComponentTemplates templates = hv1::compute_templates(power, triangle, config);

    std::mt19937_64 generator(7319421);
    std::uniform_real_distribution<double> b1_distribution(1.3, 3.4);
    std::uniform_real_distribution<double> other_distribution(-1.5, 1.2);
    for (int sample = 0; sample < 4; ++sample) {
        const hv1::BiasPoint bias{
            b1_distribution(generator),
            other_distribution(generator),
            other_distribution(generator)};
        const hv1::ComponentValues from_templates = hv1::evaluate_templates(templates, bias);
        const hv1::ComponentValues direct = hv1::compute_direct(power, triangle, config, bias);
        const std::array<double, 8> left = component_array(from_templates);
        const std::array<double, 8> right = component_array(direct);
        for (std::size_t component = 0; component < left.size(); ++component) {
            require_close(
                left[component], right[component], 2.0e-11, 1.0e-8,
                "random-bias template recomposition");
        }
    }
}

void test_topology_permutations(const PowerSpectrum& power) {
    hv1::IntegrationConfig config;
    config.qmin = 2.0e-3;
    config.qmax = 0.8;
    config.n_radial = 3;
    config.n_mu = 3;
    config.n_phi = 3;
    config.p13_epsrel = 1.0e-6;
    config.reconstruction.enabled = true;
    const std::array<double, 3> sides = {0.049, 0.067, 0.083};
    const hv1::BiasPoint bias{2.7340475186190334, -0.45, -0.31};
    const hv1::ComponentValues reference = hv1::compute_direct(
        power, hv1::Triangle{sides[0], sides[1], sides[2]}, config, bias);
    const std::array<double, 8> expected = component_array(reference);
    std::array<int, 3> permutation = {0, 1, 2};
    do {
        const hv1::ComponentValues actual = hv1::compute_direct(
            power,
            hv1::Triangle{
                sides[permutation[0]], sides[permutation[1]], sides[permutation[2]]},
            config,
            bias);
        const std::array<double, 8> values = component_array(actual);
        for (std::size_t component = 0; component < values.size(); ++component) {
            require_close(values[component], expected[component], 0.0, 0.0, "topology leg permutation");
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
}

void test_bias_layer_dm_limit(const PowerSpectrum& power) {
    hv1::IntegrationConfig config;
    config.qmin = 1.0e-3;
    config.qmax = 1.0;
    config.n_radial = 24;
    config.n_mu = 32;
    config.n_phi = 32;
    config.p13_epsrel = 1.0e-7;
    config.reconstruction.enabled = false;
    const hv1::Triangle triangle{0.05, 0.065, 0.08};
    const hv1::ComponentValues halo_dm = hv1::compute_direct(
        power, triangle, config, hv1::BiasPoint{1.0, 0.0, 0.0});

    const hv1::CanonicalTriangle canonical = hv1::canonicalize_triangle(triangle);
    const double mu = (canonical.sides.k3 * canonical.sides.k3
                       - canonical.sides.k1 * canonical.sides.k1
                       - canonical.sides.k2 * canonical.sides.k2)
                      / (2.0 * canonical.sides.k1 * canonical.sides.k2);
    marisa_b_native::NativeConfig native_config;
    native_config.qmin = config.qmin;
    native_config.qmax = config.qmax;
    native_config.epsrel = 2.0e-3;
    native_config.p13_epsrel = config.p13_epsrel;
    native_config.epsabs = 1.0e-12;
    const marisa_b_native::ComponentResult native = marisa_b_native::compute_pre_recon_gaussian(
        power,
        marisa_b_native::Triangle{canonical.sides.k1, canonical.sides.k2, mu},
        native_config);
    require_close(halo_dm.tree, native.Btree, 2.0e-12, 1.0e-8, "bias-layer DM tree limit");
    require_close(halo_dm.B321II, native.B321II, 2.0e-10, 1.0e-5, "bias-layer DM B321II limit");
    require_close(halo_dm.B222, native.B222, 3.0e-2, 1.0e-4, "bias-layer DM B222 limit");
    require_close(halo_dm.B321I, native.B321I, 3.0e-2, 1.0e-4, "bias-layer DM B321I limit");
    require_close(halo_dm.B411, native.B411, 4.0e-2, 1.0e-4, "bias-layer DM B411 limit");
    if (report_enabled()) {
        std::cout << "DMREF pre B222 " << halo_dm.B222 << " " << native.B222 << " "
                  << native.B222_stats.abserr << " " << native.B222_stats.neval << " "
                  << halo_dm.loop_nodes << "\n";
        std::cout << "DMREF pre B321I " << halo_dm.B321I << " " << native.B321I << " "
                  << native.B321I_stats.abserr << " " << native.B321I_stats.neval << " "
                  << halo_dm.loop_nodes << "\n";
        std::cout << "DMREF pre B411 " << halo_dm.B411 << " " << native.B411 << " "
                  << native.B411_stats.abserr << " " << native.B411_stats.neval << " "
                  << halo_dm.loop_nodes << "\n";
    }

    hv1::IntegrationConfig post_config = config;
    post_config.qmin = 0.021;
    post_config.qmax = 0.023;
    post_config.n_radial = 12;
    post_config.n_mu = 20;
    post_config.n_phi = 24;
    post_config.reconstruction.enabled = true;
    post_config.reconstruction.smoothing_radius = 15.0;
    post_config.reconstruction.bias_recon = 2.7340475186190334;
    post_config.reconstruction.cell_size = 8.0;
    const hv1::ComponentValues halo_dm_post = hv1::compute_direct(
        power, triangle, post_config, hv1::BiasPoint{1.0, 0.0, 0.0});
    native_config.qmin = post_config.qmin;
    native_config.qmax = post_config.qmax;
    native_config.smoothing_radius = post_config.reconstruction.smoothing_radius;
    native_config.bias_recon = post_config.reconstruction.bias_recon;
    native_config.recon_cellsize = post_config.reconstruction.cell_size;
    native_config.recon_cic_window_power = 4;
    native_config.epsrel = 2.0e-3;
    const marisa_b_native::ComponentResult native_post =
        marisa_b_native::compute_post_recon_gaussian(
            power,
            marisa_b_native::Triangle{canonical.sides.k1, canonical.sides.k2, mu},
            native_config);
    require_close(
        halo_dm_post.tree, native_post.Btree, 2.0e-12, 1.0e-8,
        "bias-layer post-reconstruction DM tree limit");
    require_close(
        halo_dm_post.B321II, native_post.B321II, 3.0e-2, 1.0e-4,
        "bias-layer post-reconstruction DM B321II limit");
    require_close(
        halo_dm_post.B222, native_post.B222, 5.0e-2, 1.0e-4,
        "bias-layer post-reconstruction DM B222 limit");
    require_close(
        halo_dm_post.B321I, native_post.B321I, 5.0e-2, 1.0e-4,
        "bias-layer post-reconstruction DM B321I limit");
    require_close(
        halo_dm_post.B411, native_post.B411, 5.0e-2, 1.0e-4,
        "bias-layer post-reconstruction DM B411 limit");
    if (report_enabled()) {
        std::cout << "DMREF post B222 " << halo_dm_post.B222 << " " << native_post.B222
                  << " " << native_post.B222_stats.abserr << " "
                  << native_post.B222_stats.neval << " " << halo_dm_post.loop_nodes << "\n";
        std::cout << "DMREF post B321I " << halo_dm_post.B321I << " "
                  << native_post.B321I << " " << native_post.B321I_stats.abserr << " "
                  << native_post.B321I_stats.neval << " " << halo_dm_post.loop_nodes << "\n";
        std::cout << "DMREF post B411 " << halo_dm_post.B411 << " " << native_post.B411
                  << " " << native_post.B411_stats.abserr << " "
                  << native_post.B411_stats.neval << " " << halo_dm_post.loop_nodes << "\n";
    }
}

void test_grid_convergence(const PowerSpectrum& power) {
    const hv1::Triangle triangle{0.047, 0.063, 0.081};
    const hv1::BiasPoint bias{2.7340475186190334, -0.4, -0.3};
    auto make_config = [](int nradial, int nmu, int nphi) {
        hv1::IntegrationConfig config;
        config.qmin = 1.0e-3;
        config.qmax = 30.0;
        config.n_radial = nradial;
        config.n_mu = nmu;
        config.n_phi = nphi;
        config.p13_epsrel = 1.0e-7;
        config.reconstruction.enabled = true;
        return config;
    };
    const hv1::ComponentValues medium = hv1::compute_direct(
        power, triangle, make_config(24, 17, 24), bias);
    const hv1::ComponentValues fine = hv1::compute_direct(
        power, triangle, make_config(32, 23, 32), bias);
    const std::array<double, 8> medium_values = component_array(medium);
    const std::array<double, 8> fine_values = component_array(fine);
    const bool report = report_enabled();
    if (report) {
        std::cout << "NODES grid " << medium.loop_nodes << " " << fine.loop_nodes << "\n";
    }
    double maximum_grid_difference = 0.0;
    for (std::size_t component = 0; component < 7; ++component) {
        const double difference =
            relative_difference(medium_values[component], fine_values[component], 1.0);
        maximum_grid_difference = std::max(maximum_grid_difference, difference);
        if (report) {
            std::cout << "GRID " << component << " " << medium_values[component]
                      << " " << fine_values[component] << " " << difference << "\n";
        }
    }
    require(
        maximum_grid_difference < 0.01,
        "fixed log-radial cubature did not meet the one-percent threshold");

    hv1::IntegrationConfig logarithmic_config = make_config(18, 13, 18);
    logarithmic_config.qmax = 1.0;
    hv1::IntegrationConfig linear_config = logarithmic_config;
    linear_config.radial_coordinate = hv1::RadialCoordinate::Linear;
    const hv1::ComponentValues logarithmic =
        hv1::compute_direct(power, triangle, logarithmic_config, bias);
    const hv1::ComponentValues linear =
        hv1::compute_direct(power, triangle, linear_config, bias);
    const std::array<double, 8> logarithmic_values = component_array(logarithmic);
    const std::array<double, 8> linear_values = component_array(linear);
    if (report) {
        std::cout << "NODES coordinate " << logarithmic.loop_nodes << " "
                  << linear.loop_nodes << "\n";
    }
    for (std::size_t component = 0; component < 7; ++component) {
        const double difference =
            relative_difference(logarithmic_values[component], linear_values[component], 1.0);
        require(difference < 0.01, "logarithmic and linear radial coordinates disagree");
        if (report) {
            std::cout << "COORD " << component << " " << logarithmic_values[component]
                      << " " << linear_values[component] << " " << difference << "\n";
        }
    }

    hv1::IntegrationConfig loose_p13 = make_config(8, 6, 8);
    loose_p13.qmax = 1.0;
    loose_p13.p13_epsrel = 1.0e-4;
    hv1::IntegrationConfig tight_p13 = loose_p13;
    tight_p13.p13_epsrel = 1.0e-8;
    const std::array<double, 8> loose_p13_values = component_array(
        hv1::compute_direct(power, triangle, loose_p13, bias));
    const std::array<double, 8> tight_p13_values = component_array(
        hv1::compute_direct(power, triangle, tight_p13, bias));
    for (std::size_t component : std::array<std::size_t, 3>{3, 4, 6}) {
        const double difference = relative_difference(
            loose_p13_values[component], tight_p13_values[component], 1.0);
        require(difference < 1.0e-6, "P13 tolerance refinement changed the result");
        if (report) {
            std::cout << "P13 " << component << " " << loose_p13_values[component]
                      << " " << tight_p13_values[component] << " " << difference << "\n";
        }
    }

    if (report) {
        for (double qmax : std::array<double, 6>{1.0, 2.0, 4.0, 10.0, 20.0, 30.0}) {
            hv1::IntegrationConfig scan = make_config(10, 8, 10);
            scan.qmax = qmax;
            const hv1::ComponentValues value = hv1::compute_direct(power, triangle, scan, bias);
            std::cout << "QMAX " << qmax << " " << value.loop_nodes << " " << value.tree
                      << " " << value.B222
                      << " " << value.B321I << " " << value.B321II << " " << value.B411
                      << " " << value.total << "\n";
        }
    }
}

void test_stochastic_basis(const PowerSpectrum& power) {
    hv1::IntegrationConfig config;
    config.qmin = 1.0e-3;
    config.qmax = 0.5;
    config.n_radial = 2;
    config.n_mu = 2;
    config.n_phi = 2;
    const hv1::Triangle triangle{0.05, 0.06, 0.075};
    const hv1::BiasPoint bias{2.4, -0.7, -0.2};
    const hv1::ComponentValues result = hv1::compute_direct(power, triangle, config, bias);
    const hv1::CanonicalTriangle canonical = hv1::canonicalize_triangle(triangle);
    const double expected = bias.b1 * bias.b1
                            * (power(canonical.sides.k1)
                               + power(canonical.sides.k2)
                               + power(canonical.sides.k3));
    require_close(result.stochastic_alpha3_raw, expected, 1.0e-14, 1.0e-10, "raw alpha3 basis");
    require_close(result.stochastic_alpha4_raw, 1.0, 0.0, 0.0, "raw alpha4 basis");
    const double total = hv1::add_residual_stochastic(
        12.0, expected, 1.0, 0.3, -0.2, 4.0e-4);
    require_close(
        total,
        12.0 + 0.3 * expected / 4.0e-4 - 0.2 / (4.0e-4 * 4.0e-4),
        1.0e-14,
        1.0e-8,
        "fit-layer stochastic normalization");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: test_halo_v1 POWER_TABLE\n";
            return 2;
        }
        TabulatedPower power(argv[1]);
        if (report_enabled()) {
            std::cout << std::setprecision(17);
            std::cout << "IDENTITY " << kBuildIdentity << "\n";
        }
        test_polynomial();
        test_bias_kernels();
        test_reconstruction_kernels();
        test_canonical_triangle();
        test_oriented_core(power);
        test_shell_geometry();
        test_fft_lattice_shell_geometry();
        test_shell_recomposition(power);
        test_template_recomposition(power);
        test_topology_permutations(power);
        test_bias_layer_dm_limit(power);
        test_grid_convergence(power);
        test_stochastic_basis(power);
        std::cout << "PASS halo_v1 checks=" << checks << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL halo_v1 after checks=" << checks << ": " << error.what() << "\n";
        return 1;
    }
}
