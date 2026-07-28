#include "lattice_shell_rule.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_legendre.h>

namespace marisa_b_eft_v2 {
namespace {

namespace hv1 = marisa_b_halo_v1;
namespace shell = marisa_b_shell_v1;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

struct LatticeMode {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float radius = 0.0F;
};

struct HarmonicBlock {
    int degree = 0;
    int order_count = 0;
    std::vector<std::complex<long double>> coefficients;
};

struct LatticeShellData {
    double lower = 0.0;
    double upper = 0.0;
    int radial_order = 0;
    int angular_order = 0;
    std::uint64_t full_count = 0;
    std::vector<LatticeMode> modes;
    QuadratureRule radial;
    std::vector<double> lagrange;
    std::vector<long double> closing_gram;
    std::vector<HarmonicBlock> harmonics;
};

struct ClosingMoments {
    std::vector<long double> gram;
    std::uint64_t pair_count = 0;
};

using ShellCacheKey = std::tuple<
    int, int, double, double, double, int>;

double legendre(int degree, double x) {
    if (degree == 0) return 1.0;
    if (degree == 1) return x;
    double previous = 1.0;
    double current = x;
    for (int index = 2; index <= degree; ++index) {
        const double next =
            ((2.0 * index - 1.0) * x * current
             - (index - 1.0) * previous)
            / index;
        previous = current;
        current = next;
    }
    return current;
}

QuadratureRule gauss_legendre(
    int count,
    double lower,
    double upper) {
    if (count < 1 || !(upper > lower)) {
        throw std::invalid_argument(
            "invalid Gauss-Legendre interval or order");
    }
    QuadratureRule rule;
    rule.nodes.resize(static_cast<std::size_t>(count));
    rule.weights.resize(static_cast<std::size_t>(count));
    const double midpoint = 0.5 * (upper + lower);
    const double half_width = 0.5 * (upper - lower);
    const int roots = (count + 1) / 2;
    for (int root = 0; root < roots; ++root) {
        double value =
            std::cos(kPi * (root + 0.75) / (count + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double previous = 1.0;
            double current = value;
            for (int degree = 2; degree <= count; ++degree) {
                const double next =
                    ((2.0 * degree - 1.0) * value * current
                     - (degree - 1.0) * previous)
                    / degree;
                previous = current;
                current = next;
            }
            derivative =
                count * (value * current - previous)
                / (value * value - 1.0);
            const double updated = value - current / derivative;
            if (std::fabs(updated - value) <= 4.0e-16) {
                value = updated;
                break;
            }
            value = updated;
        }
        double previous = 1.0;
        double current = value;
        for (int degree = 2; degree <= count; ++degree) {
            const double next =
                ((2.0 * degree - 1.0) * value * current
                 - (degree - 1.0) * previous)
                / degree;
            previous = current;
            current = next;
        }
        derivative =
            count * (value * current - previous)
            / (value * value - 1.0);
        const double weight =
            2.0 / ((1.0 - value * value)
                   * derivative * derivative);
        const int left = root;
        const int right = count - 1 - root;
        rule.nodes[static_cast<std::size_t>(left)] =
            midpoint - half_width * value;
        rule.nodes[static_cast<std::size_t>(right)] =
            midpoint + half_width * value;
        rule.weights[static_cast<std::size_t>(left)] =
            half_width * weight;
        rule.weights[static_cast<std::size_t>(right)] =
            half_width * weight;
    }
    return rule;
}

QuadratureRule stieltjes(
    const std::vector<double>& support,
    const std::vector<double>& probabilities,
    int count) {
    if (support.size() != probabilities.size()
        || support.size() < static_cast<std::size_t>(count)
        || count < 1) {
        throw std::invalid_argument("invalid Stieltjes input");
    }
    std::vector<double> previous(support.size(), 0.0);
    std::vector<double> current(support.size(), 1.0);
    std::vector<double> diagonal(static_cast<std::size_t>(count));
    std::vector<double> off_diagonal;
    off_diagonal.reserve(static_cast<std::size_t>(count - 1));
    double previous_beta = 0.0;
    for (int degree = 0; degree < count; ++degree) {
        double alpha = 0.0;
        for (std::size_t index = 0; index < support.size(); ++index) {
            alpha += probabilities[index] * support[index]
                     * current[index] * current[index];
        }
        diagonal[static_cast<std::size_t>(degree)] = alpha;
        if (degree + 1 == count) break;
        std::vector<double> residual(support.size());
        double beta_squared = 0.0;
        for (std::size_t index = 0; index < support.size(); ++index) {
            residual[index] =
                (support[index] - alpha) * current[index]
                - previous_beta * previous[index];
            beta_squared += probabilities[index]
                            * residual[index] * residual[index];
        }
        const double beta = std::sqrt(beta_squared);
        if (!(beta > 0.0) || !std::isfinite(beta)) {
            throw std::runtime_error(
                "FFT-lattice Stieltjes recurrence failed");
        }
        off_diagonal.push_back(beta);
        previous.swap(current);
        for (std::size_t index = 0; index < support.size(); ++index) {
            current[index] = residual[index] / beta;
        }
        previous_beta = beta;
    }

    gsl_matrix* jacobi = gsl_matrix_calloc(count, count);
    gsl_vector* eigenvalues = gsl_vector_alloc(count);
    gsl_matrix* eigenvectors = gsl_matrix_alloc(count, count);
    gsl_eigen_symmv_workspace* workspace =
        gsl_eigen_symmv_alloc(count);
    if (!jacobi || !eigenvalues || !eigenvectors || !workspace) {
        if (workspace) gsl_eigen_symmv_free(workspace);
        if (eigenvectors) gsl_matrix_free(eigenvectors);
        if (eigenvalues) gsl_vector_free(eigenvalues);
        if (jacobi) gsl_matrix_free(jacobi);
        throw std::bad_alloc();
    }
    for (int index = 0; index < count; ++index) {
        gsl_matrix_set(
            jacobi, index, index,
            diagonal[static_cast<std::size_t>(index)]);
        if (index + 1 < count) {
            const double beta =
                off_diagonal[static_cast<std::size_t>(index)];
            gsl_matrix_set(jacobi, index, index + 1, beta);
            gsl_matrix_set(jacobi, index + 1, index, beta);
        }
    }
    const int status =
        gsl_eigen_symmv(jacobi, eigenvalues, eigenvectors, workspace);
    gsl_eigen_symmv_free(workspace);
    gsl_matrix_free(jacobi);
    if (status != GSL_SUCCESS) {
        gsl_matrix_free(eigenvectors);
        gsl_vector_free(eigenvalues);
        throw std::runtime_error(
            std::string("FFT-lattice eigensolver failed: ")
            + gsl_strerror(status));
    }
    gsl_eigen_symmv_sort(
        eigenvalues, eigenvectors, GSL_EIGEN_SORT_VAL_ASC);
    QuadratureRule result;
    result.nodes.resize(static_cast<std::size_t>(count));
    result.weights.resize(static_cast<std::size_t>(count));
    double total_weight = 0.0;
    for (int index = 0; index < count; ++index) {
        result.nodes[static_cast<std::size_t>(index)] =
            gsl_vector_get(eigenvalues, index);
        const double first_component =
            gsl_matrix_get(eigenvectors, 0, index);
        result.weights[static_cast<std::size_t>(index)] =
            first_component * first_component;
        total_weight += first_component * first_component;
    }
    gsl_matrix_free(eigenvectors);
    gsl_vector_free(eigenvalues);
    for (double& weight : result.weights) weight /= total_weight;
    return result;
}

std::vector<double> lagrange_values(
    double value,
    const std::vector<double>& nodes) {
    std::vector<double> result(nodes.size(), 1.0);
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        for (std::size_t other = 0; other < nodes.size(); ++other) {
            if (index == other) continue;
            result[index] *=
                (value - nodes[other])
                / (nodes[index] - nodes[other]);
        }
    }
    return result;
}

std::shared_ptr<const LatticeShellData> build_shell_data(
    double lower_bound,
    double upper_bound,
    const ExactLatticeShellRuleConfig& config) {
    if (!(lower_bound >= 0.0 && upper_bound > lower_bound)
        || config.radial_order < 1 || config.angular_order < 1
        || !(config.fft_box_size > 0.0)
        || config.fft_mesh_size < 2) {
        throw std::invalid_argument(
            "invalid exact FFT-lattice shell configuration");
    }
    const ShellCacheKey key{
        config.radial_order,
        config.angular_order,
        lower_bound,
        upper_bound,
        config.fft_box_size,
        config.fft_mesh_size};
    static std::map<
        ShellCacheKey,
        std::shared_ptr<const LatticeShellData>> cache;
    static std::mutex cache_mutex;
    {
        const std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) return found->second;
    }

    auto result = std::make_shared<LatticeShellData>();
    result->lower = lower_bound;
    result->upper = upper_bound;
    result->radial_order = config.radial_order;
    result->angular_order = config.angular_order;
    const float fundamental = static_cast<float>(
        kTwoPi / static_cast<float>(config.fft_box_size));
    const float lower = static_cast<float>(lower_bound);
    const float upper = static_cast<float>(upper_bound);
    const int negative_limit = -config.fft_mesh_size / 2;
    const int positive_limit = (config.fft_mesh_size - 1) / 2;
    const int coordinate_limit =
        static_cast<int>(std::ceil(upper / fundamental)) + 1;
    const int minimum_mode =
        std::max(negative_limit, -coordinate_limit);
    const int maximum_mode =
        std::min(positive_limit, coordinate_limit);

    std::map<float, std::uint64_t> multiplicities;
    for (int nx = minimum_mode; nx <= maximum_mode; ++nx) {
        const float kx = static_cast<float>(nx) * fundamental;
        const float kx_squared = kx * kx;
        for (int ny = minimum_mode; ny <= maximum_mode; ++ny) {
            const float ky = static_cast<float>(ny) * fundamental;
            const float transverse_squared =
                kx_squared + ky * ky;
            for (int nz = minimum_mode; nz <= maximum_mode; ++nz) {
                const float kz = static_cast<float>(nz) * fundamental;
                const float radius =
                    std::sqrt(transverse_squared + kz * kz);
                if (radius >= lower && radius < upper) {
                    ++result->full_count;
                    if (radius > 0.0F) {
                        ++multiplicities[radius];
                        result->modes.push_back(
                            LatticeMode{nx, ny, nz, radius});
                    }
                }
            }
        }
    }
    if (multiplicities.size()
        < static_cast<std::size_t>(config.radial_order)) {
        throw std::invalid_argument(
            "FFT shell has fewer nonzero unique radii "
            "than the exact cubature order");
    }
    std::uint64_t nonzero_count = 0;
    for (const auto& item : multiplicities) {
        nonzero_count += item.second;
    }
    std::vector<double> support;
    std::vector<double> probabilities;
    support.reserve(multiplicities.size());
    probabilities.reserve(multiplicities.size());
    for (const auto& item : multiplicities) {
        support.push_back(static_cast<double>(item.first));
        probabilities.push_back(
            static_cast<double>(item.second)
            / static_cast<double>(nonzero_count));
    }
    result->radial =
        stieltjes(support, probabilities, config.radial_order);

    const std::size_t radial =
        static_cast<std::size_t>(config.radial_order);
    result->lagrange.resize(result->modes.size() * radial);
    for (std::size_t mode_index = 0;
         mode_index < result->modes.size();
         ++mode_index) {
        const std::vector<double> values =
            lagrange_values(
                static_cast<double>(
                    result->modes[mode_index].radius),
                result->radial.nodes);
        for (std::size_t radial_index = 0;
             radial_index < radial;
             ++radial_index) {
            result->lagrange[
                mode_index * radial + radial_index] =
                values[radial_index];
        }
    }
    result->closing_gram.assign(radial * radial, 0.0L);
    for (std::size_t mode_index = 0;
         mode_index < result->modes.size();
         ++mode_index) {
        for (std::size_t first = 0; first < radial; ++first) {
            for (std::size_t second = 0; second < radial; ++second) {
                result->closing_gram[first * radial + second] +=
                    static_cast<long double>(
                        result->lagrange[
                            mode_index * radial + first])
                    * static_cast<long double>(
                        result->lagrange[
                            mode_index * radial + second]);
            }
        }
    }

    result->harmonics.reserve(
        static_cast<std::size_t>(
            (config.angular_order + 1) / 2));
    for (int degree = 0;
         degree < config.angular_order;
         degree += 2) {
        HarmonicBlock block;
        block.degree = degree;
        block.order_count = degree / 4 + 1;
        block.coefficients.assign(
            radial
                * static_cast<std::size_t>(block.order_count),
            std::complex<long double>(0.0L, 0.0L));
        for (std::size_t mode_index = 0;
             mode_index < result->modes.size();
             ++mode_index) {
            const LatticeMode& mode = result->modes[mode_index];
            const double integer_radius = std::sqrt(
                static_cast<double>(mode.nx * mode.nx)
                + static_cast<double>(mode.ny * mode.ny)
                + static_cast<double>(mode.nz * mode.nz));
            const double cosine_theta = std::clamp(
                static_cast<double>(mode.nz) / integer_radius,
                -1.0, 1.0);
            const double phi = std::atan2(
                static_cast<double>(mode.ny),
                static_cast<double>(mode.nx));
            for (int order_index = 0;
                 order_index < block.order_count;
                 ++order_index) {
                const int order = 4 * order_index;
                const double normalized_legendre =
                    gsl_sf_legendre_sphPlm(
                        degree, order, cosine_theta);
                const std::complex<long double> harmonic =
                    static_cast<long double>(normalized_legendre)
                    * std::complex<long double>(
                        std::cos(order * phi),
                        std::sin(order * phi));
                for (std::size_t radial_index = 0;
                     radial_index < radial;
                     ++radial_index) {
                    block.coefficients[
                        radial_index
                            * static_cast<std::size_t>(
                                block.order_count)
                        + static_cast<std::size_t>(order_index)]
                        += static_cast<long double>(
                               result->lagrange[
                                   mode_index * radial
                                   + radial_index])
                           * harmonic;
                }
            }
        }
        result->harmonics.push_back(std::move(block));
    }
    {
        const std::lock_guard<std::mutex> lock(cache_mutex);
        const auto inserted = cache.emplace(key, result);
        if (!inserted.second) return inserted.first->second;
    }
    return result;
}

ClosingMoments cross_closing_moments(
    const LatticeShellData& first,
    const LatticeShellData& second) {
    const std::size_t radial_first =
        first.radial.nodes.size();
    const std::size_t radial_second =
        second.radial.nodes.size();
    ClosingMoments result;
    result.gram.assign(
        radial_first * radial_second, 0.0L);
    if (first.upper <= second.lower
        || second.upper <= first.lower) {
        return result;
    }
    if (&first == &second) {
        result.gram = first.closing_gram;
        result.pair_count =
            static_cast<std::uint64_t>(first.modes.size());
        return result;
    }
    std::map<std::tuple<int, int, int>, std::size_t> second_modes;
    for (std::size_t index = 0; index < second.modes.size(); ++index) {
        const LatticeMode& mode = second.modes[index];
        second_modes.emplace(
            std::make_tuple(mode.nx, mode.ny, mode.nz),
            index);
    }
    for (std::size_t left_index = 0;
         left_index < first.modes.size();
         ++left_index) {
        const LatticeMode& left = first.modes[left_index];
        const auto found = second_modes.find(
            std::make_tuple(-left.nx, -left.ny, -left.nz));
        if (found == second_modes.end()) continue;
        ++result.pair_count;
        const std::size_t right_index = found->second;
        for (std::size_t first_index = 0;
             first_index < radial_first;
             ++first_index) {
            for (std::size_t second_index = 0;
                 second_index < radial_second;
                 ++second_index) {
                result.gram[
                    first_index * radial_second
                    + second_index]
                    += static_cast<long double>(
                           first.lagrange[
                               left_index * radial_first
                               + first_index])
                       * static_cast<long double>(
                           second.lagrange[
                               right_index * radial_second
                               + second_index]);
            }
        }
    }
    return result;
}

std::vector<double> exact_legendre_moments(
    const LatticeShellData& first,
    const LatticeShellData& second,
    int angular_order,
    const ClosingMoments& closing) {
    const std::size_t radial_first =
        first.radial.nodes.size();
    const std::size_t radial_second =
        second.radial.nodes.size();
    std::vector<double> result(
        radial_first * radial_second
            * static_cast<std::size_t>(angular_order),
        0.0);
    const long double normalization =
        static_cast<long double>(first.full_count)
        * static_cast<long double>(second.full_count);
    for (std::size_t first_radial = 0;
         first_radial < radial_first;
         ++first_radial) {
        for (std::size_t second_radial = 0;
             second_radial < radial_second;
             ++second_radial) {
            for (int degree = 0;
                 degree < angular_order;
                 ++degree) {
                long double pair_sum = 0.0L;
                if (degree % 2 == 0) {
                    const HarmonicBlock& left =
                        first.harmonics.at(
                            static_cast<std::size_t>(
                                degree / 2));
                    const HarmonicBlock& right =
                        second.harmonics.at(
                            static_cast<std::size_t>(
                                degree / 2));
                    if (left.degree != degree
                        || right.degree != degree
                        || left.order_count
                            != right.order_count) {
                        throw std::logic_error(
                            "inconsistent exact-lattice "
                            "spherical-harmonic cache");
                    }
                    for (int order_index = 0;
                         order_index < left.order_count;
                         ++order_index) {
                        const auto product =
                            left.coefficients[
                                first_radial
                                    * static_cast<std::size_t>(
                                        left.order_count)
                                + static_cast<std::size_t>(
                                    order_index)]
                            * std::conj(
                                right.coefficients[
                                    second_radial
                                        * static_cast<std::size_t>(
                                            right.order_count)
                                    + static_cast<std::size_t>(
                                        order_index)]);
                        pair_sum += order_index == 0
                            ? std::real(product)
                            : 2.0L * std::real(product);
                    }
                    pair_sum *=
                        static_cast<long double>(4.0 * kPi)
                        / static_cast<long double>(
                            2 * degree + 1);
                }
                long double closing_sum =
                    closing.gram[
                        first_radial * radial_second
                        + second_radial];
                if (degree % 2 != 0) {
                    closing_sum = -closing_sum;
                }
                result[
                    (first_radial * radial_second
                     + second_radial)
                        * static_cast<std::size_t>(
                            angular_order)
                    + static_cast<std::size_t>(degree)]
                    = static_cast<double>(
                        (pair_sum - closing_sum)
                        / normalization);
            }
        }
    }
    return result;
}

shell::ShellNode make_node(
    double k1,
    double k2,
    double mu,
    double weight) {
    const double sine =
        std::sqrt(std::max(0.0, 1.0 - mu * mu));
    shell::ShellNode node;
    node.closed_vectors = {{
        hv1::Vec3{0.0, 0.0, k1},
        hv1::Vec3{k2 * sine, 0.0, k2 * mu},
        hv1::Vec3{-k2 * sine, 0.0, -k1 - k2 * mu}}};
    node.k1 = k1;
    node.k2 = k2;
    node.internal_mu = mu;
    node.weight = weight;
    return node;
}

hv1::Vec3 rotate_zyz(
    const hv1::Vec3& value,
    double alpha,
    double beta,
    double gamma) {
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    const double cg = std::cos(gamma);
    const double sg = std::sin(gamma);

    const double x1 = cg * value.x - sg * value.y;
    const double y1 = sg * value.x + cg * value.y;
    const double z1 = value.z;
    const double x2 = cb * x1 + sb * z1;
    const double y2 = y1;
    const double z2 = -sb * x1 + cb * z1;
    return hv1::Vec3{
        ca * x2 - sa * y2,
        sa * x2 + ca * y2,
        z2};
}

void accumulate_weight_diagnostics(
    const std::vector<shell::ShellNode>& nodes,
    double& total_weight,
    double& total_variation,
    double& maximum_absolute_weight) {
    total_weight = 0.0;
    total_variation = 0.0;
    maximum_absolute_weight = 0.0;
    for (const shell::ShellNode& node : nodes) {
        if (!std::isfinite(node.weight)) {
            throw std::runtime_error(
                "exact-lattice shell produced a non-finite weight");
        }
        total_weight += node.weight;
        total_variation += std::fabs(node.weight);
        maximum_absolute_weight =
            std::max(
                maximum_absolute_weight,
                std::fabs(node.weight));
    }
}

QuadratureRule continuum_k3_rule(
    double first,
    double second,
    int count) {
    if (!(first > 0.0 && second > 0.0) || count < 1) {
        throw std::invalid_argument(
            "invalid multilevel continuum-k3 rule");
    }
    const int base_count = std::max(160, 5 * count);
    const QuadratureRule base =
        gauss_legendre(base_count, -1.0, 1.0);
    const double lower = std::fabs(first - second);
    const double upper = first + second;
    std::vector<double> support(
        static_cast<std::size_t>(base_count));
    std::vector<double> probabilities(
        static_cast<std::size_t>(base_count));
    double total_probability = 0.0;
    for (int index = 0; index < base_count; ++index) {
        const double closing =
            0.5 * (upper - lower)
                * base.nodes[static_cast<std::size_t>(index)]
            + 0.5 * (upper + lower);
        support[static_cast<std::size_t>(index)] = closing;
        probabilities[static_cast<std::size_t>(index)] =
            0.25 * (upper - lower)
            * base.weights[static_cast<std::size_t>(index)]
            * closing / (first * second);
        total_probability +=
            probabilities[static_cast<std::size_t>(index)];
    }
    for (double& value : probabilities) {
        value /= total_probability;
    }
    return stieltjes(support, probabilities, count);
}

std::vector<double> barycentric_basis(
    const std::vector<double>& nodes,
    double value) {
    if (nodes.empty()) {
        throw std::invalid_argument(
            "empty multilevel interpolation rule");
    }
    if (nodes.size() == 1) return {1.0};
    const auto limits =
        std::minmax_element(nodes.begin(), nodes.end());
    const double midpoint =
        0.5 * (*limits.first + *limits.second);
    const double half_width =
        0.5 * (*limits.second - *limits.first);
    if (!(half_width > 0.0)) {
        throw std::invalid_argument(
            "degenerate multilevel interpolation nodes");
    }
    std::vector<long double> scaled(nodes.size());
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        scaled[index] =
            (static_cast<long double>(nodes[index]) - midpoint)
            / half_width;
    }
    const long double scaled_value =
        (static_cast<long double>(value) - midpoint)
        / half_width;
    std::vector<long double> barycentric(nodes.size(), 1.0L);
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (std::fabs(scaled_value - scaled[index])
            < 1.0e-18L) {
            std::vector<double> exact(nodes.size(), 0.0);
            exact[index] = 1.0;
            return exact;
        }
        for (std::size_t other = 0;
             other < nodes.size();
             ++other) {
            if (index == other) continue;
            barycentric[index] /=
                scaled[index] - scaled[other];
        }
    }
    long double denominator = 0.0L;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        barycentric[index] /=
            scaled_value - scaled[index];
        denominator += barycentric[index];
    }
    if (denominator == 0.0L
        || !std::isfinite(denominator)) {
        throw std::runtime_error(
            "multilevel barycentric denominator failed");
    }
    std::vector<double> result(nodes.size());
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        result[index] =
            static_cast<double>(
                barycentric[index] / denominator);
    }
    return result;
}

}  // namespace

ExactLatticeShellRule make_exact_lattice_shell_rule(
    const shell::ShellBin& bin,
    const ExactLatticeShellRuleConfig& config) {
    const auto first = build_shell_data(
        bin.k1_lower, bin.k1_upper, config);
    const auto second = build_shell_data(
        bin.k2_lower, bin.k2_upper, config);
    const ClosingMoments closing =
        cross_closing_moments(*first, *second);
    const std::vector<double> moments =
        exact_legendre_moments(
            *first, *second, config.angular_order, closing);
    const QuadratureRule angular =
        gauss_legendre(config.angular_order, -1.0, 1.0);
    const std::size_t radial_first =
        first->radial.nodes.size();
    const std::size_t radial_second =
        second->radial.nodes.size();

    ExactLatticeShellRule result;
    result.bin = bin;
    result.config = config;
    result.nodes.reserve(
        radial_first * radial_second
        * static_cast<std::size_t>(config.angular_order));
    for (std::size_t first_radial = 0;
         first_radial < radial_first;
         ++first_radial) {
        for (std::size_t second_radial = 0;
             second_radial < radial_second;
             ++second_radial) {
            for (int angular_index = 0;
                 angular_index < config.angular_order;
                 ++angular_index) {
                const double mu =
                    angular.nodes[
                        static_cast<std::size_t>(
                            angular_index)];
                long double expansion = 0.0L;
                for (int degree = 0;
                     degree < config.angular_order;
                     ++degree) {
                    const double moment =
                        moments[
                            (first_radial * radial_second
                             + second_radial)
                                * static_cast<std::size_t>(
                                    config.angular_order)
                            + static_cast<std::size_t>(degree)];
                    expansion +=
                        static_cast<long double>(
                            0.5 * (2 * degree + 1) * moment)
                        * static_cast<long double>(
                            legendre(degree, mu));
                }
                const double weight =
                    angular.weights[
                        static_cast<std::size_t>(
                            angular_index)]
                    * static_cast<double>(expansion);
                result.nodes.push_back(make_node(
                    first->radial.nodes[first_radial],
                    second->radial.nodes[second_radial],
                    mu,
                    weight));
            }
        }
    }
    result.diagnostics.first_full_modes =
        first->full_count;
    result.diagnostics.second_full_modes =
        second->full_count;
    result.diagnostics.first_nonzero_modes =
        static_cast<std::uint64_t>(first->modes.size());
    result.diagnostics.second_nonzero_modes =
        static_cast<std::uint64_t>(second->modes.size());
    const std::uint64_t full_pairs =
        first->full_count * second->full_count;
    const std::uint64_t nonzero_pairs =
        static_cast<std::uint64_t>(first->modes.size())
        * static_cast<std::uint64_t>(second->modes.size());
    result.diagnostics.zero_external_leg_pairs =
        full_pairs - nonzero_pairs;
    result.diagnostics.closing_zero_pairs =
        closing.pair_count;
    result.diagnostics.valid_pair_fraction =
        static_cast<double>(
            nonzero_pairs - closing.pair_count)
        / static_cast<double>(full_pairs);
    accumulate_weight_diagnostics(
        result.nodes,
        result.diagnostics.total_weight,
        result.diagnostics.total_variation,
        result.diagnostics.maximum_absolute_weight);
    if (std::fabs(
            result.diagnostics.total_weight
            - result.diagnostics.valid_pair_fraction)
        > 2.0e-10) {
        throw std::runtime_error(
            "exact-lattice shell valid-pair normalization failed");
    }
    return result;
}

HaarOrientedLatticeShellRule make_haar_oriented_lattice_shell_rule(
    const shell::ShellBin& bin,
    const HaarOrientedLatticeShellRuleConfig& config) {
    if (config.n_alpha < 1
        || config.n_cos_beta < 1
        || config.n_gamma < 1) {
        throw std::invalid_argument(
            "Haar-oriented lattice orders must be positive");
    }
    const ExactLatticeShellRule invariant =
        make_exact_lattice_shell_rule(bin, config.exact);
    const QuadratureRule cos_beta =
        gauss_legendre(config.n_cos_beta, -1.0, 1.0);
    const std::size_t orientation_count =
        static_cast<std::size_t>(config.n_alpha)
        * static_cast<std::size_t>(config.n_cos_beta)
        * static_cast<std::size_t>(config.n_gamma);

    HaarOrientedLatticeShellRule result;
    result.bin = bin;
    result.config = config;
    result.nodes.reserve(
        invariant.nodes.size() * orientation_count);
    for (const shell::ShellNode& base : invariant.nodes) {
        for (int alpha_index = 0;
             alpha_index < config.n_alpha;
             ++alpha_index) {
            const double alpha =
                kTwoPi * alpha_index / config.n_alpha;
            for (int beta_index = 0;
                 beta_index < config.n_cos_beta;
                 ++beta_index) {
                const double cosine =
                    cos_beta.nodes[
                        static_cast<std::size_t>(beta_index)];
                const double beta =
                    std::acos(std::clamp(cosine, -1.0, 1.0));
                const double beta_weight =
                    0.5 * cos_beta.weights[
                        static_cast<std::size_t>(beta_index)];
                for (int gamma_index = 0;
                     gamma_index < config.n_gamma;
                     ++gamma_index) {
                    const double gamma =
                        kTwoPi * gamma_index / config.n_gamma;
                    shell::ShellNode node = base;
                    for (std::size_t leg = 0;
                         leg < node.closed_vectors.size();
                         ++leg) {
                        node.closed_vectors[leg] =
                            rotate_zyz(
                                base.closed_vectors[leg],
                                alpha, beta, gamma);
                    }
                    node.alpha = alpha;
                    node.cos_beta = cosine;
                    node.gamma = gamma;
                    node.weight =
                        base.weight * beta_weight
                        / static_cast<double>(
                            config.n_alpha * config.n_gamma);
                    result.nodes.push_back(std::move(node));
                }
            }
        }
    }
    result.diagnostics.exact = invariant.diagnostics;
    result.diagnostics.invariant_nodes = invariant.nodes.size();
    result.diagnostics.orientation_nodes = orientation_count;
    accumulate_weight_diagnostics(
        result.nodes,
        result.diagnostics.total_weight,
        result.diagnostics.total_variation,
        result.diagnostics.maximum_absolute_weight);
    if (std::fabs(
            result.diagnostics.total_weight
            - invariant.diagnostics.total_weight)
        > 2.0e-12) {
        throw std::runtime_error(
            "Haar-oriented lattice normalization failed");
    }
    return result;
}

MultilevelLatticeShellRule make_multilevel_lattice_shell_rule(
    const shell::ShellBin& bin,
    const MultilevelLatticeShellRuleConfig& config) {
    if (config.interpolation_order < 1) {
        throw std::invalid_argument(
            "multilevel interpolation order must be positive");
    }
    const ExactLatticeShellRule exact =
        make_exact_lattice_shell_rule(bin, config.exact);
    const std::size_t radial =
        static_cast<std::size_t>(
            config.exact.radial_order);
    const std::size_t angular =
        static_cast<std::size_t>(
            config.exact.angular_order);
    const std::size_t interpolation =
        static_cast<std::size_t>(
            config.interpolation_order);
    if (exact.nodes.size() != radial * radial * angular) {
        throw std::logic_error(
            "exact-lattice node ordering is inconsistent");
    }

    MultilevelLatticeShellRule result;
    result.bin = bin;
    result.config = config;
    result.diagnostics.exact = exact.diagnostics;
    result.nodes.reserve(radial * radial * interpolation);
    for (std::size_t first_radial = 0;
         first_radial < radial;
         ++first_radial) {
        for (std::size_t second_radial = 0;
             second_radial < radial;
             ++second_radial) {
            const std::size_t offset =
                (first_radial * radial + second_radial)
                * angular;
            const double k1 = exact.nodes[offset].k1;
            const double k2 = exact.nodes[offset].k2;
            const QuadratureRule low =
                continuum_k3_rule(
                    k1, k2, config.interpolation_order);
            std::vector<long double> effective(
                interpolation, 0.0L);
            for (std::size_t high_index = 0;
                 high_index < angular;
                 ++high_index) {
                const shell::ShellNode& high =
                    exact.nodes[offset + high_index];
                const double high_k3 =
                    hv1::norm(high.closed_vectors[2]);
                const std::vector<double> basis =
                    barycentric_basis(low.nodes, high_k3);
                double lebesgue = 0.0;
                for (std::size_t low_index = 0;
                     low_index < interpolation;
                     ++low_index) {
                    lebesgue += std::fabs(basis[low_index]);
                    effective[low_index] +=
                        static_cast<long double>(high.weight)
                        * static_cast<long double>(
                            basis[low_index]);
                }
                result.diagnostics.maximum_sampled_lebesgue =
                    std::max(
                        result.diagnostics
                            .maximum_sampled_lebesgue,
                        lebesgue);
            }
            for (std::size_t low_index = 0;
                 low_index < interpolation;
                 ++low_index) {
                const double k3 = low.nodes[low_index];
                const double mu = std::clamp(
                    (k3 * k3 - k1 * k1 - k2 * k2)
                        / (2.0 * k1 * k2),
                    -1.0, 1.0);
                result.nodes.push_back(make_node(
                    k1,
                    k2,
                    mu,
                    static_cast<double>(
                        effective[low_index])));
            }
        }
    }
    accumulate_weight_diagnostics(
        result.nodes,
        result.diagnostics.total_weight,
        result.diagnostics.total_variation,
        result.diagnostics.maximum_absolute_weight);
    if (std::fabs(
            result.diagnostics.total_weight
            - exact.diagnostics.total_weight)
        > 2.0e-10) {
        throw std::runtime_error(
            "multilevel exact-lattice normalization failed");
    }
    return result;
}

}  // namespace marisa_b_eft_v2
