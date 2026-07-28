#include "shell_average.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_errno.h>

namespace marisa_b_shell_v1 {
namespace {

namespace hv1 = marisa_b_halo_v1;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
    std::uint64_t lattice_mode_count = 0;
    std::size_t lattice_unique_radius_count = 0;
    std::int64_t lattice_boundary_mode_adjustment = 0;
    bool fft_lattice_measure = false;
};

QuadratureRule gauss_legendre(int count, double lower, double upper) {
    if (count < 1 || !(upper > lower)) {
        throw std::invalid_argument("invalid Gauss-Legendre interval or order");
    }
    QuadratureRule rule;
    rule.nodes.resize(static_cast<std::size_t>(count));
    rule.weights.resize(static_cast<std::size_t>(count));
    const double midpoint = 0.5 * (upper + lower);
    const double half_width = 0.5 * (upper - lower);
    const int roots = (count + 1) / 2;
    for (int root = 0; root < roots; ++root) {
        double z = std::cos(kPi * (root + 0.75) / (count + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p0 = 1.0;
            double p1 = z;
            for (int degree = 2; degree <= count; ++degree) {
                const double next =
                    ((2.0 * degree - 1.0) * z * p1 - (degree - 1.0) * p0)
                    / degree;
                p0 = p1;
                p1 = next;
            }
            derivative = count * (z * p1 - p0) / (z * z - 1.0);
            const double updated = z - p1 / derivative;
            if (std::fabs(updated - z) <= 4.0e-16) {
                z = updated;
                break;
            }
            z = updated;
        }
        double p0 = 1.0;
        double p1 = z;
        for (int degree = 2; degree <= count; ++degree) {
            const double next =
                ((2.0 * degree - 1.0) * z * p1 - (degree - 1.0) * p0)
                / degree;
            p0 = p1;
            p1 = next;
        }
        derivative = count * (z * p1 - p0) / (z * z - 1.0);
        const double weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        const int left = root;
        const int right = count - 1 - root;
        rule.nodes[static_cast<std::size_t>(left)] = midpoint - half_width * z;
        rule.nodes[static_cast<std::size_t>(right)] = midpoint + half_width * z;
        rule.weights[static_cast<std::size_t>(left)] = half_width * weight;
        rule.weights[static_cast<std::size_t>(right)] = half_width * weight;
    }
    return rule;
}

QuadratureRule radial_volume_rule(int count, double lower, double upper) {
    if (!(lower >= 0.0 && upper > lower)) {
        throw std::invalid_argument("shell radii require 0 <= lower < upper");
    }
    const double lower_cubed = lower * lower * lower;
    const double upper_cubed = upper * upper * upper;
    QuadratureRule rule = gauss_legendre(count, lower_cubed, upper_cubed);
    const double normalization = upper_cubed - lower_cubed;
    for (std::size_t index = 0; index < rule.nodes.size(); ++index) {
        rule.nodes[index] = std::cbrt(rule.nodes[index]);
        rule.weights[index] /= normalization;
    }
    return rule;
}

QuadratureRule fft_lattice_radial_rule(
    int count,
    double lower,
    double upper,
    double box_size,
    int mesh_size,
    bool cap_order_to_support,
    ShellFftLatticeBinning binning,
    std::uint64_t expected_mode_count) {
    if (count < 1 || !(lower >= 0.0 && upper > lower)
        || !(box_size > 0.0) || mesh_size < 2) {
        throw std::invalid_argument("invalid FFT-lattice radial rule");
    }
    if (binning
            == ShellFftLatticeBinning::
                EstimatorModeCountConstrainedIntegerRadius
        && expected_mode_count == 0) {
        throw std::invalid_argument(
            "estimator-constrained FFT shell requires a positive "
            "expected mode count");
    }
    using CacheKey =
        std::tuple<
            int, double, double, double, int, bool, int,
            std::uint64_t>;
    static std::map<CacheKey, QuadratureRule> cache;
    static std::mutex cache_mutex;
    const CacheKey cache_key{
        count, lower, upper, box_size, mesh_size,
        cap_order_to_support, static_cast<int>(binning),
        expected_mode_count};
    {
        const std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(cache_key);
        if (found != cache.end()) return found->second;
    }
    std::map<double, std::uint64_t> multiplicities;
    std::int64_t boundary_mode_adjustment = 0;
    if (binning
        == ShellFftLatticeBinning::Float32CartesianRadius) {
        const float fundamental =
            static_cast<float>(
                kTwoPi / static_cast<float>(box_size));
        std::vector<float> frequencies(
            static_cast<std::size_t>(mesh_size));
        const int positive_count = (mesh_size + 1) / 2;
        for (int index = 0; index < mesh_size; ++index) {
            const int mode =
                index < positive_count
                    ? index : index - mesh_size;
            frequencies[static_cast<std::size_t>(index)] =
                static_cast<float>(mode) * fundamental;
        }

        const float lower_float = static_cast<float>(lower);
        const float upper_float = static_cast<float>(upper);
        for (float kz : frequencies) {
            const float kz_squared = kz * kz;
            for (float kx : frequencies) {
                const float kx_squared = kx * kx;
                for (float ky : frequencies) {
                    const float transverse_squared =
                        kx_squared + ky * ky;
                    const float radius =
                        std::sqrt(
                            transverse_squared + kz_squared);
                    if (radius >= lower_float
                        && radius < upper_float) {
                        ++multiplicities[
                            static_cast<double>(radius)];
                    }
                }
            }
        }
    } else {
        using SquaredMultiplicities =
            std::map<std::uint64_t, std::uint64_t>;
        static std::map<int, SquaredMultiplicities>
            squared_cache;
        static std::mutex squared_cache_mutex;
        SquaredMultiplicities squared_multiplicities;
        {
            const std::lock_guard<std::mutex> lock(
                squared_cache_mutex);
            const auto found =
                squared_cache.find(mesh_size);
            if (found != squared_cache.end()) {
                squared_multiplicities = found->second;
            }
        }
        if (squared_multiplicities.empty()) {
            const int positive_count =
                (mesh_size + 1) / 2;
            std::vector<std::int64_t> modes(
                static_cast<std::size_t>(mesh_size));
            for (int index = 0;
                 index < mesh_size;
                 ++index) {
                modes[static_cast<std::size_t>(index)] =
                    index < positive_count
                        ? index : index - mesh_size;
            }
            for (const std::int64_t nz : modes) {
                const std::uint64_t nz2 =
                    static_cast<std::uint64_t>(nz * nz);
                for (const std::int64_t nx : modes) {
                    const std::uint64_t nx2 =
                        static_cast<std::uint64_t>(nx * nx);
                    for (const std::int64_t ny : modes) {
                        const std::uint64_t ny2 =
                            static_cast<std::uint64_t>(
                                ny * ny);
                        ++squared_multiplicities[
                            nx2 + ny2 + nz2];
                    }
                }
            }
            const std::lock_guard<std::mutex> lock(
                squared_cache_mutex);
            const auto inserted =
                squared_cache.emplace(
                    mesh_size,
                    squared_multiplicities);
            if (!inserted.second) {
                squared_multiplicities =
                    inserted.first->second;
            }
        }

        const double fundamental =
            kTwoPi / box_size;
        const auto near_boundary =
            [](double radius, double boundary) {
                const double scale =
                    std::max(
                        1.0,
                        std::max(
                            std::fabs(radius),
                            std::fabs(boundary)));
                return std::fabs(radius - boundary)
                    <= 128.0
                       * std::numeric_limits<double>::
                             epsilon()
                       * scale;
            };
        std::map<double, std::uint64_t>
            lower_boundary;
        std::map<double, std::uint64_t>
            upper_boundary;
        for (const auto& item : squared_multiplicities) {
            const double radius =
                fundamental
                * std::sqrt(
                    static_cast<double>(item.first));
            if (near_boundary(radius, lower)) {
                lower_boundary[radius] += item.second;
            } else if (near_boundary(radius, upper)) {
                upper_boundary[radius] += item.second;
            } else if (radius > lower
                       && radius < upper) {
                multiplicities[radius] += item.second;
            }
        }
        for (const auto& item : lower_boundary) {
            multiplicities[item.first] += item.second;
        }
        std::uint64_t baseline_count = 0;
        for (const auto& item : multiplicities) {
            baseline_count += item.second;
        }
        if (expected_mode_count > baseline_count) {
            boundary_mode_adjustment =
                static_cast<std::int64_t>(
                    expected_mode_count - baseline_count);
            std::uint64_t remaining =
                expected_mode_count - baseline_count;
            for (const auto& item : upper_boundary) {
                const std::uint64_t take =
                    std::min(remaining, item.second);
                multiplicities[item.first] += take;
                remaining -= take;
                if (remaining == 0) break;
            }
            if (remaining != 0) {
                throw std::invalid_argument(
                    "expected FFT mode count cannot be "
                    "reached by assigning upper-boundary "
                    "lattice modes");
            }
        } else if (expected_mode_count < baseline_count) {
            boundary_mode_adjustment =
                -static_cast<std::int64_t>(
                    baseline_count - expected_mode_count);
            std::uint64_t remaining =
                baseline_count - expected_mode_count;
            for (const auto& item : lower_boundary) {
                const auto found =
                    multiplicities.find(item.first);
                if (found == multiplicities.end()) {
                    continue;
                }
                const std::uint64_t remove =
                    std::min(remaining, found->second);
                found->second -= remove;
                remaining -= remove;
                if (found->second == 0) {
                    multiplicities.erase(found);
                }
                if (remaining == 0) break;
            }
            if (remaining != 0) {
                throw std::invalid_argument(
                    "expected FFT mode count cannot be "
                    "reached by removing lower-boundary "
                    "lattice modes");
            }
        }
    }
    if (multiplicities.empty()) {
        throw std::invalid_argument(
            "FFT shell contains no lattice radii");
    }
    int effective_count=count;
    if (multiplicities.size()
        < static_cast<std::size_t>(count)) {
        if (!cap_order_to_support) {
            throw std::invalid_argument(
                "FFT shell has fewer unique radii "
                "than cubature order");
        }
        effective_count=
            static_cast<int>(multiplicities.size());
    }

    std::uint64_t total_modes = 0;
    for (const auto& item : multiplicities) total_modes += item.second;
    std::vector<double> support;
    std::vector<double> probabilities;
    support.reserve(multiplicities.size());
    probabilities.reserve(multiplicities.size());
    for (const auto& item : multiplicities) {
        support.push_back(item.first);
        probabilities.push_back(
            static_cast<double>(item.second) / static_cast<double>(total_modes));
    }

    std::vector<double> previous(support.size(), 0.0);
    std::vector<double> current(support.size(), 1.0);
    std::vector<double> diagonal(
        static_cast<std::size_t>(effective_count));
    std::vector<double> off_diagonal;
    off_diagonal.reserve(
        static_cast<std::size_t>(effective_count - 1));
    double previous_beta = 0.0;
    for (int degree = 0;
         degree < effective_count;
         ++degree) {
        double alpha = 0.0;
        for (std::size_t index = 0; index < support.size(); ++index) {
            alpha += probabilities[index] * support[index]
                     * current[index] * current[index];
        }
        diagonal[static_cast<std::size_t>(degree)] = alpha;
        if (degree + 1 == effective_count) break;
        std::vector<double> residual(support.size());
        double beta_squared = 0.0;
        for (std::size_t index = 0; index < support.size(); ++index) {
            residual[index] = (support[index] - alpha) * current[index]
                              - previous_beta * previous[index];
            beta_squared += probabilities[index] * residual[index] * residual[index];
        }
        const double beta = std::sqrt(beta_squared);
        if (!(beta > 0.0) || !std::isfinite(beta)) {
            throw std::runtime_error("FFT-lattice Stieltjes recurrence failed");
        }
        off_diagonal.push_back(beta);
        previous.swap(current);
        for (std::size_t index = 0; index < support.size(); ++index) {
            current[index] = residual[index] / beta;
        }
        previous_beta = beta;
    }

    gsl_matrix* jacobi =
        gsl_matrix_calloc(effective_count, effective_count);
    gsl_vector* eigenvalues =
        gsl_vector_alloc(effective_count);
    gsl_matrix* eigenvectors =
        gsl_matrix_alloc(effective_count, effective_count);
    gsl_eigen_symmv_workspace* workspace =
        gsl_eigen_symmv_alloc(effective_count);
    if (!jacobi || !eigenvalues || !eigenvectors || !workspace) {
        if (workspace) gsl_eigen_symmv_free(workspace);
        if (eigenvectors) gsl_matrix_free(eigenvectors);
        if (eigenvalues) gsl_vector_free(eigenvalues);
        if (jacobi) gsl_matrix_free(jacobi);
        throw std::bad_alloc();
    }
    for (int index = 0; index < effective_count; ++index) {
        gsl_matrix_set(jacobi, index, index, diagonal[static_cast<std::size_t>(index)]);
        if (index + 1 < effective_count) {
            const double beta = off_diagonal[static_cast<std::size_t>(index)];
            gsl_matrix_set(jacobi, index, index + 1, beta);
            gsl_matrix_set(jacobi, index + 1, index, beta);
        }
    }
    const int status = gsl_eigen_symmv(jacobi, eigenvalues, eigenvectors, workspace);
    gsl_eigen_symmv_free(workspace);
    gsl_matrix_free(jacobi);
    if (status != GSL_SUCCESS) {
        gsl_matrix_free(eigenvectors);
        gsl_vector_free(eigenvalues);
        throw std::runtime_error(
            std::string("FFT-lattice eigensolver failed: ") + gsl_strerror(status));
    }
    gsl_eigen_symmv_sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_VAL_ASC);

    QuadratureRule rule;
    rule.nodes.resize(
        static_cast<std::size_t>(effective_count));
    rule.weights.resize(
        static_cast<std::size_t>(effective_count));
    rule.lattice_mode_count=total_modes;
    rule.lattice_unique_radius_count=
        multiplicities.size();
    rule.lattice_boundary_mode_adjustment=
        boundary_mode_adjustment;
    rule.fft_lattice_measure=true;
    double weight_sum = 0.0;
    for (int index = 0; index < effective_count; ++index) {
        rule.nodes[static_cast<std::size_t>(index)] = gsl_vector_get(eigenvalues, index);
        const double first_component = gsl_matrix_get(eigenvectors, 0, index);
        rule.weights[static_cast<std::size_t>(index)] =
            first_component * first_component;
        weight_sum += rule.weights[static_cast<std::size_t>(index)];
    }
    gsl_matrix_free(eigenvectors);
    gsl_vector_free(eigenvalues);
    for (double& weight : rule.weights) weight /= weight_sum;
    {
        const std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(cache_key, rule);
    }
    return rule;
}

QuadratureRule shell_radial_rule(
    int count,
    double lower,
    double upper,
    const ShellQuadratureConfig& config) {
    if (config.radial_measure == ShellRadialMeasure::FftLattice) {
        return fft_lattice_radial_rule(
            count, lower, upper,
            config.fft_box_size, config.fft_mesh_size,
            config.cap_fft_lattice_radial_order_to_support,
            config.fft_lattice_binning,
            config.fft_lattice_expected_mode_count);
    }
    return radial_volume_rule(count, lower, upper);
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

ShellQuadratureConfig effective_shell_config(
    const hv1::IntegrationConfig& loop_config,
    ShellQuadratureConfig shell_config) {
    if (!loop_config.reconstruction.enabled
        || loop_config.reconstruction.cell_size == 0.0) {
        shell_config.average_grid_orientation = false;
    }
    return shell_config;
}

template<typename Result, typename Compute>
std::vector<Result> compute_nodes(
    const std::vector<ShellNode>& nodes,
    Compute compute) {
    std::vector<Result> values(nodes.size());
    std::exception_ptr failure;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(nodes.size());
         ++index) {
        try {
            values[static_cast<std::size_t>(index)] =
                compute(nodes[static_cast<std::size_t>(index)]);
        } catch (...) {
#pragma omp critical(marisa_b_shell_failure)
            {
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
    return values;
}

}  // namespace

std::vector<ShellNode> make_shell_nodes(
    const ShellBin& bin,
    const ShellQuadratureConfig& config) {
    if (config.n_radial < 1 || config.n_internal_mu < 1) {
        throw std::invalid_argument("shell radial and internal-mu orders must be positive");
    }
    if (config.average_grid_orientation
        && (config.n_alpha < 1 || config.n_cos_beta < 1 || config.n_gamma < 1)) {
        throw std::invalid_argument("Euler-angle orders must be positive");
    }
    const QuadratureRule k1_rule = shell_radial_rule(
        config.n_radial, bin.k1_lower, bin.k1_upper, config);
    const QuadratureRule k2_rule = shell_radial_rule(
        config.n_radial, bin.k2_lower, bin.k2_upper, config);
    QuadratureRule mu_rule = gauss_legendre(config.n_internal_mu, -1.0, 1.0);
    for (double& weight : mu_rule.weights) weight *= 0.5;

    QuadratureRule beta_rule;
    if (config.average_grid_orientation) {
        beta_rule = gauss_legendre(config.n_cos_beta, -1.0, 1.0);
        for (double& weight : beta_rule.weights) weight *= 0.5;
    } else {
        beta_rule.nodes = {1.0};
        beta_rule.weights = {1.0};
    }
    const int n_alpha = config.average_grid_orientation ? config.n_alpha : 1;
    const int n_gamma = config.average_grid_orientation ? config.n_gamma : 1;

    std::vector<ShellNode> nodes;
    nodes.reserve(
        k1_rule.nodes.size() * k2_rule.nodes.size() * mu_rule.nodes.size()
        * static_cast<std::size_t>(n_alpha) * beta_rule.nodes.size()
        * static_cast<std::size_t>(n_gamma));
    for (std::size_t i1 = 0; i1 < k1_rule.nodes.size(); ++i1) {
        for (std::size_t i2 = 0; i2 < k2_rule.nodes.size(); ++i2) {
            for (std::size_t imu = 0; imu < mu_rule.nodes.size(); ++imu) {
                const double k1 = k1_rule.nodes[i1];
                const double k2 = k2_rule.nodes[i2];
                const double mu = mu_rule.nodes[imu];
                const double sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                const std::array<hv1::Vec3, 3> base = {
                    hv1::Vec3{0.0, 0.0, k1},
                    hv1::Vec3{k2 * sine, 0.0, k2 * mu},
                    hv1::Vec3{-k2 * sine, 0.0, -k1 - k2 * mu}};
                for (int ialpha = 0; ialpha < n_alpha; ++ialpha) {
                    const double alpha = config.average_grid_orientation
                        ? kTwoPi * ialpha / n_alpha
                        : 0.0;
                    for (std::size_t ibeta = 0; ibeta < beta_rule.nodes.size(); ++ibeta) {
                        const double cos_beta = beta_rule.nodes[ibeta];
                        const double beta = config.average_grid_orientation
                            ? std::acos(std::max(-1.0, std::min(1.0, cos_beta)))
                            : 0.0;
                        for (int igamma = 0; igamma < n_gamma; ++igamma) {
                            const double gamma = config.average_grid_orientation
                                ? kTwoPi * igamma / n_gamma
                                : 0.0;
                            ShellNode node;
                            for (std::size_t index = 0; index < base.size(); ++index) {
                                node.closed_vectors[index] = rotate_zyz(
                                    base[index], alpha, beta, gamma);
                            }
                            node.k1 = k1;
                            node.k2 = k2;
                            node.internal_mu = mu;
                            node.alpha = alpha;
                            node.cos_beta = cos_beta;
                            node.gamma = gamma;
                            node.weight = k1_rule.weights[i1] * k2_rule.weights[i2]
                                          * mu_rule.weights[imu]
                                          * beta_rule.weights[ibeta]
                                          / (n_alpha * n_gamma);
                            nodes.push_back(std::move(node));
                        }
                    }
                }
            }
        }
    }
    return nodes;
}

std::vector<RadialShellNode> make_radial_shell_nodes(
    double lower,
    double upper,
    const ShellQuadratureConfig& config) {
    return make_radial_shell_rule(
        lower,upper,config).nodes;
}

RadialShellRule make_radial_shell_rule(
    double lower,
    double upper,
    const ShellQuadratureConfig& config) {
    const QuadratureRule rule = shell_radial_rule(
        config.n_radial, lower, upper, config);
    RadialShellRule result;
    result.nodes.resize(rule.nodes.size());
    result.requested_radial_order=config.n_radial;
    result.actual_radial_order=rule.nodes.size();
    result.lattice_mode_count=
        rule.lattice_mode_count;
    result.lattice_unique_radius_count=
        rule.lattice_unique_radius_count;
    result.lattice_boundary_mode_adjustment=
        rule.lattice_boundary_mode_adjustment;
    result.fft_lattice_measure=
        rule.fft_lattice_measure;
    for (std::size_t index = 0; index < rule.nodes.size(); ++index) {
        result.nodes[index] =
            RadialShellNode{
                rule.nodes[index],rule.weights[index]};
    }
    return result;
}

ShellComponentTemplates compute_shell_templates(
    const PowerSpectrum& linear_power,
    const ShellBin& bin,
    const hv1::IntegrationConfig& loop_config,
    const ShellQuadratureConfig& shell_config) {
    const ShellQuadratureConfig effective = effective_shell_config(loop_config, shell_config);
    const std::vector<ShellNode> nodes = make_shell_nodes(bin, effective);
    const std::vector<hv1::ComponentTemplates> values =
        compute_nodes<hv1::ComponentTemplates>(
            nodes,
            [&linear_power, &loop_config](const ShellNode& node) {
                return hv1::compute_templates_vectors(
                    linear_power, node.closed_vectors, loop_config);
            });

    ShellComponentTemplates result;
    result.bin = bin;
    result.shell_nodes = nodes.size();
    result.grid_orientation_averaged = effective.average_grid_orientation;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const double weight = nodes[index].weight;
        result.tree += values[index].tree * weight;
        result.B222 += values[index].B222 * weight;
        result.B321I += values[index].B321I * weight;
        result.B321II += values[index].B321II * weight;
        result.B411 += values[index].B411 * weight;
        result.one_loop += values[index].one_loop * weight;
        result.total += values[index].total * weight;
        result.stochastic_alpha3_raw += values[index].stochastic_alpha3_raw * weight;
        result.total_loop_nodes += values[index].loop_nodes;
    }
    return result;
}

ShellComponentValues compute_shell_direct(
    const PowerSpectrum& linear_power,
    const ShellBin& bin,
    const hv1::IntegrationConfig& loop_config,
    const ShellQuadratureConfig& shell_config,
    const hv1::BiasPoint& bias) {
    const ShellQuadratureConfig effective = effective_shell_config(loop_config, shell_config);
    const std::vector<ShellNode> nodes = make_shell_nodes(bin, effective);
    const std::vector<hv1::ComponentValues> values =
        compute_nodes<hv1::ComponentValues>(
            nodes,
            [&linear_power, &loop_config, &bias](const ShellNode& node) {
                return hv1::compute_direct_vectors(
                    linear_power, node.closed_vectors, loop_config, bias);
            });

    ShellComponentValues result;
    result.bin = bin;
    result.shell_nodes = nodes.size();
    result.grid_orientation_averaged = effective.average_grid_orientation;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const double weight = nodes[index].weight;
        result.tree += weight * values[index].tree;
        result.B222 += weight * values[index].B222;
        result.B321I += weight * values[index].B321I;
        result.B321II += weight * values[index].B321II;
        result.B411 += weight * values[index].B411;
        result.one_loop += weight * values[index].one_loop;
        result.total += weight * values[index].total;
        result.stochastic_alpha3_raw += weight * values[index].stochastic_alpha3_raw;
        result.total_loop_nodes += values[index].loop_nodes;
    }
    return result;
}

ShellComponentValues evaluate_shell_templates(
    const ShellComponentTemplates& templates,
    const hv1::BiasPoint& bias) {
    ShellComponentValues result;
    result.bin = templates.bin;
    result.tree = templates.tree.evaluate(bias);
    result.B222 = templates.B222.evaluate(bias);
    result.B321I = templates.B321I.evaluate(bias);
    result.B321II = templates.B321II.evaluate(bias);
    result.B411 = templates.B411.evaluate(bias);
    result.one_loop = templates.one_loop.evaluate(bias);
    result.total = templates.total.evaluate(bias);
    result.stochastic_alpha3_raw = templates.stochastic_alpha3_raw.evaluate(bias);
    result.stochastic_alpha4_raw = templates.stochastic_alpha4_raw;
    result.shell_nodes = templates.shell_nodes;
    result.total_loop_nodes = templates.total_loop_nodes;
    result.grid_orientation_averaged = templates.grid_orientation_averaged;
    return result;
}

}  // namespace marisa_b_shell_v1
