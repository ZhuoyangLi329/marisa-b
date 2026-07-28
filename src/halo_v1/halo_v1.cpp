#include "halo_v1.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "PowerSpectrum.h"
#include "marisa_b_native.h"

namespace marisa_b_halo_v1 {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kZeroTolerance = 1.0e-12;

int monomial_key(const Monomial& powers) {
    return (static_cast<int>(powers.b1) << 16)
           | (static_cast<int>(powers.b2) << 8)
           | static_cast<int>(powers.bK2);
}

bool same_monomial(const Monomial& left, const Monomial& right) {
    return monomial_key(left) == monomial_key(right);
}

constexpr int kPolynomialExponentCount = 7;
constexpr int kPolynomialDenseSize =
    kPolynomialExponentCount * kPolynomialExponentCount * kPolynomialExponentCount;

struct PolynomialProductWorkspace {
    std::array<double, kPolynomialDenseSize> coefficients = {};
    std::array<std::uint32_t, kPolynomialDenseSize> stamps = {};
    std::array<int, 84> touched = {};
    std::uint32_t generation = 0;
    std::size_t touched_count = 0;

    void begin() {
        ++generation;
        if (generation == 0) {
            stamps.fill(0);
            generation = 1;
        }
        touched_count = 0;
    }

    void add(int index, double value) {
        if (stamps[index] != generation) {
            if (touched_count == touched.size()) {
                throw std::overflow_error("bias polynomial has more than 84 monomials");
            }
            stamps[index] = generation;
            coefficients[index] = value;
            touched[touched_count++] = index;
        } else {
            coefficients[index] += value;
        }
    }
};

thread_local PolynomialProductWorkspace polynomial_product_workspace;

double integer_power(double base, int exponent) {
    double result = 1.0;
    for (int index = 0; index < exponent; ++index) result *= base;
    return result;
}

double clamp_cosine(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

double cosine(const Vec3& left, const Vec3& right) {
    const double denominator = norm(left) * norm(right);
    if (denominator == 0.0) return 0.0;
    return clamp_cosine(dot(left, right) / denominator);
}

Vec3 sum_vectors(const std::vector<Vec3>& vectors) {
    Vec3 total;
    for (const Vec3& vector : vectors) total = add(total, vector);
    return total;
}

Vec3 scale_vector(const Vec3& vector, double scalar) {
    return Vec3{scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

Vec3 cross_product(const Vec3& left, const Vec3& right) {
    return Vec3{
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x};
}

Vec3 unit_vector(const Vec3& vector) {
    const double length = norm(vector);
    if (!(length > kZeroTolerance)) {
        throw std::invalid_argument("cannot normalize a zero vector");
    }
    return scale_vector(vector, 1.0 / length);
}

struct OrthonormalFrame {
    Vec3 x;
    Vec3 y;
    Vec3 z;
};

OrthonormalFrame triangle_frame(const std::array<Vec3, 3>& external) {
    const Vec3 z = unit_vector(external[0]);
    Vec3 transverse = subtract(
        external[1], scale_vector(z, dot(external[1], z)));
    if (norm(transverse) <= kZeroTolerance * std::max(norm(external[1]), 1.0)) {
        const std::array<Vec3, 3> axes = {
            Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
        const Vec3* best = &axes[0];
        for (const Vec3& axis : axes) {
            if (std::fabs(dot(axis, z)) < std::fabs(dot(*best, z))) best = &axis;
        }
        transverse = subtract(*best, scale_vector(z, dot(*best, z)));
    }
    const Vec3 x = unit_vector(transverse);
    const Vec3 y = unit_vector(cross_product(z, x));
    return OrthonormalFrame{x, y, z};
}

bool is_zero_vector(const Vec3& vector) {
    return norm(vector) <= kZeroTolerance;
}

std::pair<int, int> find_opposite_pair(const std::vector<Vec3>& vectors) {
    double best = std::numeric_limits<double>::infinity();
    std::pair<int, int> result(-1, -1);
    for (int left = 0; left < static_cast<int>(vectors.size()); ++left) {
        for (int right = left + 1; right < static_cast<int>(vectors.size()); ++right) {
            const double scale = std::max({norm(vectors[left]), norm(vectors[right]), 1.0});
            const double residual = norm(add(vectors[left], vectors[right])) / scale;
            if (residual <= kZeroTolerance && residual < best) {
                best = residual;
                result = std::make_pair(left, right);
            }
        }
    }
    return result;
}

bool vector_less(const Vec3& left, const Vec3& right) {
    const double left_norm = norm(left);
    const double right_norm = norm(right);
    if (left_norm != right_norm) return left_norm < right_norm;
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    return left.z < right.z;
}

KernelTemplate scale_kernel(KernelTemplate kernel, double scalar) {
    kernel.regular *= scalar;
    for (TadpoleTerm& term : kernel.matter_f3_tadpoles) term.coefficient *= scalar;
    return kernel;
}

void add_tadpole(std::vector<TadpoleTerm>& target, const TadpoleTerm& addition) {
    for (TadpoleTerm& term : target) {
        const double scale = std::max({term.external_k, addition.external_k, 1.0});
        if (std::fabs(term.external_k - addition.external_k) <= kZeroTolerance * scale) {
            term.coefficient += addition.coefficient;
            return;
        }
    }
    target.push_back(addition);
}

KernelTemplate add_kernel(KernelTemplate left, const KernelTemplate& right) {
    left.regular += right.regular;
    for (const TadpoleTerm& term : right.matter_f3_tadpoles) {
        add_tadpole(left.matter_f3_tadpoles, term);
    }
    return left;
}

KernelTemplate multiply_kernel(const KernelTemplate& left, const KernelTemplate& right) {
    if (!left.matter_f3_tadpoles.empty() && !right.matter_f3_tadpoles.empty()) {
        throw std::logic_error("a product of two matter F3 tadpoles is outside n <= 4");
    }

    KernelTemplate result;
    result.regular = left.regular * right.regular;
    for (const TadpoleTerm& term : left.matter_f3_tadpoles) {
        add_tadpole(
            result.matter_f3_tadpoles,
            TadpoleTerm{term.external_k, term.coefficient * right.regular});
    }
    for (const TadpoleTerm& term : right.matter_f3_tadpoles) {
        add_tadpole(
            result.matter_f3_tadpoles,
            TadpoleTerm{term.external_k, term.coefficient * left.regular});
    }
    return result;
}

KernelTemplate multiply_kernel(KernelTemplate kernel, const Polynomial& polynomial) {
    kernel.regular = kernel.regular * polynomial;
    for (TadpoleTerm& term : kernel.matter_f3_tadpoles) {
        term.coefficient = term.coefficient * polynomial;
    }
    return kernel;
}

KernelTemplate matter_kernel(const std::vector<Vec3>& momenta) {
    const int order = static_cast<int>(momenta.size());
    if (order < 1 || order > 4) {
        throw std::invalid_argument("matter kernel order must be between one and four");
    }

    KernelTemplate result;
    if (order == 1) {
        result.regular = Polynomial::constant(1.0);
        return result;
    }
    if (is_zero_vector(sum_vectors(momenta))) {
        result.regular = Polynomial::constant(0.0);
        return result;
    }

    if (order == 2) {
        result.regular = Polynomial::constant(marisa_b_native::F2eds(
            norm(momenta[0]), norm(momenta[1]), cosine(momenta[0], momenta[1])));
        return result;
    }

    const std::pair<int, int> opposite = find_opposite_pair(momenta);
    if (order == 3 && opposite.first >= 0) {
        int external = 0;
        while (external == opposite.first || external == opposite.second) ++external;
        result.matter_f3_tadpoles.push_back(
            TadpoleTerm{norm(momenta[external]), Polynomial::constant(1.0)});
        return result;
    }

    if (order == 3) {
        result.regular = Polynomial::constant(marisa_b_native::F3eds(
            norm(momenta[0]),
            norm(momenta[1]),
            norm(momenta[2]),
            cosine(momenta[1], momenta[2]),
            cosine(momenta[0], momenta[1]),
            cosine(momenta[0], momenta[2])));
        return result;
    }

    if (opposite.first < 0) {
        throw std::invalid_argument(
            "the MARISA-B F4 backend is defined for an opposite loop pair");
    }

    std::vector<Vec3> external;
    for (int index = 0; index < order; ++index) {
        if (index != opposite.first && index != opposite.second) external.push_back(momenta[index]);
    }
    std::sort(external.begin(), external.end(), vector_less);
    Vec3 q = momenta[opposite.first];
    Vec3 minus_q = momenta[opposite.second];
    if (vector_less(minus_q, q)) std::swap(q, minus_q);

    result.regular = Polynomial::constant(marisa_b_native::F4edsb(
        norm(external[0]),
        norm(external[1]),
        norm(q),
        norm(minus_q),
        cosine(external[1], q),
        cosine(external[1], minus_q),
        cosine(external[0], external[1]),
        cosine(external[0], q),
        cosine(external[0], minus_q)));
    return result;
}

int popcount(unsigned int mask) {
    int count = 0;
    while (mask != 0U) {
        count += static_cast<int>(mask & 1U);
        mask >>= 1U;
    }
    return count;
}

double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double value = 1.0;
    for (int index = 1; index <= k; ++index) {
        value *= static_cast<double>(n - k + index) / static_cast<double>(index);
    }
    return value;
}

double factorial(int n) {
    double value = 1.0;
    for (int index = 2; index <= n; ++index) value *= static_cast<double>(index);
    return value;
}

std::vector<Vec3> select_vectors(const std::vector<Vec3>& momenta, unsigned int mask) {
    std::vector<Vec3> selected;
    for (int index = 0; index < static_cast<int>(momenta.size()); ++index) {
        if ((mask & (1U << index)) != 0U) selected.push_back(momenta[index]);
    }
    return selected;
}

KernelTemplate quadratic_density_kernel(const std::vector<Vec3>& momenta) {
    const int order = static_cast<int>(momenta.size());
    KernelTemplate result;
    if (order < 2) return result;
    const unsigned int full_mask = (1U << order) - 1U;
    for (int left_order = 1; left_order < order; ++left_order) {
        const double weight = 0.5 / binomial(order, left_order);
        for (unsigned int mask = 1U; mask < full_mask; ++mask) {
            if (popcount(mask) != left_order) continue;
            const std::vector<Vec3> left = select_vectors(momenta, mask);
            const std::vector<Vec3> right = select_vectors(momenta, full_mask ^ mask);
            result = add_kernel(
                std::move(result),
                scale_kernel(multiply_kernel(matter_kernel(left), matter_kernel(right)), weight));
        }
    }
    return result;
}

KernelTemplate quadratic_tidal_kernel(const std::vector<Vec3>& momenta) {
    const int order = static_cast<int>(momenta.size());
    KernelTemplate result;
    if (order < 2) return result;
    const unsigned int full_mask = (1U << order) - 1U;
    for (int left_order = 1; left_order < order; ++left_order) {
        const double combinatorial = 1.0 / binomial(order, left_order);
        for (unsigned int mask = 1U; mask < full_mask; ++mask) {
            if (popcount(mask) != left_order) continue;
            const std::vector<Vec3> left = select_vectors(momenta, mask);
            const std::vector<Vec3> right = select_vectors(momenta, full_mask ^ mask);
            const double weight = combinatorial * tidal_s2(sum_vectors(left), sum_vectors(right));
            result = add_kernel(
                std::move(result),
                scale_kernel(multiply_kernel(matter_kernel(left), matter_kernel(right)), weight));
        }
    }
    return result;
}

void generate_partitions_recursive(
    int next,
    int count,
    std::vector<std::vector<int>>& blocks,
    std::vector<std::vector<std::vector<int>>>& output) {
    if (next == count) {
        output.push_back(blocks);
        return;
    }
    for (int block = 0; block < static_cast<int>(blocks.size()); ++block) {
        blocks[block].push_back(next);
        generate_partitions_recursive(next + 1, count, blocks, output);
        blocks[block].pop_back();
    }
    blocks.push_back(std::vector<int>(1, next));
    generate_partitions_recursive(next + 1, count, blocks, output);
    blocks.pop_back();
}

const std::vector<std::vector<std::vector<int>>>& set_partitions(int count) {
    if (count < 1 || count > 4) {
        throw std::invalid_argument("partition order must be one through four");
    }
    static const std::array<std::vector<std::vector<std::vector<int>>>, 5> cache = [] {
        std::array<std::vector<std::vector<std::vector<int>>>, 5> result;
        for (int order = 1; order <= 4; ++order) {
            std::vector<std::vector<int>> blocks;
            generate_partitions_recursive(0, order, blocks, result[order]);
        }
        return result;
    }();
    return cache[count];
}

std::vector<Vec3> block_vectors(
    const std::vector<Vec3>& momenta,
    const std::vector<int>& block) {
    std::vector<Vec3> result;
    result.reserve(block.size());
    for (int index : block) result.push_back(momenta[index]);
    return result;
}

double sinc(double argument) {
    if (std::fabs(argument) < 1.0e-6) {
        const double square = argument * argument;
        return 1.0 - square / 6.0 + square * square / 120.0;
    }
    return std::sin(argument) / argument;
}

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int count, double lower, double upper) {
    if (count < 1 || !(upper > lower)) throw std::invalid_argument("invalid Gauss-Legendre rule");
    QuadratureRule rule;
    rule.nodes.resize(count);
    rule.weights.resize(count);
    const int half = (count + 1) / 2;
    const double midpoint = 0.5 * (lower + upper);
    const double half_width = 0.5 * (upper - lower);
    for (int index = 0; index < half; ++index) {
        double root = std::cos(kPi * (static_cast<double>(index) + 0.75)
                               / (static_cast<double>(count) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p_previous = 1.0;
            double p_current = root;
            for (int order = 2; order <= count; ++order) {
                const double p_next = ((2.0 * order - 1.0) * root * p_current
                                       - (order - 1.0) * p_previous)
                                      / static_cast<double>(order);
                p_previous = p_current;
                p_current = p_next;
            }
            const double p_n = count == 1 ? root : p_current;
            const double p_nm1 = count == 1 ? 1.0 : p_previous;
            derivative = static_cast<double>(count) * (root * p_n - p_nm1)
                         / (root * root - 1.0);
            const double update = p_n / derivative;
            root -= update;
            if (std::fabs(update) < 4.0 * std::numeric_limits<double>::epsilon()) break;
        }
        const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        rule.nodes[index] = midpoint - half_width * root;
        rule.nodes[count - 1 - index] = midpoint + half_width * root;
        rule.weights[index] = half_width * weight;
        rule.weights[count - 1 - index] = half_width * weight;
    }
    return rule;
}

QuadratureRule composite_radial_rule(
    int count_per_segment,
    double qmin,
    double qmax,
    const std::array<Vec3, 3>& external,
    RadialCoordinate coordinate) {
    std::vector<double> breakpoints = {qmin, qmax};
    for (const Vec3& momentum : external) {
        const double k = norm(momentum);
        if (k > qmin && k < qmax) breakpoints.push_back(k);
    }
    std::sort(breakpoints.begin(), breakpoints.end());
    breakpoints.erase(
        std::unique(
            breakpoints.begin(),
            breakpoints.end(),
            [](double left, double right) {
                return std::fabs(left - right)
                       <= kZeroTolerance * std::max({left, right, 1.0});
            }),
        breakpoints.end());

    QuadratureRule result;
    result.nodes.reserve(count_per_segment * (breakpoints.size() - 1));
    result.weights.reserve(count_per_segment * (breakpoints.size() - 1));
    for (std::size_t segment = 0; segment + 1 < breakpoints.size(); ++segment) {
        const bool logarithmic = coordinate == RadialCoordinate::Logarithmic;
        QuadratureRule piece = logarithmic
            ? gauss_legendre(
                  count_per_segment,
                  std::log(breakpoints[segment]),
                  std::log(breakpoints[segment + 1]))
            : gauss_legendre(
                  count_per_segment,
                  breakpoints[segment],
                  breakpoints[segment + 1]);
        for (std::size_t index = 0; index < piece.nodes.size(); ++index) {
            const double q = logarithmic ? std::exp(piece.nodes[index]) : piece.nodes[index];
            piece.nodes[index] = q;
            piece.weights[index] *= logarithmic ? q * q * q : q * q;
        }
        result.nodes.insert(result.nodes.end(), piece.nodes.begin(), piece.nodes.end());
        result.weights.insert(result.weights.end(), piece.weights.begin(), piece.weights.end());
    }
    return result;
}

Vec3 loop_vector(double q, double mu, double phi, const OrthonormalFrame& frame) {
    const double transverse = q * std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return add(
        add(
            scale_vector(frame.x, transverse * std::cos(phi)),
            scale_vector(frame.y, transverse * std::sin(phi))),
        scale_vector(frame.z, q * mu));
}

double safe_power(const PowerSpectrum& power, double k) {
    return power(std::max(k, 1.0e-12));
}

double matter_f3_tadpole_integral(
    const PowerSpectrum& power,
    double external_k,
    const IntegrationConfig& config) {
    marisa_b_native::NativeConfig native;
    native.qmin = config.qmin;
    native.qmax = config.qmax;
    native.p13_epsrel = config.p13_epsrel;
    native.epsabs = config.p13_epsabs;
    const double cache_stable_k = std::round(external_k * 1.0e12) / 1.0e12;
    static std::mutex p13_cache_mutex;
    std::lock_guard<std::mutex> lock(p13_cache_mutex);
    const marisa_b_native::ComponentStats p13 =
        marisa_b_native::P13_dd(power, cache_stable_k, native);
    const double pk = safe_power(power, cache_stable_k);
    if (pk == 0.0) throw std::runtime_error("P_L(k) is zero in the P13 tadpole conversion");
    return p13.value / (6.0 * pk);
}

KernelTemplate active_kernel(
    const std::vector<Vec3>& momenta,
    const ReconstructionConfig& reconstruction) {
    return reconstruction.enabled
        ? reconstructed_kernel(momenta, reconstruction)
        : pre_reconstruction_kernel(momenta);
}

template<typename Value>
struct CoreResult {
    CanonicalTriangle triangle;
    Value tree;
    Value B222;
    Value B321I;
    Value B321II;
    Value B411;
    Value one_loop;
    Value total;
    Value stochastic_alpha3_raw;
    std::size_t loop_nodes = 0;
};

template<typename Project>
auto compute_core(
    const PowerSpectrum& power,
    const CanonicalTriangle& input_triangle,
    const IntegrationConfig& config,
    Project project)
    -> CoreResult<decltype(project(std::declval<const Polynomial&>()))> {
    using Value = decltype(project(std::declval<const Polynomial&>()));
    if (!(config.qmin > 0.0 && config.qmax > config.qmin)) {
        throw std::invalid_argument("integration requires 0 < qmin < qmax");
    }
    if (config.n_radial < 1 || config.n_mu < 1 || config.n_phi < 1) {
        throw std::invalid_argument("all integration grid sizes must be positive");
    }

    CoreResult<Value> result;
    result.triangle = input_triangle;
    const std::array<Vec3, 3> external = {
        result.triangle.k1, result.triangle.k2, result.triangle.k3};
    const OrthonormalFrame frame = triangle_frame(external);
    std::array<double, 3> pk = {};
    std::array<KernelTemplate, 3> k1_templates;
    std::array<Value, 3> k1_values;
    for (int index = 0; index < 3; ++index) {
        pk[index] = safe_power(power, norm(external[index]));
        k1_templates[index] = active_kernel({external[index]}, config.reconstruction);
        k1_values[index] = project(k1_templates[index].regular);
    }

    result.tree = Value{};
    for (int left = 0; left < 3; ++left) {
        for (int right = left + 1; right < 3; ++right) {
            const Value k2 = project(active_kernel(
                {external[left], external[right]}, config.reconstruction).regular);
            result.tree += 2.0 * pk[left] * pk[right] * k1_values[left] * k1_values[right] * k2;
        }
    }

    const QuadratureRule radial_rule = composite_radial_rule(
        config.n_radial,
        config.qmin,
        config.qmax,
        external,
        config.radial_coordinate);
    const QuadratureRule mu_rule = gauss_legendre(config.n_mu, -1.0, 1.0);
    const QuadratureRule phi_rule = gauss_legendre(config.n_phi, 0.0, kTwoPi);
    result.loop_nodes = radial_rule.nodes.size()
                        * static_cast<std::size_t>(config.n_mu)
                        * static_cast<std::size_t>(config.n_phi);

    result.B222 = Value{};
    result.B321I = Value{};
    result.B411 = Value{};
    std::array<Value, 3> i3 = {Value{}, Value{}, Value{}};
    const double measure_prefactor = 1.0 / (kTwoPi * kTwoPi * kTwoPi);

    for (std::size_t iq = 0; iq < radial_rule.nodes.size(); ++iq) {
        const double q = radial_rule.nodes[iq];
        for (int imu = 0; imu < config.n_mu; ++imu) {
            for (int iphi = 0; iphi < config.n_phi; ++iphi) {
                const Vec3 loop = loop_vector(
                    q, mu_rule.nodes[imu], phi_rule.nodes[iphi], frame);
                const Vec3 minus_loop = negate(loop);
                const double weight = radial_rule.weights[iq] * mu_rule.weights[imu]
                                      * phi_rule.weights[iphi]
                                      * measure_prefactor;
                const double pq = safe_power(power, q);

                const Vec3 k1_minus_q = subtract(external[0], loop);
                const Vec3 k2_plus_q = add(external[1], loop);
                const Value b222_k1 = project(active_kernel(
                    {loop, k1_minus_q}, config.reconstruction).regular);
                const Value b222_k2 = project(active_kernel(
                    {minus_loop, k2_plus_q}, config.reconstruction).regular);
                const Value b222_k3 = project(active_kernel(
                    {negate(k1_minus_q), negate(k2_plus_q)}, config.reconstruction).regular);
                result.B222 += 8.0 * weight * pq
                               * safe_power(power, norm(k1_minus_q))
                               * safe_power(power, norm(k2_plus_q))
                               * b222_k1 * b222_k2 * b222_k3;

                std::array<Value, 3> k2_by_j;
                std::array<Vec3, 3> kj_minus_q;
                std::array<double, 3> p_kj_minus_q = {};
                for (int j = 0; j < 3; ++j) {
                    kj_minus_q[j] = subtract(external[j], loop);
                    p_kj_minus_q[j] = safe_power(power, norm(kj_minus_q[j]));
                    k2_by_j[j] = project(active_kernel(
                        {loop, kj_minus_q[j]}, config.reconstruction).regular);
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (i == j) continue;
                        const Value k3 = project(active_kernel(
                            {external[i], loop, kj_minus_q[j]}, config.reconstruction).regular);
                        result.B321I += 6.0 * weight * pk[i] * pq * p_kj_minus_q[j]
                                        * k1_values[i] * k2_by_j[j] * k3;
                    }
                }

                for (int i = 0; i < 3; ++i) {
                    const KernelTemplate k3_tadpole = active_kernel(
                        {external[i], loop, minus_loop}, config.reconstruction);
                    i3[i] += weight * pq * project(k3_tadpole.regular);
                }

                for (int i = 0; i < 3; ++i) {
                    for (int j = i + 1; j < 3; ++j) {
                        const KernelTemplate k4 = active_kernel(
                            {external[i], external[j], loop, minus_loop},
                            config.reconstruction);
                        result.B411 += 12.0 * weight * pq * pk[i] * pk[j]
                                       * k1_values[i] * k1_values[j]
                                       * project(k4.regular);
                    }
                }
            }
        }
    }

    const double reference_q = std::sqrt(config.qmin * config.qmax);
    const Vec3 reference_loop = loop_vector(reference_q, 0.371, 1.127, frame);
    const Vec3 reference_minus_loop = negate(reference_loop);
    for (int i = 0; i < 3; ++i) {
        const KernelTemplate k3_reference = active_kernel(
            {external[i], reference_loop, reference_minus_loop},
            config.reconstruction);
        for (const TadpoleTerm& tadpole : k3_reference.matter_f3_tadpoles) {
            i3[i] += matter_f3_tadpole_integral(power, tadpole.external_k, config)
                     * project(tadpole.coefficient);
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            const KernelTemplate k4_reference = active_kernel(
                {external[i], external[j], reference_loop, reference_minus_loop},
                config.reconstruction);
            for (const TadpoleTerm& tadpole : k4_reference.matter_f3_tadpoles) {
                result.B411 += 12.0 * pk[i] * pk[j] * k1_values[i] * k1_values[j]
                               * matter_f3_tadpole_integral(power, tadpole.external_k, config)
                               * project(tadpole.coefficient);
            }
        }
    }

    result.B321II = Value{};
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            const Value k2 = project(active_kernel(
                {external[i], external[j]}, config.reconstruction).regular);
            result.B321II += 6.0 * pk[i] * pk[j] * k2
                             * (k1_values[i] * i3[j] + k1_values[j] * i3[i]);
        }
    }

    result.one_loop = result.B222 + result.B321I + result.B321II + result.B411;
    result.total = result.tree + result.one_loop;
    const Polynomial b1_squared = Polynomial::variable_b1() * Polynomial::variable_b1();
    result.stochastic_alpha3_raw = project(
        b1_squared * (pk[0] + pk[1] + pk[2]));
    return result;
}

}  // namespace

Polynomial::Polynomial(const Polynomial& other) noexcept
    : size_(other.size_) {
    std::copy_n(other.terms_.begin(), size_, terms_.begin());
}

Polynomial::Polynomial(Polynomial&& other) noexcept
    : size_(other.size_) {
    std::copy_n(other.terms_.begin(), size_, terms_.begin());
    other.size_ = 0;
}

Polynomial& Polynomial::operator=(const Polynomial& other) noexcept {
    if (this != &other) {
        size_ = other.size_;
        std::copy_n(other.terms_.begin(), size_, terms_.begin());
    }
    return *this;
}

Polynomial& Polynomial::operator=(Polynomial&& other) noexcept {
    if (this != &other) {
        size_ = other.size_;
        std::copy_n(other.terms_.begin(), size_, terms_.begin());
        other.size_ = 0;
    }
    return *this;
}

void Polynomial::append(Monomial powers, double coefficient) {
    if (coefficient == 0.0) return;
    const int total_degree = static_cast<int>(powers.b1)
                             + static_cast<int>(powers.b2)
                             + static_cast<int>(powers.bK2);
    if (total_degree > kMaximumTotalDegree) {
        throw std::overflow_error("bias polynomial total degree exceeds six");
    }
    if (size_ == kMaximumTerms) {
        throw std::overflow_error("bias polynomial has more than 84 monomials");
    }
    if (size_ != 0 && monomial_key(terms_[size_ - 1].powers) >= monomial_key(powers)) {
        throw std::logic_error("bias polynomial terms are not strictly ordered");
    }
    terms_[size_++] = PolynomialTerm{powers, coefficient};
}

Polynomial Polynomial::constant(double value) {
    Polynomial result;
    result.append(Monomial{0, 0, 0}, value);
    return result;
}

Polynomial Polynomial::variable_b1() {
    Polynomial result;
    result.append(Monomial{1, 0, 0}, 1.0);
    return result;
}

Polynomial Polynomial::variable_b2() {
    Polynomial result;
    result.append(Monomial{0, 1, 0}, 1.0);
    return result;
}

Polynomial Polynomial::variable_bK2() {
    Polynomial result;
    result.append(Monomial{0, 0, 1}, 1.0);
    return result;
}

double Polynomial::evaluate(const BiasPoint& bias) const {
    double value = 0.0;
    for (std::size_t index = 0; index < size_; ++index) {
        const PolynomialTerm& term = terms_[index];
        value += term.coefficient
                 * integer_power(bias.b1, term.powers.b1)
                 * integer_power(bias.b2, term.powers.b2)
                 * integer_power(bias.bK2, term.powers.bK2);
    }
    return value;
}

double Polynomial::coefficient(int b1_power, int b2_power, int bK2_power) const {
    if (b1_power < 0 || b2_power < 0 || bK2_power < 0
        || b1_power > 255 || b2_power > 255 || bK2_power > 255) {
        return 0.0;
    }
    const Monomial target{
        static_cast<std::uint8_t>(b1_power),
        static_cast<std::uint8_t>(b2_power),
        static_cast<std::uint8_t>(bK2_power)};
    for (std::size_t index = 0; index < size_; ++index) {
        const PolynomialTerm& term = terms_[index];
        if (same_monomial(term.powers, target)) return term.coefficient;
    }
    return 0.0;
}

bool Polynomial::empty() const {
    return size_ == 0;
}

std::vector<PolynomialTerm> Polynomial::terms() const {
    return std::vector<PolynomialTerm>(terms_.begin(), terms_.begin() + size_);
}

std::string Polynomial::expression() const {
    if (size_ == 0) return "0";
    std::ostringstream output;
    output << std::setprecision(17);
    for (std::size_t index = 0; index < size_; ++index) {
        const PolynomialTerm& term = terms_[index];
        if (index != 0) output << (term.coefficient >= 0.0 ? " + " : " - ");
        else if (term.coefficient < 0.0) output << "-";
        output << std::fabs(term.coefficient);
        if (term.powers.b1 != 0) output << "*b1^" << static_cast<int>(term.powers.b1);
        if (term.powers.b2 != 0) output << "*b2^" << static_cast<int>(term.powers.b2);
        if (term.powers.bK2 != 0) output << "*bK2^" << static_cast<int>(term.powers.bK2);
    }
    return output.str();
}

Polynomial& Polynomial::operator+=(const Polynomial& other) {
    Polynomial combined;
    std::size_t left = 0;
    std::size_t right = 0;
    while (left < size_ || right < other.size_) {
        if (right == other.size_
            || (left < size_
                && monomial_key(terms_[left].powers)
                       < monomial_key(other.terms_[right].powers))) {
            combined.append(terms_[left].powers, terms_[left].coefficient);
            ++left;
        } else if (left == size_
                   || monomial_key(other.terms_[right].powers)
                          < monomial_key(terms_[left].powers)) {
            combined.append(other.terms_[right].powers, other.terms_[right].coefficient);
            ++right;
        } else {
            combined.append(
                terms_[left].powers,
                terms_[left].coefficient + other.terms_[right].coefficient);
            ++left;
            ++right;
        }
    }
    *this = std::move(combined);
    return *this;
}

Polynomial& Polynomial::operator-=(const Polynomial& other) {
    Polynomial combined;
    std::size_t left = 0;
    std::size_t right = 0;
    while (left < size_ || right < other.size_) {
        if (right == other.size_
            || (left < size_
                && monomial_key(terms_[left].powers)
                       < monomial_key(other.terms_[right].powers))) {
            combined.append(terms_[left].powers, terms_[left].coefficient);
            ++left;
        } else if (left == size_
                   || monomial_key(other.terms_[right].powers)
                          < monomial_key(terms_[left].powers)) {
            combined.append(other.terms_[right].powers, -other.terms_[right].coefficient);
            ++right;
        } else {
            combined.append(
                terms_[left].powers,
                terms_[left].coefficient - other.terms_[right].coefficient);
            ++left;
            ++right;
        }
    }
    *this = std::move(combined);
    return *this;
}

Polynomial& Polynomial::operator*=(double scalar) {
    if (scalar == 0.0) {
        size_ = 0;
        return *this;
    }
    for (std::size_t index = 0; index < size_; ++index) {
        terms_[index].coefficient *= scalar;
    }
    return *this;
}

Polynomial operator+(Polynomial left, const Polynomial& right) {
    left += right;
    return left;
}

Polynomial operator-(Polynomial left, const Polynomial& right) {
    left -= right;
    return left;
}

Polynomial operator*(const Polynomial& left, const Polynomial& right) {
    if (left.size_ == 0 || right.size_ == 0) return Polynomial();
    PolynomialProductWorkspace& workspace = polynomial_product_workspace;
    workspace.begin();
    for (std::size_t left_index = 0; left_index < left.size_; ++left_index) {
        const PolynomialTerm& left_term = left.terms_[left_index];
        for (std::size_t right_index = 0; right_index < right.size_; ++right_index) {
            const PolynomialTerm& right_term = right.terms_[right_index];
            const int b1_power = static_cast<int>(left_term.powers.b1)
                                 + static_cast<int>(right_term.powers.b1);
            const int b2_power = static_cast<int>(left_term.powers.b2)
                                 + static_cast<int>(right_term.powers.b2);
            const int bK2_power = static_cast<int>(left_term.powers.bK2)
                                  + static_cast<int>(right_term.powers.bK2);
            if (b1_power + b2_power + bK2_power > Polynomial::kMaximumTotalDegree) {
                throw std::overflow_error("bias polynomial total degree exceeds six");
            }
            const int dense_index =
                (b1_power * kPolynomialExponentCount + b2_power)
                    * kPolynomialExponentCount
                + bK2_power;
            workspace.add(
                dense_index, left_term.coefficient * right_term.coefficient);
        }
    }
    std::sort(
        workspace.touched.begin(),
        workspace.touched.begin() + workspace.touched_count);
    Polynomial result;
    for (std::size_t touched_index = 0;
         touched_index < workspace.touched_count;
         ++touched_index) {
        const int dense_index = workspace.touched[touched_index];
        const int b1_power = dense_index
                             / (kPolynomialExponentCount * kPolynomialExponentCount);
        const int remainder = dense_index
                              % (kPolynomialExponentCount * kPolynomialExponentCount);
        const int b2_power = remainder / kPolynomialExponentCount;
        const int bK2_power = remainder % kPolynomialExponentCount;
        result.append(
            Monomial{
                static_cast<std::uint8_t>(b1_power),
                static_cast<std::uint8_t>(b2_power),
                static_cast<std::uint8_t>(bK2_power)},
            workspace.coefficients[dense_index]);
    }
    return result;
}

Polynomial operator*(Polynomial polynomial, double scalar) {
    polynomial *= scalar;
    return polynomial;
}

Polynomial operator*(double scalar, Polynomial polynomial) {
    polynomial *= scalar;
    return polynomial;
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return Vec3{left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return Vec3{left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 negate(const Vec3& vector) {
    return Vec3{-vector.x, -vector.y, -vector.z};
}

double dot(const Vec3& left, const Vec3& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double norm(const Vec3& vector) {
    return std::sqrt(std::max(0.0, dot(vector, vector)));
}

double tidal_s2(const Vec3& left, const Vec3& right) {
    if (is_zero_vector(left) || is_zero_vector(right)) return 0.0;
    const double mu = cosine(left, right);
    return mu * mu - 1.0 / 3.0;
}

CanonicalTriangle canonicalize_triangle(const Triangle& triangle) {
    std::array<double, 3> sides = {triangle.k1, triangle.k2, triangle.k3};
    for (double side : sides) {
        if (!(side > 0.0) || !std::isfinite(side)) {
            throw std::invalid_argument("triangle sides must be finite and positive");
        }
    }
    std::sort(sides.begin(), sides.end());
    if (sides[0] + sides[1] < sides[2] - 1.0e-12 * sides[2]) {
        throw std::invalid_argument("triangle inequality is not satisfied");
    }
    const double mu = clamp_cosine(
        (sides[2] * sides[2] - sides[0] * sides[0] - sides[1] * sides[1])
        / (2.0 * sides[0] * sides[1]));
    const double sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));

    CanonicalTriangle result;
    result.sides = Triangle{sides[0], sides[1], sides[2]};
    result.k1 = Vec3{0.0, 0.0, sides[0]};
    result.k2 = Vec3{sides[1] * sine, 0.0, sides[1] * mu};
    result.k3 = negate(add(result.k1, result.k2));
    return result;
}

CanonicalTriangle canonicalize_closed_vectors(const std::array<Vec3, 3>& vectors) {
    std::array<double, 3> lengths = {};
    std::array<double, 3> ordering_lengths = {};
    double scale = 1.0;
    for (std::size_t index = 0; index < vectors.size(); ++index) {
        lengths[index] = norm(vectors[index]);
        if (!(lengths[index] > 0.0) || !std::isfinite(lengths[index])) {
            throw std::invalid_argument("triangle vectors must be finite and nonzero");
        }
        ordering_lengths[index] = std::round(lengths[index] * 1.0e12) / 1.0e12;
        scale = std::max(scale, lengths[index]);
    }
    const Vec3 closure = add(add(vectors[0], vectors[1]), vectors[2]);
    if (norm(closure) > 2.0e-11 * scale) {
        throw std::invalid_argument("triangle vectors do not close");
    }

    std::array<std::size_t, 3> order = {0, 1, 2};
    std::stable_sort(
        order.begin(), order.end(),
        [&ordering_lengths](std::size_t left, std::size_t right) {
            return ordering_lengths[left] < ordering_lengths[right];
        });
    const std::array<double, 3> sorted_lengths = {
        lengths[order[0]], lengths[order[1]], lengths[order[2]]};
    if (sorted_lengths[0] + sorted_lengths[1]
        < sorted_lengths[2] - 2.0e-11 * scale) {
        throw std::invalid_argument("triangle inequality is not satisfied");
    }

    CanonicalTriangle result;
    result.sides = Triangle{
        sorted_lengths[0], sorted_lengths[1], sorted_lengths[2]};
    result.k1 = vectors[order[0]];
    result.k2 = vectors[order[1]];
    result.k3 = vectors[order[2]];
    return result;
}

double reconstruction_window(const Vec3& momentum, const ReconstructionConfig& config) {
    const double gaussian = std::exp(
        -0.5 * dot(momentum, momentum) * config.smoothing_radius * config.smoothing_radius);
    if (config.cell_size == 0.0) return gaussian;
    const double half_cell = 0.5 * config.cell_size;
    const double wx = sinc(momentum.x * half_cell);
    const double wy = sinc(momentum.y * half_cell);
    const double wz = sinc(momentum.z * half_cell);
    return gaussian * integer_power(wx, 4) * integer_power(wy, 4) * integer_power(wz, 4);
}

double reconstruction_bias_denominator(
    const Vec3& block_momentum,
    const ReconstructionConfig& config) {
    if (!(config.bias_recon>0.0)
        ||!std::isfinite(config.bias_recon)
        ||!(config.local_png_bias_kmin>=0.0)
        ||!std::isfinite(config.local_png_bias_kmin)
        ||!std::isfinite(config.local_png_bias_amplitude)) {
        throw std::invalid_argument(
            "invalid reconstruction-bias denominator configuration");
    }
    const double q=norm(block_momentum);
    double result=config.bias_recon;
    if (config.local_png_bias_transfer!=nullptr
        &&config.local_png_bias_amplitude!=0.0
        &&q>=config.local_png_bias_kmin) {
        const double transfer=
            (*(config.local_png_bias_transfer))(q);
        if (!(transfer>0.0) ||!std::isfinite(transfer)) {
            throw std::domain_error(
                "local-PNG reconstruction denominator sampled invalid M(k)");
        }
        result+=config.local_png_bias_amplitude/transfer;
    }
    if (!(result>0.0) ||!std::isfinite(result)) {
        throw std::domain_error(
            "local-PNG reconstruction denominator is non-positive");
    }
    return result;
}

double reconstruction_shift_factor(
    const Vec3& output_momentum,
    const Vec3& block_momentum,
    const ReconstructionConfig& config) {
    const double denominator = dot(block_momentum, block_momentum);
    if (denominator <= kZeroTolerance * kZeroTolerance) return 0.0;
    return -dot(output_momentum, block_momentum) / denominator
           * reconstruction_window(block_momentum, config)
           /reconstruction_bias_denominator(
               block_momentum,config);
}

ReconstructionShiftVariation
reconstruction_shift_factor_local_png_variation(
    const Vec3& output_momentum,
    const Vec3& block_momentum,
    const ReconstructionConfig& fixed_config,
    const PowerSpectrum& transfer,
    double kmin) {
    if (fixed_config.local_png_bias_transfer!=nullptr
        ||fixed_config.local_png_bias_amplitude!=0.0
        ||!(kmin>=0.0) ||!std::isfinite(kmin)) {
        throw std::invalid_argument(
            "local-PNG shift variation requires a fixed base configuration");
    }
    ReconstructionShiftVariation result;
    result.value=reconstruction_shift_factor(
        output_momentum,block_momentum,fixed_config);
    const double q=norm(block_momentum);
    if (!(q>=kmin) ||result.value==0.0) return result;
    const double transfer_value=transfer(q);
    if (!(transfer_value>0.0)
        ||!std::isfinite(transfer_value)) {
        throw std::domain_error(
            "local-PNG shift variation sampled invalid M(k)");
    }
    result.local_png_denominator_direction=
        -result.value/(fixed_config.bias_recon*transfer_value);
    return result;
}

KernelTemplate pre_reconstruction_kernel(const std::vector<Vec3>& momenta) {
    if (momenta.empty() || momenta.size() > 4) {
        throw std::invalid_argument("halo kernel order must be one through four");
    }
    KernelTemplate result = multiply_kernel(matter_kernel(momenta), Polynomial::variable_b1());
    result = add_kernel(
        std::move(result),
        multiply_kernel(quadratic_density_kernel(momenta), Polynomial::variable_b2()));
    result = add_kernel(
        std::move(result),
        multiply_kernel(quadratic_tidal_kernel(momenta), Polynomial::variable_bK2()));
    return result;
}

KernelTemplate reconstructed_kernel(
    const std::vector<Vec3>& momenta,
    const ReconstructionConfig& config) {
    if (!config.enabled) return pre_reconstruction_kernel(momenta);
    const int order = static_cast<int>(momenta.size());
    if (order < 1 || order > 4) {
        throw std::invalid_argument("reconstructed halo kernel order must be one through four");
    }
    const Vec3 output_momentum = sum_vectors(momenta);
    KernelTemplate result;
    for (const std::vector<std::vector<int>>& partition : set_partitions(order)) {
        double symmetry = 1.0 / factorial(order);
        for (const std::vector<int>& block : partition) {
            symmetry *= factorial(static_cast<int>(block.size()));
        }
        for (int density_index = 0;
             density_index < static_cast<int>(partition.size());
             ++density_index) {
            const std::vector<Vec3> density_vectors =
                block_vectors(momenta, partition[density_index]);
            if (is_zero_vector(sum_vectors(density_vectors))) continue;
            KernelTemplate term = pre_reconstruction_kernel(density_vectors);
            bool vanishes = false;
            for (int block_index = 0;
                 block_index < static_cast<int>(partition.size());
                 ++block_index) {
                if (block_index == density_index) continue;
                const std::vector<Vec3> shift_vectors =
                    block_vectors(momenta, partition[block_index]);
                const double shift = reconstruction_shift_factor(
                    output_momentum, sum_vectors(shift_vectors), config);
                if (shift == 0.0) {
                    vanishes = true;
                    break;
                }
                term = multiply_kernel(term, pre_reconstruction_kernel(shift_vectors));
                term = scale_kernel(std::move(term), shift);
            }
            if (!vanishes) result = add_kernel(std::move(result), scale_kernel(std::move(term), symmetry));
        }
    }
    return result;
}

ComponentTemplates compute_templates(
    const PowerSpectrum& linear_power,
    const Triangle& triangle,
    const IntegrationConfig& config) {
    const auto identity = [](const Polynomial& polynomial) { return polynomial; };
    const CoreResult<Polynomial> core = compute_core(
        linear_power, canonicalize_triangle(triangle), config, identity);
    ComponentTemplates result;
    result.triangle = core.triangle;
    result.tree = core.tree;
    result.B222 = core.B222;
    result.B321I = core.B321I;
    result.B321II = core.B321II;
    result.B411 = core.B411;
    result.one_loop = core.one_loop;
    result.total = core.total;
    result.stochastic_alpha3_raw = core.stochastic_alpha3_raw;
    result.loop_nodes = core.loop_nodes;
    return result;
}

ComponentTemplates compute_templates_vectors(
    const PowerSpectrum& linear_power,
    const std::array<Vec3, 3>& closed_vectors,
    const IntegrationConfig& config) {
    const auto identity = [](const Polynomial& polynomial) { return polynomial; };
    const CoreResult<Polynomial> core = compute_core(
        linear_power, canonicalize_closed_vectors(closed_vectors), config, identity);
    ComponentTemplates result;
    result.triangle = core.triangle;
    result.tree = core.tree;
    result.B222 = core.B222;
    result.B321I = core.B321I;
    result.B321II = core.B321II;
    result.B411 = core.B411;
    result.one_loop = core.one_loop;
    result.total = core.total;
    result.stochastic_alpha3_raw = core.stochastic_alpha3_raw;
    result.loop_nodes = core.loop_nodes;
    return result;
}

ComponentValues compute_direct(
    const PowerSpectrum& linear_power,
    const Triangle& triangle,
    const IntegrationConfig& config,
    const BiasPoint& bias) {
    const auto evaluate = [&bias](const Polynomial& polynomial) {
        return polynomial.evaluate(bias);
    };
    const CoreResult<double> core = compute_core(
        linear_power, canonicalize_triangle(triangle), config, evaluate);
    ComponentValues result;
    result.triangle = core.triangle;
    result.tree = core.tree;
    result.B222 = core.B222;
    result.B321I = core.B321I;
    result.B321II = core.B321II;
    result.B411 = core.B411;
    result.one_loop = core.one_loop;
    result.total = core.total;
    result.stochastic_alpha3_raw = core.stochastic_alpha3_raw;
    result.loop_nodes = core.loop_nodes;
    return result;
}

ComponentValues compute_direct_vectors(
    const PowerSpectrum& linear_power,
    const std::array<Vec3, 3>& closed_vectors,
    const IntegrationConfig& config,
    const BiasPoint& bias) {
    const auto evaluate = [&bias](const Polynomial& polynomial) {
        return polynomial.evaluate(bias);
    };
    const CoreResult<double> core = compute_core(
        linear_power, canonicalize_closed_vectors(closed_vectors), config, evaluate);
    ComponentValues result;
    result.triangle = core.triangle;
    result.tree = core.tree;
    result.B222 = core.B222;
    result.B321I = core.B321I;
    result.B321II = core.B321II;
    result.B411 = core.B411;
    result.one_loop = core.one_loop;
    result.total = core.total;
    result.stochastic_alpha3_raw = core.stochastic_alpha3_raw;
    result.loop_nodes = core.loop_nodes;
    return result;
}

ComponentValues evaluate_templates(
    const ComponentTemplates& templates,
    const BiasPoint& bias) {
    ComponentValues result;
    result.triangle = templates.triangle;
    result.tree = templates.tree.evaluate(bias);
    result.B222 = templates.B222.evaluate(bias);
    result.B321I = templates.B321I.evaluate(bias);
    result.B321II = templates.B321II.evaluate(bias);
    result.B411 = templates.B411.evaluate(bias);
    result.one_loop = templates.one_loop.evaluate(bias);
    result.total = templates.total.evaluate(bias);
    result.stochastic_alpha3_raw = templates.stochastic_alpha3_raw.evaluate(bias);
    result.stochastic_alpha4_raw = templates.stochastic_alpha4_raw;
    result.loop_nodes = templates.loop_nodes;
    return result;
}

double add_residual_stochastic(
    double deterministic_total,
    double alpha3_raw_basis,
    double alpha4_raw_basis,
    double alpha3,
    double alpha4,
    double number_density) {
    if (!(number_density > 0.0)) throw std::invalid_argument("number density must be positive");
    return deterministic_total
           + alpha3 * alpha3_raw_basis / number_density
           + alpha4 * alpha4_raw_basis / (number_density * number_density);
}

}  // namespace marisa_b_halo_v1
