/*
 * MARISA-B native 核心执行大纲：
 * 1. 基础向量与 SPT F_n kernel；
 * 2. 冻结的 dark-matter pre/post reconstruction kernel；
 * 3. 独立的 halo bias-v1 Gaussian kernel K_n=b1 F_n+b2 D_n+bK2 T_n；
 * 4. tree、B222、B321I、B321II/B123I、B411/B114 的 loop 积分；
 * 5. 冻结的 local-PNG matter response（旧模式）；
 * 6. halo bias-v1 的 A_n/C_n/KIC_n 响应、reconstruction product rule 与
 *    tree+B222+B321I+B321II+B411 的 fixed-cutoff truncated 一环入口。
 *
 * halo bias-v1 走独立函数，不替换旧 DM 函数，因此旧模式的数值调用路径
 * 保持不变。新增函数的参数与返回值说明写在各自定义上方。
 */

#include "marisa_b_native.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "PowerSpectrum.h"
#include "Quadrature.h"

namespace marisa_b_native {
namespace {

const real kPi = 3.141592653589793238462643383279502884;
const real kTwoPi = 2.0 * kPi;
const real kReactXmin = -9.999561e-01;
const real kTiny = 1.0e-24;

real sqr(real x) { return x * x; }

real sqrt_nonneg(real x) {
    return std::sqrt(std::max<real>(0.0, x));
}

real p_safe(const PowerSpectrum& P_L, real k) {
    return P_L(std::max<real>(k, 1.0e-12));
}

struct Vec3 {
    real x = 0.0;
    real y = 0.0;
    real z = 0.0;
};

Vec3 make_vec(real x, real y, real z) {
    Vec3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

Vec3 add_vec(const Vec3& a, const Vec3& b) {
    return make_vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 sub_vec(const Vec3& a, const Vec3& b) {
    return make_vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 neg_vec(const Vec3& a) {
    return make_vec(-a.x, -a.y, -a.z);
}

real dot_vec(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

real norm_vec(const Vec3& a) {
    return std::sqrt(std::max<real>(0.0, dot_vec(a, a)));
}

Vec3 sum_vecs(const std::vector<Vec3>& vectors) {
    Vec3 out;
    for (std::vector<Vec3>::const_iterator it = vectors.begin(); it != vectors.end(); ++it) {
        out = add_vec(out, *it);
    }
    return out;
}

std::vector<Vec3> triangle_vecs(const Triangle& tri) {
    const real mu = std::max<real>(-1.0, std::min<real>(1.0, tri.mu12));
    const real sint = std::sqrt(std::max<real>(0.0, 1.0 - mu * mu));
    std::vector<Vec3> out;
    out.push_back(make_vec(0.0, 0.0, tri.k1));
    out.push_back(make_vec(tri.k2 * sint, 0.0, tri.k2 * mu));
    out.push_back(neg_vec(add_vec(out[0], out[1])));
    return out;
}

std::vector<Vec3> triangle_vecs(
    const ClosedTriangleVectors& vectors) {
    std::vector<Vec3> out;
    out.reserve(3);
    for (const auto& vector : vectors) {
        out.push_back(
            make_vec(vector[0], vector[1], vector[2]));
    }
    return out;
}

Triangle triangle_from_vecs(
    const std::vector<Vec3>& vectors) {
    Triangle out;
    if (vectors.size() != 3) return out;
    out.k1 = norm_vec(vectors[0]);
    out.k2 = norm_vec(vectors[1]);
    const real denominator = out.k1 * out.k2;
    out.mu12 = denominator > 0.0
        ? dot_vec(vectors[0], vectors[1]) / denominator
        : 0.0;
    out.mu12 = std::max<real>(
        -1.0, std::min<real>(1.0, out.mu12));
    return out;
}

void validate_closed_triangle_vecs(
    const std::vector<Vec3>& vectors,
    real singular_floor) {
    if (vectors.size() != 3) {
        throw std::invalid_argument(
            "closed triangle vector API requires exactly three legs");
    }
    real scale = 0.0;
    for (const Vec3& vector : vectors) {
        if (!std::isfinite(vector.x)
            || !std::isfinite(vector.y)
            || !std::isfinite(vector.z)) {
            throw std::invalid_argument(
                "closed triangle vector API received a non-finite leg");
        }
        const real length = norm_vec(vector);
        if (!(length > singular_floor)) {
            throw std::invalid_argument(
                "closed triangle vector API received a zero external leg");
        }
        scale = std::max(scale, length);
    }
    const Vec3 closure = sum_vecs(vectors);
    const real tolerance =
        4096.0 * std::numeric_limits<real>::epsilon()
        * std::max<real>(scale, 1.0);
    if (norm_vec(closure) > tolerance) {
        throw std::invalid_argument(
            "closed triangle vector API received non-closing legs");
    }
}

/*
 * A spherical hard cutoff on the native diagram variable q is not invariant
 * under the loop-momentum shifts induced by relabelling the three external
 * legs.  New halo-v1 modes therefore choose one deterministic representative:
 * the two shortest sides are k1/k2 and the longest side closes the triangle.
 * This makes the regulated functional permutation invariant without averaging
 * six separately regulated integrals.  Frozen legacy DM modes do not call this
 * helper and retain their historical routing bit for bit.
 */
Triangle canonical_halo_v1_loop_triangle(const Triangle& tri) {
    std::vector<real> sides;
    sides.push_back(tri.k1);
    sides.push_back(tri.k2);
    sides.push_back(std::sqrt(std::max<real>(
        0.0,
        pow2(tri.k1) + pow2(tri.k2) + 2.0 * tri.k1 * tri.k2 * tri.mu12)));
    std::sort(sides.begin(), sides.end());
    Triangle routed;
    routed.k1 = sides[0];
    routed.k2 = sides[1];
    const real denominator = 2.0 * routed.k1 * routed.k2;
    routed.mu12 = denominator > 0.0
        ? (pow2(sides[2]) - pow2(routed.k1) - pow2(routed.k2)) / denominator
        : tri.mu12;
    routed.mu12 = std::max<real>(-1.0, std::min<real>(1.0, routed.mu12));
    return routed;
}

Vec3 q_vec(real q, real mu_q1, real phi) {
    const real sinq = std::sqrt(std::max<real>(0.0, 1.0 - mu_q1 * mu_q1));
    return make_vec(q * sinq * std::cos(phi), q * sinq * std::sin(phi), q * mu_q1);
}

real cosine_vecs(const Vec3& a, const Vec3& b) {
    const real na = norm_vec(a);
    const real nb = norm_vec(b);
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot_vec(a, b) / (na * nb);
}

real f2_vec(const Vec3& a, const Vec3& b) {
    return F2eds(norm_vec(a), norm_vec(b), cosine_vecs(a, b));
}

real alpha_kernel_vec(const Vec3& left, const Vec3& right) {
    const real denom = dot_vec(left, left);
    if (denom <= 0.0) return 0.0;
    return dot_vec(add_vec(left, right), left) / denom;
}

real beta_kernel_vec(const Vec3& left, const Vec3& right) {
    const real left2 = dot_vec(left, left);
    const real right2 = dot_vec(right, right);
    if (left2 <= 0.0 || right2 <= 0.0) return 0.0;
    const Vec3 total = add_vec(left, right);
    return dot_vec(total, total) * dot_vec(left, right) / (2.0 * left2 * right2);
}

real spt_unsym_vec(char kind, const std::vector<Vec3>& vectors) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return 1.0;
    const real denom = (2.0 * n + 3.0) * (n - 1.0);
    real total = 0.0;
    for (int m = 1; m < n; ++m) {
        std::vector<Vec3> left(vectors.begin(), vectors.begin() + m);
        std::vector<Vec3> right(vectors.begin() + m, vectors.end());
        const Vec3 k_left = sum_vecs(left);
        const Vec3 k_right = sum_vecs(right);
        const real g_left = spt_unsym_vec('G', left);
        const real f_right = spt_unsym_vec('F', right);
        const real g_right = spt_unsym_vec('G', right);
        const real alpha = alpha_kernel_vec(k_left, k_right);
        const real beta = beta_kernel_vec(k_left, k_right);
        if (kind == 'F') {
            total += g_left * ((2.0 * n + 1.0) * alpha * f_right + 2.0 * beta * g_right) / denom;
        } else {
            total += g_left * (3.0 * alpha * f_right + 2.0 * n * beta * g_right) / denom;
        }
    }
    return total;
}

real spt_f_sym_vec(const std::vector<Vec3>& vectors) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return 1.0;
    std::vector<int> order;
    for (int i = 0; i < n; ++i) order.push_back(i);
    real total = 0.0;
    int count = 0;
    do {
        std::vector<Vec3> ordered;
        for (std::vector<int>::const_iterator it = order.begin(); it != order.end(); ++it) {
            ordered.push_back(vectors[static_cast<std::size_t>(*it)]);
        }
        total += spt_unsym_vec('F', ordered);
        ++count;
    } while (std::next_permutation(order.begin(), order.end()));
    return count > 0 ? total / static_cast<real>(count) : 0.0;
}

/*
 * 对称速度散度 kernel G_n。
 *
 * advected PNG operator phi(q) 的递推式需要 G_m，而 Gaussian halo kernel
 * 只需要 F_m。这里复用同一个 EdS 非对称递推并对全部排列平均；n<=3，
 * 因而额外开销有界。零和子集由 alpha/beta 的安全分支处理。
 */
real spt_g_sym_vec(const std::vector<Vec3>& vectors) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return 1.0;
    std::vector<int> order;
    for (int i = 0; i < n; ++i) order.push_back(i);
    real total = 0.0;
    int count = 0;
    do {
        std::vector<Vec3> ordered;
        for (std::vector<int>::const_iterator it = order.begin(); it != order.end(); ++it) {
            ordered.push_back(vectors[static_cast<std::size_t>(*it)]);
        }
        total += spt_unsym_vec('G', ordered);
        ++count;
    } while (std::next_permutation(order.begin(), order.end()));
    return count > 0 ? total / static_cast<real>(count) : 0.0;
}

real f3_vec(const Vec3& a, const Vec3& b, const Vec3& c) {
    const real ka = norm_vec(a);
    const real kb = norm_vec(b);
    const real kc = norm_vec(c);
    if (std::min(std::min(ka, kb), kc) <= 0.0) return 0.0;
    const real kbc = norm_vec(add_vec(b, c));
    const real kab = norm_vec(add_vec(a, b));
    const real kac = norm_vec(add_vec(a, c));
    if (std::min(std::min(kbc, kab), kac) <= 1.0e-12) {
        std::vector<Vec3> fallback;
        fallback.push_back(a);
        fallback.push_back(b);
        fallback.push_back(c);
        return spt_f_sym_vec(fallback);
    }
    return F3edsb(ka, kb, kc, kbc, kab, kac, cosine_vecs(b, c), cosine_vecs(a, b), cosine_vecs(a, c));
}

real f4_loop_pair_vec(const Vec3& a, const Vec3& b, const Vec3& q) {
    const real ka = norm_vec(a);
    const real kb = norm_vec(b);
    const real kq = norm_vec(q);
    if (std::min(std::min(ka, kb), kq) <= 0.0) return 0.0;
    return F4edsb(
        ka,
        kb,
        kq,
        kq,
        cosine_vecs(b, neg_vec(q)),
        cosine_vecs(b, q),
        cosine_vecs(a, b),
        cosine_vecs(a, neg_vec(q)),
        cosine_vecs(a, q));
}

real pre_kernel_vec(const std::vector<Vec3>& vectors) {
    const std::size_t n = vectors.size();
    if (n == 1) return 1.0;
    if (n == 2) return f2_vec(vectors[0], vectors[1]);
    if (n == 3) return f3_vec(vectors[0], vectors[1], vectors[2]);
    if (n == 4) {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = i + 1; j < 4; ++j) {
                if (norm_vec(add_vec(vectors[i], vectors[j])) <= 1.0e-10) {
                    std::vector<Vec3> others;
                    for (std::size_t m = 0; m < 4; ++m) {
                        if (m != i && m != j) others.push_back(vectors[m]);
                    }
                    return f4_loop_pair_vec(others[0], others[1], vectors[i]);
                }
            }
        }
        return spt_f_sym_vec(vectors);
    }
    return 0.0;
}

/*
 * 计算 tidal operator K^2 的二阶角核 S2(a,b)=mu^2-1/3。
 * 参数 a、b 是 Fourier 波矢；任一零模时返回 0。返回值无量纲。
 */
real tidal_s2_vec(const Vec3& a, const Vec3& b) {
    const real na = norm_vec(a);
    const real nb = norm_vec(b);
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    const real mu = dot_vec(a, b) / (na * nb);
    return mu * mu - 1.0 / 3.0;
}

struct HaloKernelComponents {
    real matter = 0.0;
    real delta2 = 0.0;
    real tidal2 = 0.0;
};

/*
 * 把固定的 Eulerian bias functional 扰动展开到 n<=4。
 *
 * 输入 vectors 是进入同一个对称 kernel 的 n 个线性波矢；输出分别是
 * F_n、[delta^2/2]_n 与 [K^2]_n 的系数。D_n/T_n 与现有 Python 诊断
 * fit_quijote_halo_pre_recon_bias1loop_diag.py 使用完全相同的组合公式。
 */
HaloKernelComponents halo_bias_v1_components_vec(const std::vector<Vec3>& vectors) {
    HaloKernelComponents out;
    const std::size_t n = vectors.size();
    out.matter = pre_kernel_vec(vectors);
    if (n == 1) return out;
    if (n == 2) {
        out.delta2 = 0.5;
        out.tidal2 = tidal_s2_vec(vectors[0], vectors[1]);
        return out;
    }
    if (n == 3) {
        real f2_sum = 0.0;
        real tidal_sum = 0.0;
        for (std::size_t single = 0; single < 3; ++single) {
            std::vector<Vec3> pair;
            for (std::size_t index = 0; index < 3; ++index) {
                if (index != single) pair.push_back(vectors[index]);
            }
            const real f2 = pre_kernel_vec(pair);
            f2_sum += f2;
            tidal_sum += tidal_s2_vec(vectors[single], sum_vecs(pair)) * f2;
        }
        out.delta2 = f2_sum / 3.0;
        out.tidal2 = 2.0 * tidal_sum / 3.0;
        return out;
    }
    if (n == 4) {
        real f3_sum = 0.0;
        real tidal_single_sum = 0.0;
        for (std::size_t single = 0; single < 4; ++single) {
            std::vector<Vec3> rest;
            for (std::size_t index = 0; index < 4; ++index) {
                if (index != single) rest.push_back(vectors[index]);
            }
            const real f3 = pre_kernel_vec(rest);
            f3_sum += f3;
            tidal_single_sum += tidal_s2_vec(vectors[single], sum_vecs(rest)) * f3;
        }

        const int pairings[3][4] = {
            {0, 1, 2, 3},
            {0, 2, 1, 3},
            {0, 3, 1, 2},
        };
        real f2_pair_sum = 0.0;
        real tidal_pair_sum = 0.0;
        for (int row = 0; row < 3; ++row) {
            std::vector<Vec3> left;
            left.push_back(vectors[static_cast<std::size_t>(pairings[row][0])]);
            left.push_back(vectors[static_cast<std::size_t>(pairings[row][1])]);
            std::vector<Vec3> right;
            right.push_back(vectors[static_cast<std::size_t>(pairings[row][2])]);
            right.push_back(vectors[static_cast<std::size_t>(pairings[row][3])]);
            const real f2_left = pre_kernel_vec(left);
            const real f2_right = pre_kernel_vec(right);
            f2_pair_sum += f2_left * f2_right;
            tidal_pair_sum += tidal_s2_vec(sum_vecs(left), sum_vecs(right)) * f2_left * f2_right;
        }
        out.delta2 = 0.25 * f3_sum + f2_pair_sum / 6.0;
        out.tidal2 = 0.5 * tidal_single_sum + tidal_pair_sum / 3.0;
        return out;
    }
    return out;
}

/*
 * 组合 halo bias-v1 Gaussian kernel。
 * 输入是 n=1..4 个波矢和固定 bias 参数；输出 K_n。
 * bphi/bphidelta 在 fNL=0 时不进入该函数，留给 PNG 响应层使用。
 */
real halo_bias_v1_kernel_vec(const std::vector<Vec3>& vectors, const HaloBiasV1Params& bias) {
    const HaloKernelComponents components = halo_bias_v1_components_vec(vectors);
    return bias.b1 * components.matter + bias.b2 * components.delta2 + bias.bK2 * components.tidal2;
}

real smoothing_kernel_post(real k, const NativeConfig& config) {
    return std::exp(-0.5 * pow2(k * config.smoothing_radius));
}

real sinc_pi(real x) {
    if (std::fabs(x) < 1.0e-8) return 1.0 - (kPi * kPi * x * x) / 6.0;
    return std::sin(kPi * x) / (kPi * x);
}

real cic_mesh_window_post(const Vec3& vec, const NativeConfig& config) {
    if (config.recon_cellsize <= 0.0 || config.recon_cic_window_power <= 0) return 1.0;
    const real wx = sinc_pi(vec.x * config.recon_cellsize / kTwoPi);
    const real wy = sinc_pi(vec.y * config.recon_cellsize / kTwoPi);
    const real wz = sinc_pi(vec.z * config.recon_cellsize / kTwoPi);
    return std::pow(wx * wy * wz, static_cast<real>(config.recon_cic_window_power));
}

real shift_factor_post(const Vec3& total_vec, const std::vector<Vec3>& block_vecs, const NativeConfig& config) {
    const Vec3 block_sum = sum_vecs(block_vecs);
    const real q2 = dot_vec(block_sum, block_sum);
    if (q2 <= 1.0e-12) return 0.0;
    const real q = std::sqrt(q2);
    const real window = smoothing_kernel_post(q, config) * cic_mesh_window_post(block_sum, config);
    return dot_vec(total_vec, block_sum) / q2 * (-window / config.bias_recon);
}

int factorial_small(int n) {
    int out = 1;
    for (int i = 2; i <= n; ++i) out *= i;
    return out;
}

void partition_rec(
    const std::vector<int>& items,
    std::size_t index,
    std::vector<std::vector<int> >& current,
    std::vector<std::vector<std::vector<int> > >& out) {
    if (index == items.size()) {
        out.push_back(current);
        return;
    }
    const int item = items[index];
    current.push_back(std::vector<int>(1, item));
    partition_rec(items, index + 1, current, out);
    current.pop_back();
    for (std::size_t block = 0; block < current.size(); ++block) {
        current[block].push_back(item);
        partition_rec(items, index + 1, current, out);
        current[block].pop_back();
    }
}

std::vector<std::vector<std::vector<int> > > set_partitions_indices(const std::vector<int>& items) {
    std::vector<std::vector<std::vector<int> > > out;
    std::vector<std::vector<int> > current;
    partition_rec(items, 0, current, out);
    return out;
}

std::vector<Vec3> select_vectors(const std::vector<Vec3>& vectors, const std::vector<int>& labels) {
    std::vector<Vec3> out;
    for (std::vector<int>::const_iterator it = labels.begin(); it != labels.end(); ++it) {
        out.push_back(vectors[static_cast<std::size_t>(*it)]);
    }
    return out;
}

/*
 * 一个 kernel 的 Gaussian 值及其 O(fNL) 线性响应。
 * response 始终表示在 fNL=0 处的导数，不包含 fNL 本身。
 */
struct KernelResponse {
    real gaussian = 0.0;
    real response = 0.0;
};

/* 定义在 reconstruction kernel 之后；此处前置声明供 PNG 递推使用。 */
real p_safe_vec(const PowerSpectrum& P_L, const Vec3& vec);

int bit_count_small(int mask) {
    int count = 0;
    while (mask != 0) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

real binomial_small(int n, int m) {
    if (m < 0 || m > n) return 0.0;
    return static_cast<real>(factorial_small(n))
           / static_cast<real>(factorial_small(m) * factorial_small(n - m));
}

void split_vectors_by_mask(
    const std::vector<Vec3>& vectors,
    int mask,
    std::vector<Vec3>& selected,
    std::vector<Vec3>& complement) {
    selected.clear();
    complement.clear();
    for (std::size_t index = 0; index < vectors.size(); ++index) {
        if (mask & (1 << static_cast<int>(index))) {
            selected.push_back(vectors[index]);
        } else {
            complement.push_back(vectors[index]);
        }
    }
}

real inverse_transfer_vec(const PowerSpectrum& transfer_m, const Vec3& vector) {
    const real transfer = p_safe_vec(transfer_m, vector);
    return std::fabs(transfer) > kTiny ? 1.0 / transfer : 0.0;
}

/*
 * A_n=[phi(q)]_n：把 Lagrangian primordial potential 平流到 Eulerian
 * 位置后的对称 kernel，严格实现
 *
 * A_n = 1/(n-1) sum_m < (K_S.K_T/K_S^2) G_m(S) A_{n-m}(T) >_{|S|=m} .
 *
 * 输入 vectors 是 n 个线性 Gaussian 波矢，transfer_m=M(k,z)。n=2 时本式
 * 化为 mu/2 [k1/(k2 M1)+k2/(k1 M2)]，即 Barreira tree-level
 * b_phi advection 项。K_S=0 的项同时含消失的 bulk-flow kernel；在精确
 * q,-q 配对处按其对称极限置零，避免 0/0 污染积分。
 */
real advected_phi_kernel_vec(
    const std::vector<Vec3>& vectors,
    const PowerSpectrum& transfer_m) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return inverse_transfer_vec(transfer_m, vectors[0]);
    if (n < 1 || n > 4) return 0.0;

    real total = 0.0;
    const int all_masks = 1 << n;
    for (int m = 1; m < n; ++m) {
        real subset_sum = 0.0;
        for (int mask = 1; mask < all_masks - 1; ++mask) {
            if (bit_count_small(mask) != m) continue;
            std::vector<Vec3> selected;
            std::vector<Vec3> complement;
            split_vectors_by_mask(vectors, mask, selected, complement);
            const Vec3 k_selected = sum_vecs(selected);
            const Vec3 k_complement = sum_vecs(complement);
            const real selected_norm2 = dot_vec(k_selected, k_selected);
            if (selected_norm2 <= 1.0e-24) continue;
            const real displacement = dot_vec(k_selected, k_complement) / selected_norm2;
            subset_sum += displacement
                          * spt_g_sym_vec(selected)
                          * advected_phi_kernel_vec(complement, transfer_m);
        }
        total += subset_sum / binomial_small(n, m);
    }
    return total / static_cast<real>(n - 1);
}

/*
 * C_n=[phi(q) delta]_n 的对称乘积 kernel。
 * 每个 m 阶 A_m 与 (n-m) 阶 F_{n-m} 对所有子集平均；C_1=0。特别地
 * C_2=(1/M1+1/M2)/2，因而 b_{phi delta} 的 tree 极限与文献完全一致。
 */
real advected_phi_delta_kernel_vec(
    const std::vector<Vec3>& vectors,
    const PowerSpectrum& transfer_m) {
    const int n = static_cast<int>(vectors.size());
    if (n < 2 || n > 4) return 0.0;
    const int all_masks = 1 << n;
    real total = 0.0;
    for (int m = 1; m < n; ++m) {
        real subset_sum = 0.0;
        for (int mask = 1; mask < all_masks - 1; ++mask) {
            if (bit_count_small(mask) != m) continue;
            std::vector<Vec3> selected;
            std::vector<Vec3> complement;
            split_vectors_by_mask(vectors, mask, selected, complement);
            subset_sum += advected_phi_kernel_vec(selected, transfer_m)
                          * pre_kernel_vec(complement);
        }
        total += subset_sum / binomial_small(n, m);
    }
    return total;
}

/*
 * local-PNG 初始条件的 quadratic response。
 *
 * Q_ij=M(|ki+kj|)/(Mi Mj)，并把合并后的 ki+kj 作为一个输入送入
 * Gaussian halo K_{n-1}。系数 (2/n) 来自对称 kernel 约定。精确反向
 * 波矢的合并模为零且 M(0)=0，这一项显式置零以避免对插值表做无意义的
 * k=0 外推。
 */
real local_ic_response_kernel_vec(
    const std::vector<Vec3>& vectors,
    const PowerSpectrum& transfer_m,
    const HaloBiasV1Params& bias) {
    const int n = static_cast<int>(vectors.size());
    if (n < 2 || n > 4) return 0.0;
    real total = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const Vec3 combined = add_vec(vectors[static_cast<std::size_t>(i)], vectors[static_cast<std::size_t>(j)]);
            if (norm_vec(combined) <= 1.0e-12) continue;
            const real mi = p_safe_vec(transfer_m, vectors[static_cast<std::size_t>(i)]);
            const real mj = p_safe_vec(transfer_m, vectors[static_cast<std::size_t>(j)]);
            if (std::fabs(mi) <= kTiny || std::fabs(mj) <= kTiny) continue;
            const real qij = p_safe_vec(transfer_m, combined) / (mi * mj);
            std::vector<Vec3> gaussian_args;
            gaussian_args.push_back(combined);
            for (int index = 0; index < n; ++index) {
                if (index != i && index != j) {
                    gaussian_args.push_back(vectors[static_cast<std::size_t>(index)]);
                }
            }
            total += qij * halo_bias_v1_kernel_vec(gaussian_args, bias);
        }
    }
    return (2.0 / static_cast<real>(n)) * total;
}

/*
 * pre-reconstruction halo bias-v1 的双 kernel。
 * Gaussian 部分仍是已经冻结的 K_n=b1 F_n+b2 D_n+bK2 T_n；PNG 响应是
 * local 初始条件响应加 b_phi A_n+b_{phi delta} C_n。这里只保留对 fNL
 * 的一阶项，因此不会产生任意 PNG 参数之间的乘积。
 */
KernelResponse halo_bias_v1_local_png_kernel_response_vec(
    const std::vector<Vec3>& vectors,
    const PowerSpectrum& transfer_m,
    const HaloBiasV1Params& bias) {
    KernelResponse out;
    out.gaussian = halo_bias_v1_kernel_vec(vectors, bias);
    out.response = local_ic_response_kernel_vec(vectors, transfer_m, bias)
                   + bias.bphi * advected_phi_kernel_vec(vectors, transfer_m)
                   + bias.bphidelta * advected_phi_delta_kernel_vec(vectors, transfer_m);
    return out;
}

real zrec_kernel_vec(const std::vector<Vec3>& vectors, const NativeConfig& config) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return 1.0;
    if (n < 1 || n > 4) return 0.0;
    const Vec3 total_vec = sum_vecs(vectors);
    const int nmask = 1 << n;
    real total = 0.0;
    for (int mask = 1; mask < nmask; ++mask) {
        std::vector<int> density_labels;
        std::vector<int> remaining_labels;
        for (int label = 0; label < n; ++label) {
            if (mask & (1 << label)) {
                density_labels.push_back(label);
            } else {
                remaining_labels.push_back(label);
            }
        }
        const std::vector<Vec3> density_vectors = select_vectors(vectors, density_labels);
        const real density_kernel = pre_kernel_vec(density_vectors);
        if (density_kernel == 0.0) continue;
        const std::vector<std::vector<std::vector<int> > > partitions = set_partitions_indices(remaining_labels);
        for (std::vector<std::vector<std::vector<int> > >::const_iterator part = partitions.begin(); part != partitions.end(); ++part) {
            real coeff = static_cast<real>(factorial_small(static_cast<int>(density_labels.size()))) / static_cast<real>(factorial_small(n));
            real term = density_kernel;
            for (std::vector<std::vector<int> >::const_iterator block = part->begin(); block != part->end(); ++block) {
                const std::vector<Vec3> block_vectors = select_vectors(vectors, *block);
                coeff *= static_cast<real>(factorial_small(static_cast<int>(block->size())));
                term *= shift_factor_post(total_vec, block_vectors, config) * pre_kernel_vec(block_vectors);
                if (term == 0.0) break;
            }
            total += coeff * term;
        }
    }
    return total;
}

/*
 * 将 halo K_n 映射到标准 reconstruction 后的 Z_rec,n。
 *
 * vectors: 当前 n 阶 kernel 的线性波矢，支持 n=1..4；
 * config: R、b_rec 与可选 mesh window；
 * bias: pre-recon halo kernel 的同一组物理参数；
 * 返回: reconstructed halo kernel。位移场来自 delta_h/b_rec，因此 density
 * block 与 shifted block 都使用同一个 halo K_n，这与当前 Python post 诊断一致。
 */
real halo_bias_v1_zrec_kernel_vec(
    const std::vector<Vec3>& vectors,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const int n = static_cast<int>(vectors.size());
    if (n == 1) return halo_bias_v1_kernel_vec(vectors, bias);
    if (n < 1 || n > 4) return 0.0;
    const Vec3 total_vec = sum_vecs(vectors);
    const int nmask = 1 << n;
    real total = 0.0;
    for (int mask = 1; mask < nmask; ++mask) {
        std::vector<int> density_labels;
        std::vector<int> remaining_labels;
        for (int label = 0; label < n; ++label) {
            if (mask & (1 << label)) {
                density_labels.push_back(label);
            } else {
                remaining_labels.push_back(label);
            }
        }
        const std::vector<Vec3> density_vectors = select_vectors(vectors, density_labels);
        const real density_kernel = halo_bias_v1_kernel_vec(density_vectors, bias);
        if (density_kernel == 0.0) continue;
        const std::vector<std::vector<std::vector<int> > > partitions = set_partitions_indices(remaining_labels);
        for (std::vector<std::vector<std::vector<int> > >::const_iterator part = partitions.begin(); part != partitions.end(); ++part) {
            real coeff = static_cast<real>(factorial_small(static_cast<int>(density_labels.size()))) / static_cast<real>(factorial_small(n));
            real term = density_kernel;
            for (std::vector<std::vector<int> >::const_iterator block = part->begin(); block != part->end(); ++block) {
                const std::vector<Vec3> block_vectors = select_vectors(vectors, *block);
                coeff *= static_cast<real>(factorial_small(static_cast<int>(block->size())));
                term *= shift_factor_post(total_vec, block_vectors, config) * halo_bias_v1_kernel_vec(block_vectors, bias);
                if (term == 0.0) break;
            }
            total += coeff * term;
        }
    }
    return total;
}

/*
 * 将 (K_n^G,K_n^R) 通过与 Gaussian reconstruction 完全相同的 partition
 * 公式映射为 (Z_n^G,Z_n^R)。每个乘积只保留一次 response：density block
 * 响应或任意一个 shift block 响应。shift 的几何因子与 1/b_rec 固定，
 * 因而本函数不会把 reconstruction bias 当作待响应参数。
 */
KernelResponse halo_bias_v1_local_png_zrec_kernel_response_vec(
    const std::vector<Vec3>& vectors,
    const PowerSpectrum& transfer_m,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const int n = static_cast<int>(vectors.size());
    if (n < 1 || n > 4) return KernelResponse();
    if (n == 1) return halo_bias_v1_local_png_kernel_response_vec(vectors, transfer_m, bias);

    const Vec3 total_vec = sum_vecs(vectors);
    const int nmask = 1 << n;
    KernelResponse total;
    for (int mask = 1; mask < nmask; ++mask) {
        std::vector<int> density_labels;
        std::vector<int> remaining_labels;
        for (int label = 0; label < n; ++label) {
            if (mask & (1 << label)) {
                density_labels.push_back(label);
            } else {
                remaining_labels.push_back(label);
            }
        }
        const std::vector<Vec3> density_vectors = select_vectors(vectors, density_labels);
        const KernelResponse density =
            halo_bias_v1_local_png_kernel_response_vec(density_vectors, transfer_m, bias);
        if (density.gaussian == 0.0 && density.response == 0.0) continue;

        const std::vector<std::vector<std::vector<int> > > partitions =
            set_partitions_indices(remaining_labels);
        for (std::vector<std::vector<std::vector<int> > >::const_iterator part = partitions.begin();
             part != partitions.end(); ++part) {
            real coeff = static_cast<real>(factorial_small(static_cast<int>(density_labels.size())))
                         / static_cast<real>(factorial_small(n));
            KernelResponse term = density;
            for (std::vector<std::vector<int> >::const_iterator block = part->begin();
                 block != part->end(); ++block) {
                const std::vector<Vec3> block_vectors = select_vectors(vectors, *block);
                coeff *= static_cast<real>(factorial_small(static_cast<int>(block->size())));
                const real shift_geometry = shift_factor_post(total_vec, block_vectors, config);
                const KernelResponse block_kernel =
                    halo_bias_v1_local_png_kernel_response_vec(block_vectors, transfer_m, bias);
                const real previous_gaussian = term.gaussian;
                const real previous_response = term.response;
                term.gaussian = previous_gaussian * shift_geometry * block_kernel.gaussian;
                term.response = shift_geometry
                                * (previous_response * block_kernel.gaussian
                                   + previous_gaussian * block_kernel.response);
            }
            total.gaussian += coeff * term.gaussian;
            total.response += coeff * term.response;
        }
    }
    return total;
}

real p_safe_vec(const PowerSpectrum& P_L, const Vec3& vec) {
    return p_safe(P_L, norm_vec(vec));
}

real p13_shape(real r) {
    if (r <= 0.0) return 0.0;
    if (r < 1.0e-2) {
        return -168.0 + (928.0 / 5.0) * pow2(r) - (4512.0 / 35.0) * pow4(r) + (416.0 / 21.0) * pow6(r);
    }
    if (std::fabs(r - 1.0) < 1.0e-10) {
        return -88.0 + 8.0 * (r - 1.0);
    }
    if (r > 100.0) {
        return -488.0 / 5.0 + (96.0 / 5.0) / pow2(r) - (160.0 / 21.0) / pow4(r) - (1376.0 / 1155.0) / pow6(r);
    }
    return 12.0 / pow2(r) - 158.0 + 100.0 * pow2(r) - 42.0 * pow4(r)
           + 3.0 / pow3(r) * pow3(r * r - 1.0) * (7.0 * r * r + 2.0) * std::log((1.0 + r) / std::fabs(1.0 - r));
}

real B222_integrand(const PowerSpectrum& P_L, real k1, real k2, real x, real k3, real r, real u, real o) {
    (void)k3;
    const real k2s = pow2(k2);
    const real p = k1 * r;
    const real ps = pow2(p);
    const real d = std::sqrt(1.0 + r * r - 2.0 * r * u);
    if (d < 1.0e-5) return 0.0;

    const real k1mp = k1 * d;
    const real k2p = u * x - std::sqrt((1.0 - pow2(u)) * (1.0 - pow2(x))) * std::cos(o);
    const real k2pp = std::sqrt(k2s + ps + 2.0 * k2 * p * k2p);
    return 8.0 * pow2(r) * p_safe(P_L, p)
           * (F2eds(r * k1, k1mp, (u - r) / d)
              * F2eds(p, k2pp, -(k2 * k2p + p) / k2pp)
              * F2eds(k1mp, k2pp, (k2 * x + p * u - k2 * r * k2p - p * r) / d / k2pp)
              * p_safe(P_L, k1mp) * p_safe(P_L, k2pp));
}

real B321I_integrand(const PowerSpectrum& P_L, real k1, real k2, real x, real k3, real r, real u, real o) {
    const real k1s = pow2(k1);
    const real k2s = pow2(k2);
    const real p = k1 * r;
    const real ps = pow2(p);
    const real d = std::sqrt(1.0 + r * r - 2.0 * r * u);
    const real d1 = std::sqrt(1.0 + r * r + 2.0 * r * u);
    if (d < 1.0e-5 || d1 < 1.0e-5) return 0.0;

    const real k1mp = k1 * d;
    const real k2p = u * x - std::sqrt((1.0 - pow2(u)) * (1.0 - pow2(x))) * std::cos(o);
    const real k2mp = std::sqrt(k2s + ps - 2.0 * k2 * p * k2p);
    const real k3mp = std::sqrt(k2s + k1s + ps + 2.0 * k1 * k2 * x + 2.0 * k1s * r * u + 2.0 * k2 * p * k2p);

    const real kernel1 = F2eds(r * k1, k1mp, (u - r) / d);
    const real kernel2 = F2eds(r * k1, k2mp, (k2 * k2p - p) / k2mp);
    const real kernel3 = F2eds(r * k1, k3mp, -(k1 * u + k2 * k2p + p) / k3mp);

    return 6.0 * pow2(r) * p_safe(P_L, p)
           * (kernel2 * F3eds(k1, p, k2mp, (k2 * k2p - p) / k2mp, u, (k2 * x - p * u) / k2mp) * p_safe(P_L, k2mp) * p_safe(P_L, k1)
              + kernel3 * F3eds(k1, p, k3mp, -(k1 * u + k2 * k2p + p) / k3mp, u, -(k1 + k2 * x + p * u) / k3mp) * p_safe(P_L, k3mp) * p_safe(P_L, k1)
              + kernel1 * F3eds(k2, p, k1mp, (u - r) / d, k2p, (x - r * k2p) / d) * p_safe(P_L, k1mp) * p_safe(P_L, k2)
              + kernel3 * F3eds(k2, p, k3mp, -(k1 * u + k2 * k2p + p) / k3mp, k2p, -(k2 + k1 * x + p * k2p) / k3mp) * p_safe(P_L, k3mp) * p_safe(P_L, k2)
              + kernel1 * F3eds(k3, p, k1mp, (u - r) / d, -(k1 * u + k2 * k2p) / k3, (-k1 + p * u - k2 * x + k2 * r * k2p) / d / k3) * p_safe(P_L, k1mp) * p_safe(P_L, k3)
              + kernel2 * F3eds(k3, p, k2mp, (k2 * k2p - p) / k2mp, -(k1 * u + k2 * k2p) / k3, -(k1 * k2 * x - k1 * p * u + k2s - k2 * p * k2p) / k3 / k2mp) * p_safe(P_L, k2mp) * p_safe(P_L, k3));
}

real B411_integrand(const PowerSpectrum& P_L, real k1, real k2, real x, real k3, real r, real u, real o) {
    const real p = k1 * r;
    const real k2p = u * x - std::sqrt((1.0 - pow2(u)) * (1.0 - pow2(x))) * std::cos(o);
    return 12.0 * pow2(r) * p_safe(P_L, p)
           * (F4edsb(k1, k2, p, p, -k2p, k2p, x, -u, u) * p_safe(P_L, k1) * p_safe(P_L, k2)
              + F4edsb(k2, k3, p, p, (k1 * u + k2 * k2p) / k3, -(k1 * u + k2 * k2p) / k3, -(k1 * x + k2) / k3, -k2p, k2p) * p_safe(P_L, k2) * p_safe(P_L, k3)
              + F4edsb(k3, k1, p, p, -u, u, -(k1 + k2 * x) / k3, (k1 * u + k2 * k2p) / k3, -(k1 * u + k2 * k2p) / k3) * p_safe(P_L, k3) * p_safe(P_L, k1));
}

real cos_q_b(real x, real u, real o) {
    const real sin_q = sqrt_nonneg(1.0 - pow2(u));
    const real sin_b = sqrt_nonneg(1.0 - pow2(x));
    return u * x - sin_q * sin_b * std::cos(o);
}

real local_png_b0_dfNL(const PowerSpectrum& P_L, const PowerSpectrum& transfer_m, real k1, real k2, real k3, real b1) {
    const real p1 = p_safe(P_L, k1);
    const real p2 = p_safe(P_L, k2);
    const real p3 = p_safe(P_L, k3);
    const real m1 = p_safe(transfer_m, k1);
    const real m2 = p_safe(transfer_m, k2);
    const real m3 = p_safe(transfer_m, k3);
    if (std::min(std::min(m1, m2), m3) <= 0.0) return 0.0;
    return 2.0 * pow3(b1) * (p1 * p2 * m3 / (m1 * m2)
                              + p2 * p3 * m1 / (m2 * m3)
                              + p3 * p1 * m2 / (m3 * m1));
}

real local_png_b0_dfNL_vec(const PowerSpectrum& P_L, const PowerSpectrum& transfer_m, const Vec3& a, const Vec3& b, const Vec3& c, real b1) {
    return local_png_b0_dfNL(P_L, transfer_m, norm_vec(a), norm_vec(b), norm_vec(c), b1);
}

bool local_png_b0_triplet_is_resolved(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const NativeConfig& config) {
    return local_png_primordial_triplet_is_resolved(
        norm_vec(a),norm_vec(b),norm_vec(c),config);
}

real local_png_b0_dfNL_vec_finite_box(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const NativeConfig& config,
    real b1) {
    if (!local_png_b0_triplet_is_resolved(a,b,c,config)) {
        return 0.0;
    }
    return local_png_b0_dfNL_vec(
        P_L,transfer_m,a,b,c,b1);
}

/*
 * 提取 fNL^2 后的 single-source local primordial trispectrum。
 *
 * Sefusatti 2009 Eq. (3):
 *
 *   T_phi/fNL^2 = 4 [Pphi(k1) Pphi(k2)
 *                       {Pphi(|k1+k3|)+Pphi(|k1+k4|)}
 *                     + 5 permutations].
 *
 * 四个 vectors 必须闭合。六个无序 external pairs 各产生两个 exchange
 * channel，共十二项。对 pair 内代表腿的选择不影响结果，因为闭合条件令
 * |ki+ka|/|ki+kb| 与交换 i<->j 后的两条 diagonal 仅互换。
 *
 * png_ir_cutoff 同时作用于四条 external primordial mode 与每一条 exchange
 * propagator。这样 finite-box cutoff 会排除 q->0、k-q->0 以及
 * |q+ki|->0 collapsed channel；只对 loop variable |q| 设置 qmin 不足以
 * 完成这件事。
 */
real local_png_t0_fNL2_coefficient_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const std::vector<Vec3>& vectors,
    const NativeConfig& config) {
    if (vectors.size() != 4) return 0.0;
    const real ir_cutoff =
        config.png_ir_cutoff > 0.0 ? config.png_ir_cutoff : config.qmin;

    real pphi[4] = {0.0, 0.0, 0.0, 0.0};
    real transfer_product = 1.0;
    for (int i = 0; i < 4; ++i) {
        const real k = norm_vec(vectors[static_cast<std::size_t>(i)]);
        if (k < ir_cutoff) return 0.0;
        const real m = p_safe(transfer_m, k);
        if (std::fabs(m) <= kTiny) return 0.0;
        pphi[i] = p_safe(P_L, k) / pow2(m);
        transfer_product *= m;
    }

    const auto exchange_pphi =
        [&P_L, &transfer_m, ir_cutoff](const Vec3& momentum) -> real {
            const real k = norm_vec(momentum);
            if (k < ir_cutoff) return 0.0;
            const real m = p_safe(transfer_m, k);
            if (std::fabs(m) <= kTiny) return 0.0;
            return p_safe(P_L, k) / pow2(m);
        };

    real permutation_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            int remaining[2] = {-1, -1};
            int count = 0;
            for (int label = 0; label < 4; ++label) {
                if (label != i && label != j) remaining[count++] = label;
            }
            const real diagonal_a = exchange_pphi(add_vec(
                vectors[static_cast<std::size_t>(i)],
                vectors[static_cast<std::size_t>(remaining[0])]));
            const real diagonal_b = exchange_pphi(add_vec(
                vectors[static_cast<std::size_t>(i)],
                vectors[static_cast<std::size_t>(remaining[1])]));
            permutation_sum += pphi[i] * pphi[j] * (diagonal_a + diagonal_b);
        }
    }
    return 4.0 * transfer_product * permutation_sum;
}

/* 定义位于通用 loop wrapper 段；这里前置声明供 B112II 诊断使用。 */
template <typename Function>
ComponentStats integrate_loop_component(
    Function func,
    real k1,
    const NativeConfig& config);

real local_png_B112II_fNL2_orientation_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& qv,
    const NativeConfig& config,
    bool post_recon) {
    const Vec3 cmq = sub_vec(c, qv);
    std::vector<Vec3> trispectrum_vectors;
    trispectrum_vectors.reserve(4);
    trispectrum_vectors.push_back(a);
    trispectrum_vectors.push_back(b);
    trispectrum_vectors.push_back(qv);
    trispectrum_vectors.push_back(cmq);
    const real t0 = local_png_t0_fNL2_coefficient_vec(
        P_L, transfer_m, trispectrum_vectors, config);
    if (t0 == 0.0) return 0.0;
    std::vector<Vec3> kernel_args;
    kernel_args.reserve(2);
    kernel_args.push_back(qv);
    kernel_args.push_back(cmq);
    const real kernel = post_recon
        ? zrec_kernel_vec(kernel_args, config)
        : pre_kernel_vec(kernel_args);
    return kernel * t0;
}

/*
 * 一个固定 B112II orientation：
 *
 *   Z1(a) Z1(b) int_q Z2(q,c-q) T0(a,b,q,c-q).
 *
 * DM 的 Z1=1；pre 使用 F2，post 使用 finite-R Z_rec,2。积分仍采用现有
 * d^3q/(2pi)^3 convention，返回的是已经提取 fNL^2 后的系数。
 */
ComponentStats local_png_B112II_fNL2_ordered(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const NativeConfig& config,
    bool post_recon) {
    const real scale = norm_vec(c);
    if (scale <= config.singular_floor) return ComponentStats();
    return integrate_loop_component(
        [&P_L, &transfer_m, &config, a, b, c, scale, post_recon](
            real r,
            real u,
            real o) {
            const Vec3 qv = q_vec(scale * r, u, o);
            return pow2(r)
                   * local_png_B112II_fNL2_orientation_integrand(
                       P_L,
                       transfer_m,
                       a,
                       b,
                       c,
                       qv,
                       config,
                       post_recon);
        },
        scale,
        config);
}

real radical_inverse(unsigned long long index, int base) {
    real inverse_base = 1.0 / static_cast<real>(base);
    real factor = inverse_base;
    real value = 0.0;
    while (index > 0) {
        value += factor * static_cast<real>(index % static_cast<unsigned long long>(base));
        index /= static_cast<unsigned long long>(base);
        factor *= inverse_base;
    }
    return value;
}

/*
 * 多中心低差异 B112II 积分器。
 *
 * 三个 cyclic orientations 的 collapsed surfaces 只会围绕
 * q={0,+/-k1,+/-k2,+/-k3} 出现。每个中心使用局部 log-radius Halton
 * proposal，并以全部 proposal 的等权 mixture density 做 balance-heuristic
 * importance weighting。不同 replicate 对同一 Halton 序列施加固定
 * Cranley-Patterson rotation；其 scatter 只作为数值误差诊断。
 *
 * 这一路径与 adaptive ReACT cubature 并存，专门解决低-k shell 投影时原始
 * 球坐标在 shifted hard-cutoff surfaces 上频繁达到 1e7 evaluation 上限的
 * 问题。外腿先 canonicalize，因此同一物理三角形的六种输入表示使用完全
 * 相同的中心和采样点。
 */
ComponentStats local_png_B112II_fNL2_multicenter_qmc(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const std::vector<Vec3>& vecs,
    const NativeConfig& config,
    bool post_recon) {
    ComponentStats out;
    if (vecs.size() != 3) return out;
    const real ir_cutoff =
        config.png_ir_cutoff > 0.0 ? config.png_ir_cutoff : config.qmin;
    const int power = std::max(4, std::min(20, config.png_b112ii_qmc_power));
    const int replicates =
        std::max(2, std::min(32, config.png_b112ii_qmc_replicates));
    const unsigned long long samples_per_component =
        static_cast<unsigned long long>(1) << power;

    std::vector<Vec3> centers;
    const auto add_unique_center = [&centers](const Vec3& candidate) {
        for (std::vector<Vec3>::const_iterator it = centers.begin();
             it != centers.end();
             ++it) {
            if (norm_vec(sub_vec(candidate, *it)) < 1.0e-14) return;
        }
        centers.push_back(candidate);
    };
    add_unique_center(make_vec(0.0, 0.0, 0.0));
    for (std::size_t i = 0; i < vecs.size(); ++i) {
        add_unique_center(vecs[i]);
        add_unique_center(neg_vec(vecs[i]));
    }
    const int center_count = static_cast<int>(centers.size());
    std::vector<real> radial_min(static_cast<std::size_t>(center_count), ir_cutoff);
    std::vector<real> radial_max(static_cast<std::size_t>(center_count), config.qmax);
    std::vector<real> radial_log_span(static_cast<std::size_t>(center_count), 0.0);
    for (int center_index = 0; center_index < center_count; ++center_index) {
        const real center_norm = norm_vec(centers[static_cast<std::size_t>(center_index)]);
        radial_min[static_cast<std::size_t>(center_index)] =
            center_norm < 1.0e-14 ? config.qmin : ir_cutoff;
        radial_max[static_cast<std::size_t>(center_index)] =
            config.qmax + center_norm;
        radial_log_span[static_cast<std::size_t>(center_index)] =
            std::log(radial_max[static_cast<std::size_t>(center_index)]
                     / radial_min[static_cast<std::size_t>(center_index)]);
    }

    const auto mixture_density =
        [&centers, &radial_min, &radial_max, &radial_log_span, center_count](
            const Vec3& qv) -> real {
            real density = 0.0;
            for (int center_index = 0; center_index < center_count; ++center_index) {
                const real radius = norm_vec(sub_vec(
                    qv, centers[static_cast<std::size_t>(center_index)]));
                if (radius < radial_min[static_cast<std::size_t>(center_index)]
                    || radius > radial_max[static_cast<std::size_t>(center_index)]
                    || radius <= 0.0) {
                    continue;
                }
                density +=
                    1.0
                    / (4.0
                       * kPi
                       * radial_log_span[static_cast<std::size_t>(center_index)]
                       * pow3(radius));
            }
            return density / static_cast<real>(center_count);
        };

    const auto summed_integrand =
        [&P_L, &transfer_m, &vecs, &config, post_recon](const Vec3& qv) -> real {
            return local_png_B112II_fNL2_orientation_integrand(
                       P_L,
                       transfer_m,
                       vecs[0],
                       vecs[1],
                       vecs[2],
                       qv,
                       config,
                       post_recon)
                   + local_png_B112II_fNL2_orientation_integrand(
                       P_L,
                       transfer_m,
                       vecs[1],
                       vecs[2],
                       vecs[0],
                       qv,
                       config,
                       post_recon)
                   + local_png_B112II_fNL2_orientation_integrand(
                       P_L,
                       transfer_m,
                       vecs[2],
                       vecs[0],
                       vecs[1],
                       qv,
                       config,
                       post_recon);
        };

    std::vector<real> replicate_values;
    replicate_values.reserve(static_cast<std::size_t>(replicates));
    for (int replicate = 0; replicate < replicates; ++replicate) {
        const real shift_q =
            std::fmod((static_cast<real>(replicate) + 1.0) * 0.6180339887498948482, 1.0);
        const real shift_mu =
            std::fmod((static_cast<real>(replicate) + 1.0) * 0.4142135623730950488, 1.0);
        const real shift_phi =
            std::fmod((static_cast<real>(replicate) + 1.0) * 0.7320508075688772935, 1.0);
        real replicate_value = 0.0;
        for (int center_index = 0; center_index < center_count; ++center_index) {
            real component_sum = 0.0;
            for (unsigned long long sample = 1;
                 sample <= samples_per_component;
                 ++sample) {
                const real uq = std::fmod(radical_inverse(sample, 2) + shift_q, 1.0);
                const real umu = std::fmod(radical_inverse(sample, 3) + shift_mu, 1.0);
                const real uphi = std::fmod(radical_inverse(sample, 5) + shift_phi, 1.0);
                const real radius =
                    radial_min[static_cast<std::size_t>(center_index)]
                    * std::exp(
                        radial_log_span[static_cast<std::size_t>(center_index)] * uq);
                const real mu = -1.0 + 2.0 * umu;
                const real phi = kTwoPi * uphi;
                const Vec3 qv = add_vec(
                    centers[static_cast<std::size_t>(center_index)],
                    q_vec(radius, mu, phi));
                const real qnorm = norm_vec(qv);
                if (qnorm < config.qmin || qnorm > config.qmax) continue;
                const real density = mixture_density(qv);
                if (density <= 0.0) continue;
                component_sum +=
                    summed_integrand(qv) / density / pow3(kTwoPi);
            }
            replicate_value +=
                component_sum / static_cast<real>(samples_per_component);
        }
        replicate_values.push_back(
            replicate_value / static_cast<real>(center_count));
    }

    real mean = 0.0;
    for (std::vector<real>::const_iterator it = replicate_values.begin();
         it != replicate_values.end();
         ++it) {
        mean += *it;
    }
    mean /= static_cast<real>(replicates);
    real squared = 0.0;
    for (std::vector<real>::const_iterator it = replicate_values.begin();
         it != replicate_values.end();
         ++it) {
        squared += pow2(*it - mean);
    }
    out.value = mean;
    out.abserr = std::sqrt(
        squared
        / static_cast<real>(replicates)
        / static_cast<real>(replicates - 1));
    out.neval = static_cast<int>(
        samples_per_component
        * static_cast<unsigned long long>(center_count)
        * static_cast<unsigned long long>(replicates));
    return out;
}

struct LocalPngQmcCenter {
    Vec3 position;
    real minimum_radius = 0.0;
};

/*
 * 通用 local-PNG 多中心低差异积分器。
 *
 * `physical_integrand` 接受 Cartesian q，并且不含 d^3q/(2pi)^3 的
 * measure。每个 proposal 在一个指定 collapsed center 周围按
 * d(log radius) dOmega 采样；balance-heuristic mixture 保证不同
 * center 的重叠区域只计一次。origin proposal 的 support 必须覆盖
 * integrand 的全部非零 spherical q-domain。replicate scatter 是数值
 * 标准误估计，生产层还必须以加密 QMC 独立比较，不能把它解释成统计误差。
 */
template <typename Function>
ComponentStats integrate_local_png_multicenter_qmc(
    Function physical_integrand,
    const std::vector<LocalPngQmcCenter>& requested_centers,
    const NativeConfig& config) {
    ComponentStats out;
    if (!(config.qmin > 0.0)
        || !(config.qmax > config.qmin)
        || requested_centers.empty()) {
        return out;
    }
    const int power =
        std::max(4, std::min(20, config.png_b112ii_qmc_power));
    const int replicates =
        std::max(2, std::min(32, config.png_b112ii_qmc_replicates));
    const unsigned long long samples_per_component =
        static_cast<unsigned long long>(1) << power;

    std::vector<LocalPngQmcCenter> centers;
    centers.reserve(requested_centers.size());
    for (const LocalPngQmcCenter& requested : requested_centers) {
        if (!(requested.minimum_radius > 0.0)
            || !std::isfinite(requested.minimum_radius)
            || !std::isfinite(requested.position.x)
            || !std::isfinite(requested.position.y)
            || !std::isfinite(requested.position.z)) {
            continue;
        }
        bool duplicate = false;
        for (LocalPngQmcCenter& existing : centers) {
            if (norm_vec(sub_vec(
                    requested.position, existing.position)) < 1.0e-14) {
                /*
                 * 较小半径 proposal 有更大的 support；物理 cutoff 仍由
                 * integrand 本身执行，因此合并不会重新引入零模。
                 */
                existing.minimum_radius =
                    std::min(
                        existing.minimum_radius,
                        requested.minimum_radius);
                duplicate = true;
                break;
            }
        }
        if (!duplicate) centers.push_back(requested);
    }
    if (centers.empty()) return out;
    const int center_count = static_cast<int>(centers.size());

    std::vector<real> radial_max(
        static_cast<std::size_t>(center_count), 0.0);
    std::vector<real> radial_log_span(
        static_cast<std::size_t>(center_count), 0.0);
    for (int center_index = 0;
         center_index < center_count;
         ++center_index) {
        const LocalPngQmcCenter& center =
            centers[static_cast<std::size_t>(center_index)];
        radial_max[static_cast<std::size_t>(center_index)] =
            config.qmax + norm_vec(center.position);
        if (!(radial_max[static_cast<std::size_t>(center_index)]
              > center.minimum_radius)) {
            return ComponentStats();
        }
        radial_log_span[static_cast<std::size_t>(center_index)] =
            std::log(
                radial_max[static_cast<std::size_t>(center_index)]
                / center.minimum_radius);
    }

    const auto mixture_density =
        [&centers,
         &radial_max,
         &radial_log_span,
         center_count](const Vec3& qv) -> real {
            real density = 0.0;
            for (int center_index = 0;
                 center_index < center_count;
                 ++center_index) {
                const LocalPngQmcCenter& center =
                    centers[static_cast<std::size_t>(center_index)];
                const real radius =
                    norm_vec(sub_vec(qv, center.position));
                if (radius < center.minimum_radius
                    || radius
                       > radial_max[
                           static_cast<std::size_t>(center_index)]
                    || radius <= 0.0) {
                    continue;
                }
                density +=
                    1.0
                    / (4.0
                       * kPi
                       * radial_log_span[
                           static_cast<std::size_t>(center_index)]
                       * pow3(radius));
            }
            return density / static_cast<real>(center_count);
        };

    std::vector<real> replicate_values;
    replicate_values.reserve(static_cast<std::size_t>(replicates));
    for (int replicate = 0; replicate < replicates; ++replicate) {
        real replicate_value = 0.0;
        for (int center_index = 0;
             center_index < center_count;
             ++center_index) {
            const LocalPngQmcCenter& center =
                centers[static_cast<std::size_t>(center_index)];
            const real sequence_index =
                static_cast<real>(
                    1
                    + replicate
                    + 17 * center_index);
            const real shift_q =
                std::fmod(
                    sequence_index * 0.6180339887498948482,
                    1.0);
            const real shift_mu =
                std::fmod(
                    sequence_index * 0.4142135623730950488,
                    1.0);
            const real shift_phi =
                std::fmod(
                    sequence_index * 0.7320508075688772935,
                    1.0);
            real component_sum = 0.0;
            for (unsigned long long sample = 1;
                 sample <= samples_per_component;
                 ++sample) {
                const real uq =
                    std::fmod(
                        radical_inverse(sample, 2) + shift_q,
                        1.0);
                const real umu =
                    std::fmod(
                        radical_inverse(sample, 3) + shift_mu,
                        1.0);
                const real uphi =
                    std::fmod(
                        radical_inverse(sample, 5) + shift_phi,
                        1.0);
                const real radius =
                    center.minimum_radius
                    * std::exp(
                        radial_log_span[
                            static_cast<std::size_t>(center_index)]
                        * uq);
                const Vec3 qv =
                    add_vec(
                        center.position,
                        q_vec(
                            radius,
                            -1.0 + 2.0 * umu,
                            kTwoPi * uphi));
                const real qnorm = norm_vec(qv);
                if (qnorm < config.qmin
                    || qnorm > config.qmax) {
                    continue;
                }
                const real density = mixture_density(qv);
                if (!(density > 0.0)) continue;
                component_sum +=
                    physical_integrand(qv)
                    / density
                    / pow3(kTwoPi);
            }
            replicate_value +=
                component_sum
                / static_cast<real>(samples_per_component);
        }
        replicate_values.push_back(
            replicate_value / static_cast<real>(center_count));
    }

    real mean = 0.0;
    for (const real value : replicate_values) mean += value;
    mean /= static_cast<real>(replicates);
    real squared = 0.0;
    for (const real value : replicate_values) {
        squared += pow2(value - mean);
    }
    out.value = mean;
    out.abserr =
        std::sqrt(
            squared
            / static_cast<real>(replicates)
            / static_cast<real>(replicates - 1));
    out.neval =
        static_cast<int>(
            samples_per_component
            * static_cast<unsigned long long>(center_count)
            * static_cast<unsigned long long>(replicates));
    return out;
}

real P12_png_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real k,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    (void)o;
    const real p = k * r;
    const real d = std::sqrt(1.0 + r * r - 2.0 * r * u);
    const real kmq = k * d;
    if (p<local_png_ir_cutoff(config)
        ||kmq<local_png_ir_cutoff(config)) {
        return 0.0;
    }
    return 2.0 * pow2(r) * F2eds(p, kmq, (u - r) / d) * local_png_b0_dfNL(P_L, transfer_m, k, p, kmq, b1);
}

real B122II_png_ordered_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    const real p = ka * r;
    const real v = cos_q_b(xab, u, o);
    const real apq2 = ka * ka + p * p + 2.0 * ka * p * u;
    const real bmq2 = kb * kb + p * p - 2.0 * kb * p * v;
    if (apq2 <= 1.0e-10 || bmq2 <= 1.0e-10) return 0.0;
    const real apq = std::sqrt(apq2);
    const real bmq = std::sqrt(bmq2);
    const real png_cutoff=local_png_ir_cutoff(config);
    if (p<png_cutoff ||apq<png_cutoff
        ||bmq<config.qmin) {
        return 0.0;
    }
    const real cos_q_bmq = (kb * v - p) / bmq;
    const real cos_apq_bmq = (ka * kb * xab - ka * p * u + kb * p * v - p * p) / (apq * bmq);
    return 4.0 * pow2(r)
           * F2eds(p, bmq, cos_q_bmq)
           * F2eds(apq, bmq, cos_apq_bmq)
           * local_png_b0_dfNL(P_L, transfer_m, ka, p, apq, b1)
           * p_safe(P_L, bmq);
}

real B113II_png_ordered_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    const real p = ka * r;
    const real v = cos_q_b(xab, u, o);
    const real bmq2 = kb * kb + p * p - 2.0 * kb * p * v;
    if (bmq2 <= 1.0e-10) return 0.0;
    const real bmq = std::sqrt(bmq2);
    const real png_cutoff=local_png_ir_cutoff(config);
    if (p<png_cutoff ||bmq<png_cutoff) return 0.0;
    const real cos_q_bmq = (kb * v - p) / bmq;
    const real cos_a_bmq = (kb * xab - p * u) / bmq;
    return 3.0 * pow2(r)
           * p_safe(P_L, ka)
           * F3eds(ka, p, bmq, cos_q_bmq, u, cos_a_bmq)
           * local_png_b0_dfNL(P_L, transfer_m, kb, p, bmq, b1);
}

real P12_png_post_integrand_physical_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& kvec,
    const NativeConfig& config,
    real b1,
    const Vec3& qv) {
    const Vec3 kmq = sub_vec(kvec, qv);
    if (!local_png_b0_triplet_is_resolved(
            kvec,qv,kmq,config)) {
        return 0.0;
    }
    std::vector<Vec3> z2;
    z2.push_back(qv);
    z2.push_back(kmq);
    return 2.0 * zrec_kernel_vec(z2, config)
           * local_png_b0_dfNL_vec_finite_box(
               P_L,transfer_m,kvec,qv,kmq,config,b1);
}

real P12_png_post_integrand_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& kvec,
    real scale,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    return
        pow2(r)
        * P12_png_post_integrand_physical_vec(
            P_L,
            transfer_m,
            kvec,
            config,
            b1,
            q_vec(scale * r, u, o));
}

real P12_png_post_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real k,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    return P12_png_post_integrand_vec(
        P_L, transfer_m, make_vec(0.0, 0.0, k), k,
        config, b1, r, u, o);
}

real B122II_png_post_ordered_integrand_physical_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    const NativeConfig& config,
    real b1,
    const Vec3& qv) {
    const Vec3 apq = add_vec(avec, qv);
    const Vec3 bmq = sub_vec(bvec, qv);
    if (!local_png_b0_triplet_is_resolved(
            avec,qv,apq,config)
        ||norm_vec(bmq)<config.qmin) {
        return 0.0;
    }

    std::vector<Vec3> z2_left;
    z2_left.push_back(qv);
    z2_left.push_back(bmq);
    std::vector<Vec3> z2_right;
    z2_right.push_back(apq);
    z2_right.push_back(bmq);
    return 4.0 * zrec_kernel_vec(z2_left, config)
           * zrec_kernel_vec(z2_right, config)
           * local_png_b0_dfNL_vec_finite_box(
               P_L,transfer_m,avec,qv,apq,config,b1)
           * p_safe_vec(P_L, bmq);
}

real B122II_png_post_ordered_integrand_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    real scale,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    return
        pow2(r)
        * B122II_png_post_ordered_integrand_physical_vec(
            P_L,
            transfer_m,
            avec,
            bvec,
            config,
            b1,
            q_vec(scale * r, u, o));
}

real B122II_png_post_ordered_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    const real sin_ab = sqrt_nonneg(1.0 - pow2(xab));
    return B122II_png_post_ordered_integrand_vec(
        P_L,
        transfer_m,
        make_vec(0.0, 0.0, ka),
        make_vec(kb * sin_ab, 0.0, kb * xab),
        ka,
        config,
        b1,
        r,
        u,
        o);
}

real B113II_png_post_ordered_integrand_physical_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    const NativeConfig& config,
    real b1,
    const Vec3& qv) {
    const Vec3 bmq = sub_vec(bvec, qv);
    if (!local_png_b0_triplet_is_resolved(
            bvec,qv,bmq,config)) {
        return 0.0;
    }

    std::vector<Vec3> z3;
    z3.push_back(avec);
    z3.push_back(qv);
    z3.push_back(bmq);
    return 3.0 * p_safe_vec(P_L, avec)
           * zrec_kernel_vec(z3, config)
           * local_png_b0_dfNL_vec_finite_box(
               P_L,transfer_m,bvec,qv,bmq,config,b1);
}

real B113II_png_post_ordered_integrand_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    real scale,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    return
        pow2(r)
        * B113II_png_post_ordered_integrand_physical_vec(
            P_L,
            transfer_m,
            avec,
            bvec,
            config,
            b1,
            q_vec(scale * r, u, o));
}

real B113II_png_post_ordered_integrand(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1,
    real r,
    real u,
    real o) {
    const real sin_ab = sqrt_nonneg(1.0 - pow2(xab));
    return B113II_png_post_ordered_integrand_vec(
        P_L,
        transfer_m,
        make_vec(0.0, 0.0, ka),
        make_vec(kb * sin_ab, 0.0, kb * xab),
        ka,
        config,
        b1,
        r,
        u,
        o);
}

template <typename Function>
ComponentStats integrate_loop_component_eps(Function func, real k1, const NativeConfig& config, real epsrel) {
    real c[3] = {config.qmin / k1, -1.0, 0.0};
    real d[3] = {config.qmax / k1, 0.99999999, kTwoPi};
    real abserr = 0.0;
    int neval = 0;
    const real raw = Integrate<3>(func, c, d, epsrel, config.epsabs, &abserr, &neval);
    const real prefactor = pow3(k1 / kTwoPi);
    ComponentStats out;
    out.value = prefactor * raw;
    out.abserr = std::fabs(prefactor) * abserr;
    out.neval = neval;
    return out;
}

template <typename Function>
ComponentStats integrate_loop_component(Function func, real k1, const NativeConfig& config) {
    return integrate_loop_component_eps(func, k1, config, config.epsrel);
}

/*
 * New halo-v1 diagrams span many decades in q/k1.  A linear radial coordinate
 * lets the adaptive partition depend strongly on the bias coefficients and
 * contaminates polynomial-template extraction.  Here x=log(q/k1), hence
 * r^2 dr=r^3 dx: callers retain their historical r^2 factor and this wrapper
 * supplies the additional Jacobian r.  The spherical qmin/qmax domain is
 * unchanged.  Legacy DM and explicit-PNG paths keep the linear-r wrapper.
 */
template <typename Function>
ComponentStats integrate_halo_loop_component_log_eps(
    Function func,
    real k1,
    const NativeConfig& config,
    real epsrel) {
    real c[3] = {std::log(config.qmin / k1), -1.0, 0.0};
    real d[3] = {std::log(config.qmax / k1), 0.99999999, kTwoPi};
    real abserr = 0.0;
    int neval = 0;
    const real raw = Integrate<3>(
        [&func](real log_r, real u, real o) {
            const real r = std::exp(log_r);
            return r * func(r, u, o);
        },
        c,
        d,
        epsrel,
        config.epsabs,
        &abserr,
        &neval);
    const real prefactor = pow3(k1 / kTwoPi);
    ComponentStats out;
    out.value = prefactor * raw;
    out.abserr = std::fabs(prefactor) * abserr;
    out.neval = neval;
    return out;
}

template <typename Function>
ComponentStats integrate_halo_loop_component_log(
    Function func,
    real k1,
    const NativeConfig& config) {
    return integrate_halo_loop_component_log_eps(func, k1, config, config.epsrel);
}

ComponentStats add_stats(const ComponentStats& a, const ComponentStats& b) {
    ComponentStats out;
    out.value = a.value + b.value;
    out.abserr = a.abserr + b.abserr;
    out.neval = a.neval + b.neval;
    return out;
}

typedef std::function<real(const std::vector<Vec3>&)> GaussianKernelFunction;

/*
 * 计算 I3(k)=int d^3q/(2pi)^3 K3(k,q,-q) P_L(q)。
 * k 为外腿模长，kernel 可取 pre halo K_n 或 post halo Z_rec,n；返回积分值、
 * 自适应积分误差和调用次数。B321II/B123I 的组合因子 6 在上层统一乘入。
 */
ComponentStats generic_i3_loop(
    const PowerSpectrum& P_L,
    real k,
    const NativeConfig& config,
    const GaussianKernelFunction& kernel) {
    const Vec3 kvec = make_vec(0.0, 0.0, k);
    return integrate_halo_loop_component_log_eps(
        [&P_L, &kernel, kvec, k](real r, real u, real o) {
            const Vec3 qv = q_vec(k * r, u, o);
            std::vector<Vec3> args;
            args.push_back(kvec);
            args.push_back(qv);
            args.push_back(neg_vec(qv));
            return pow2(r) * kernel(args) * p_safe_vec(P_L, qv);
        },
        k,
        config,
        config.p13_epsrel);
}

/*
 * 任意 Gaussian tracer kernel 的 B222 integrand。
 * vecs 是闭合三角形三条外腿，qv 是 loop 波矢；返回尚未乘 r^2 的 integrand。
 */
real generic_b222_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const NativeConfig& config,
    const GaussianKernelFunction& kernel) {
    const Vec3 k1_minus_q = sub_vec(vecs[0], qv);
    const Vec3 k2_plus_q = add_vec(vecs[1], qv);
    if (norm_vec(k1_minus_q) <= config.singular_floor || norm_vec(k2_plus_q) <= config.singular_floor) return 0.0;
    std::vector<Vec3> k2a;
    k2a.push_back(qv);
    k2a.push_back(k1_minus_q);
    std::vector<Vec3> k2b;
    k2b.push_back(neg_vec(qv));
    k2b.push_back(k2_plus_q);
    std::vector<Vec3> k2c;
    k2c.push_back(k1_minus_q);
    k2c.push_back(k2_plus_q);
    return 8.0 * kernel(k2a) * kernel(k2b) * kernel(k2c)
           * p_safe_vec(P_L, qv) * p_safe_vec(P_L, k1_minus_q) * p_safe_vec(P_L, k2_plus_q);
}

/*
 * 任意 Gaussian tracer kernel 的 B321I integrand。
 * 已显式求和三种 split leg 与另外两条 linear leg，并包含图的组合因子 6。
 */
real generic_b321i_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const NativeConfig& config,
    const GaussianKernelFunction& kernel) {
    real total = 0.0;
    const real p_q = p_safe_vec(P_L, qv);
    for (int split_index = 0; split_index < 3; ++split_index) {
        const Vec3 split_minus_q = sub_vec(vecs[split_index], qv);
        if (norm_vec(split_minus_q) <= config.singular_floor) continue;
        std::vector<Vec3> k2_args;
        k2_args.push_back(qv);
        k2_args.push_back(split_minus_q);
        const real k2 = kernel(k2_args);
        const real p_split = p_safe_vec(P_L, split_minus_q);
        for (int linear_index = 0; linear_index < 3; ++linear_index) {
            if (linear_index == split_index) continue;
            std::vector<Vec3> k1_args(1, vecs[linear_index]);
            std::vector<Vec3> k3_args;
            k3_args.push_back(vecs[linear_index]);
            k3_args.push_back(qv);
            k3_args.push_back(split_minus_q);
            total += kernel(k1_args) * p_safe_vec(P_L, vecs[linear_index])
                     * k2 * kernel(k3_args) * p_q * p_split;
        }
    }
    return 6.0 * total;
}

/*
 * 任意 Gaussian tracer kernel 的 B411/B114 integrand。
 * 对三组外腿 pair 求和并包含图的组合因子 12。
 */
real generic_b411_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const GaussianKernelFunction& kernel) {
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    real total = 0.0;
    const real p_q = p_safe_vec(P_L, qv);
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k1_i(1, vecs[i]);
        std::vector<Vec3> k1_j(1, vecs[j]);
        std::vector<Vec3> k4_args;
        k4_args.push_back(vecs[i]);
        k4_args.push_back(vecs[j]);
        k4_args.push_back(qv);
        k4_args.push_back(neg_vec(qv));
        total += kernel(k1_i) * kernel(k1_j)
                 * p_safe_vec(P_L, vecs[i]) * p_safe_vec(P_L, vecs[j])
                 * kernel(k4_args) * p_q;
    }
    return 12.0 * total;
}

/*
 * 用给定 K_n 统一计算 Gaussian tracer 的 tree+one-loop bispectrum。
 *
 * P_L/tri/config 分别是线性功率谱、三角形与积分设置；kernel 必须返回
 * n=1..4 的对称 tracer kernel。返回 ComponentResult，其中 B321II 在 post
 * 语境等价于 B123I，B411 在 post 语境等价于 B114。
 */
ComponentResult compute_gaussian_with_kernel(
    const PowerSpectrum& P_L,
    const Triangle& tri,
    const NativeConfig& config,
    const GaussianKernelFunction& kernel) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const std::vector<Vec3> vecs = triangle_vecs(tri);
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    real p_ext[3] = {p_safe_vec(P_L, vecs[0]), p_safe_vec(P_L, vecs[1]), p_safe_vec(P_L, vecs[2])};
    real k1_ext[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        std::vector<Vec3> args(1, vecs[i]);
        k1_ext[i] = kernel(args);
        out.stochastic_alpha3_basis += k1_ext[i] * k1_ext[i] * p_ext[i];
    }
    out.stochastic_alpha4_basis = 1.0;

    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k2_args;
        k2_args.push_back(vecs[i]);
        k2_args.push_back(vecs[j]);
        out.Btree += 2.0 * k1_ext[i] * k1_ext[j] * kernel(k2_args) * p_ext[i] * p_ext[j];
    }

    out.P13_k1_stats = generic_i3_loop(P_L, norm_vec(vecs[0]), config, kernel);
    out.P13_k2_stats = generic_i3_loop(P_L, norm_vec(vecs[1]), config, kernel);
    out.P13_k3_stats = generic_i3_loop(P_L, norm_vec(vecs[2]), config, kernel);
    const ComponentStats i3[3] = {out.P13_k1_stats, out.P13_k2_stats, out.P13_k3_stats};
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k2_args;
        k2_args.push_back(vecs[i]);
        k2_args.push_back(vecs[j]);
        out.B321II += 6.0 * kernel(k2_args) * p_ext[i] * p_ext[j]
                      * (k1_ext[i] * i3[j].value + k1_ext[j] * i3[i].value);
    }

    out.B222_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &config, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * generic_b222_integrand(P_L, vecs, qv, config, kernel);
        },
        tri.k1,
        config);
    out.B321I_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &config, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * generic_b321i_integrand(P_L, vecs, qv, config, kernel);
        },
        tri.k1,
        config);
    out.B411_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * generic_b411_integrand(P_L, vecs, qv, kernel);
        },
        tri.k1,
        config);

    out.B222 = out.B222_stats.value;
    out.B321I = out.B321I_stats.value;
    out.B411 = out.B411_stats.value;
    out.Bloopterms = out.B222 + out.B321I + out.B411;
    out.B1loop = out.Bloopterms + out.B321II;
    out.Btotal = out.Btree + out.B1loop;
    return out;
}

typedef std::function<KernelResponse(const std::vector<Vec3>&)> ResponseKernelFunction;

/*
 * I3^R(k)=int d^3q/(2pi)^3 K3^R(k,q,-q) P_L(q)。Gaussian I3 已由
 * compute_gaussian_with_kernel 计算；这里只积分线性 PNG 响应。
 */
ComponentStats generic_response_i3_loop(
    const PowerSpectrum& P_L,
    real k,
    const NativeConfig& config,
    const ResponseKernelFunction& kernel) {
    const Vec3 kvec = make_vec(0.0, 0.0, k);
    return integrate_halo_loop_component_log_eps(
        [&P_L, &kernel, kvec, k](real r, real u, real o) {
            const Vec3 qv = q_vec(k * r, u, o);
            std::vector<Vec3> args;
            args.push_back(kvec);
            args.push_back(qv);
            args.push_back(neg_vec(qv));
            return pow2(r) * kernel(args).response * p_safe_vec(P_L, qv);
        },
        k,
        config,
        config.p13_epsrel);
}

real generic_response_b222_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const NativeConfig& config,
    const ResponseKernelFunction& kernel) {
    const Vec3 k1_minus_q = sub_vec(vecs[0], qv);
    const Vec3 k2_plus_q = add_vec(vecs[1], qv);
    if (norm_vec(k1_minus_q) <= config.singular_floor
        || norm_vec(k2_plus_q) <= config.singular_floor) return 0.0;
    std::vector<Vec3> args_a;
    args_a.push_back(qv);
    args_a.push_back(k1_minus_q);
    std::vector<Vec3> args_b;
    args_b.push_back(neg_vec(qv));
    args_b.push_back(k2_plus_q);
    std::vector<Vec3> args_c;
    args_c.push_back(k1_minus_q);
    args_c.push_back(k2_plus_q);
    const KernelResponse a = kernel(args_a);
    const KernelResponse b = kernel(args_b);
    const KernelResponse c = kernel(args_c);
    const real product_response = a.response * b.gaussian * c.gaussian
                                  + a.gaussian * b.response * c.gaussian
                                  + a.gaussian * b.gaussian * c.response;
    return 8.0 * product_response
           * p_safe_vec(P_L, qv)
           * p_safe_vec(P_L, k1_minus_q)
           * p_safe_vec(P_L, k2_plus_q);
}

real generic_response_b321i_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const NativeConfig& config,
    const ResponseKernelFunction& kernel) {
    real total = 0.0;
    const real p_q = p_safe_vec(P_L, qv);
    for (int split_index = 0; split_index < 3; ++split_index) {
        const Vec3 split_minus_q = sub_vec(vecs[split_index], qv);
        if (norm_vec(split_minus_q) <= config.singular_floor) continue;
        std::vector<Vec3> k2_args;
        k2_args.push_back(qv);
        k2_args.push_back(split_minus_q);
        const KernelResponse k2 = kernel(k2_args);
        const real p_split = p_safe_vec(P_L, split_minus_q);
        for (int linear_index = 0; linear_index < 3; ++linear_index) {
            if (linear_index == split_index) continue;
            std::vector<Vec3> k1_args(1, vecs[linear_index]);
            std::vector<Vec3> k3_args;
            k3_args.push_back(vecs[linear_index]);
            k3_args.push_back(qv);
            k3_args.push_back(split_minus_q);
            const KernelResponse k1 = kernel(k1_args);
            const KernelResponse k3 = kernel(k3_args);
            const real product_response = k1.response * k2.gaussian * k3.gaussian
                                          + k1.gaussian * k2.response * k3.gaussian
                                          + k1.gaussian * k2.gaussian * k3.response;
            total += product_response
                     * p_safe_vec(P_L, vecs[linear_index]) * p_q * p_split;
        }
    }
    return 6.0 * total;
}

real generic_response_b411_integrand(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const Vec3& qv,
    const ResponseKernelFunction& kernel) {
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    const real p_q = p_safe_vec(P_L, qv);
    real total = 0.0;
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k1_i_args(1, vecs[i]);
        std::vector<Vec3> k1_j_args(1, vecs[j]);
        std::vector<Vec3> k4_args;
        k4_args.push_back(vecs[i]);
        k4_args.push_back(vecs[j]);
        k4_args.push_back(qv);
        k4_args.push_back(neg_vec(qv));
        const KernelResponse k1_i = kernel(k1_i_args);
        const KernelResponse k1_j = kernel(k1_j_args);
        const KernelResponse k4 = kernel(k4_args);
        const real product_response = k1_i.response * k1_j.gaussian * k4.gaussian
                                      + k1_i.gaussian * k1_j.response * k4.gaussian
                                      + k1_i.gaussian * k1_j.gaussian * k4.response;
        total += product_response
                 * p_safe_vec(P_L, vecs[i])
                 * p_safe_vec(P_L, vecs[j]) * p_q;
    }
    return 12.0 * total;
}

/*
 * 对完整 Gaussian bispectrum 泛函逐图求一次 Gateaux 导数。
 *
 * Gaussian 输入已经包含 tree/B222/B321I/B321II/B411 及三个 I3^G；kernel
 * 同时返回 K^G 与 K^R。输出只填写新增的 dB/d fNL diagram 字段。积分测度、
 * cutoff 与 Gaussian 基底完全一致，因此二者可逐图做数值收敛比较。这里
 * 固定采用 Gaussian diagram 原生的 loop momentum routing；把结果改写成旧
 * 显式 B0 图需要平移 q，而有限球形硬 cutoff 不在该平移下不变。由此产生的
 * regulated routing 差异属于 fixed-cutoff scheme dependence，不能通过修改
 * KIC 的 2/n 组合系数来吸收。
 */
ComponentResult compute_local_png_response_with_kernel(
    const PowerSpectrum& P_L,
    const Triangle& tri,
    const NativeConfig& config,
    const ResponseKernelFunction& kernel,
    const ComponentResult& gaussian) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const std::vector<Vec3> vecs = triangle_vecs(tri);
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    real p_ext[3] = {
        p_safe_vec(P_L, vecs[0]),
        p_safe_vec(P_L, vecs[1]),
        p_safe_vec(P_L, vecs[2])};
    KernelResponse k1_ext[3];
    for (int i = 0; i < 3; ++i) {
        std::vector<Vec3> args(1, vecs[i]);
        k1_ext[i] = kernel(args);
    }

    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k2_args;
        k2_args.push_back(vecs[i]);
        k2_args.push_back(vecs[j]);
        const KernelResponse k2 = kernel(k2_args);
        const real product_response =
            k1_ext[i].response * k1_ext[j].gaussian * k2.gaussian
            + k1_ext[i].gaussian * k1_ext[j].response * k2.gaussian
            + k1_ext[i].gaussian * k1_ext[j].gaussian * k2.response;
        out.dBdfNL_local_tree += 2.0 * product_response * p_ext[i] * p_ext[j];
    }

    out.dP13dfNL_local_k1_stats =
        generic_response_i3_loop(P_L, norm_vec(vecs[0]), config, kernel);
    out.dP13dfNL_local_k2_stats =
        generic_response_i3_loop(P_L, norm_vec(vecs[1]), config, kernel);
    out.dP13dfNL_local_k3_stats =
        generic_response_i3_loop(P_L, norm_vec(vecs[2]), config, kernel);
    const ComponentStats response_i3[3] = {
        out.dP13dfNL_local_k1_stats,
        out.dP13dfNL_local_k2_stats,
        out.dP13dfNL_local_k3_stats};
    const ComponentStats gaussian_i3[3] = {
        gaussian.P13_k1_stats,
        gaussian.P13_k2_stats,
        gaussian.P13_k3_stats};

    real b321ii_error = 0.0;
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> k2_args;
        k2_args.push_back(vecs[i]);
        k2_args.push_back(vecs[j]);
        const KernelResponse k2 = kernel(k2_args);
        const real gaussian_bracket =
            k1_ext[i].gaussian * gaussian_i3[j].value
            + k1_ext[j].gaussian * gaussian_i3[i].value;
        const real response_bracket =
            k1_ext[i].response * gaussian_i3[j].value
            + k1_ext[i].gaussian * response_i3[j].value
            + k1_ext[j].response * gaussian_i3[i].value
            + k1_ext[j].gaussian * response_i3[i].value;
        const real prefactor = 6.0 * p_ext[i] * p_ext[j];
        out.dBdfNL_local_B321II += prefactor
                                       * (k2.response * gaussian_bracket
                                          + k2.gaussian * response_bracket);
        b321ii_error += std::fabs(prefactor)
                        * (std::fabs(k2.response)
                               * (std::fabs(k1_ext[i].gaussian) * gaussian_i3[j].abserr
                                  + std::fabs(k1_ext[j].gaussian) * gaussian_i3[i].abserr)
                           + std::fabs(k2.gaussian)
                               * (std::fabs(k1_ext[i].response) * gaussian_i3[j].abserr
                                  + std::fabs(k1_ext[i].gaussian) * response_i3[j].abserr
                                  + std::fabs(k1_ext[j].response) * gaussian_i3[i].abserr
                                  + std::fabs(k1_ext[j].gaussian) * response_i3[i].abserr));
    }
    out.dBdfNL_local_B321II_stats.value = out.dBdfNL_local_B321II;
    out.dBdfNL_local_B321II_stats.abserr = b321ii_error;
    for (int i = 0; i < 3; ++i) {
        out.dBdfNL_local_B321II_stats.neval += gaussian_i3[i].neval + response_i3[i].neval;
    }

    out.dBdfNL_local_B222_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &config, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r)
                   * generic_response_b222_integrand(P_L, vecs, qv, config, kernel);
        },
        tri.k1,
        config);
    out.dBdfNL_local_B321I_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &config, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r)
                   * generic_response_b321i_integrand(P_L, vecs, qv, config, kernel);
        },
        tri.k1,
        config);
    out.dBdfNL_local_B411_stats = integrate_halo_loop_component_log(
        [&P_L, &vecs, &kernel, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * generic_response_b411_integrand(P_L, vecs, qv, kernel);
        },
        tri.k1,
        config);
    out.dBdfNL_local_B222 = out.dBdfNL_local_B222_stats.value;
    out.dBdfNL_local_B321I = out.dBdfNL_local_B321I_stats.value;
    out.dBdfNL_local_B411 = out.dBdfNL_local_B411_stats.value;
    out.dBdfNL_local_1loop =
        out.dBdfNL_local_B222
        + out.dBdfNL_local_B321I
        + out.dBdfNL_local_B321II
        + out.dBdfNL_local_B411;
    out.dBdfNL_local_total = out.dBdfNL_local_tree + out.dBdfNL_local_1loop;
    return out;
}

real post_b222_integrand(const PowerSpectrum& P_L, const std::vector<Vec3>& vecs, const Vec3& qv, const NativeConfig& config) {
    const Vec3 k1_minus_q = sub_vec(vecs[0], qv);
    const Vec3 k2_plus_q = add_vec(vecs[1], qv);
    if (norm_vec(k1_minus_q) <= config.singular_floor || norm_vec(k2_plus_q) <= config.singular_floor) return 0.0;
    std::vector<Vec3> z2a;
    z2a.push_back(qv);
    z2a.push_back(k1_minus_q);
    std::vector<Vec3> z2b;
    z2b.push_back(neg_vec(qv));
    z2b.push_back(k2_plus_q);
    std::vector<Vec3> z2c;
    z2c.push_back(k1_minus_q);
    z2c.push_back(k2_plus_q);
    return 8.0
           * zrec_kernel_vec(z2a, config)
           * zrec_kernel_vec(z2b, config)
           * zrec_kernel_vec(z2c, config)
           * p_safe_vec(P_L, qv)
           * p_safe_vec(P_L, k1_minus_q)
           * p_safe_vec(P_L, k2_plus_q);
}

real post_b321i_integrand(const PowerSpectrum& P_L, const std::vector<Vec3>& vecs, const Vec3& qv, const NativeConfig& config) {
    real local = 0.0;
    real p_ext[3] = {p_safe_vec(P_L, vecs[0]), p_safe_vec(P_L, vecs[1]), p_safe_vec(P_L, vecs[2])};
    const real p_q = p_safe_vec(P_L, qv);
    for (int split_index = 0; split_index < 3; ++split_index) {
        const Vec3 split_minus_q = sub_vec(vecs[split_index], qv);
        if (norm_vec(split_minus_q) <= config.singular_floor) continue;
        std::vector<Vec3> z2_split;
        z2_split.push_back(qv);
        z2_split.push_back(split_minus_q);
        const real z2 = zrec_kernel_vec(z2_split, config);
        const real p_split_minus_q = p_safe_vec(P_L, split_minus_q);
        for (int linear_index = 0; linear_index < 3; ++linear_index) {
            if (linear_index == split_index) continue;
            std::vector<Vec3> z3;
            z3.push_back(vecs[linear_index]);
            z3.push_back(qv);
            z3.push_back(split_minus_q);
            local += p_ext[linear_index] * z2 * zrec_kernel_vec(z3, config) * p_q * p_split_minus_q;
        }
    }
    return 6.0 * local;
}

real post_b114_integrand(const PowerSpectrum& P_L, const std::vector<Vec3>& vecs, const Vec3& qv, const NativeConfig& config) {
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    real local = 0.0;
    const real p_q = p_safe_vec(P_L, qv);
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> z4;
        z4.push_back(vecs[i]);
        z4.push_back(vecs[j]);
        z4.push_back(qv);
        z4.push_back(neg_vec(qv));
        local += p_safe_vec(P_L, vecs[i]) * p_safe_vec(P_L, vecs[j]) * zrec_kernel_vec(z4, config) * p_q;
    }
    return 12.0 * local;
}

ComponentStats post_i3_loop_vec(
    const PowerSpectrum& P_L,
    const Vec3& kvec,
    const NativeConfig& config) {
    using NumericKey=
        std::pair<std::array<int,4>,std::array<real,12>>;
    using CacheKey=
        std::pair<const PowerSpectrum*,NumericKey>;
    static thread_local std::map<
        CacheKey,ComponentStats> cache;
    const CacheKey key{
        &P_L,
        NumericKey{
            std::array<int,4>{{
                config.recon_cic_window_power,
                config.png_linear_multicenter_qmc ? 1 : 0,
                config.png_b112ii_qmc_power,
                config.png_b112ii_qmc_replicates}},
            std::array<real,12>{{
                kvec.x,kvec.y,kvec.z,
                config.qmin,config.qmax,config.epsrel,
                config.p13_epsrel,config.epsabs,
                config.smoothing_radius,config.bias_recon,
                config.recon_cellsize,config.singular_floor}}}};
    std::map<CacheKey, ComponentStats>::const_iterator found =
        cache.find(key);
    if (found != cache.end()) return found->second;
    const real k=norm_vec(kvec);
    if (k<=config.singular_floor) return ComponentStats();
    if (config.png_linear_multicenter_qmc) {
        std::vector<LocalPngQmcCenter> centers;
        centers.push_back(
            LocalPngQmcCenter{
                make_vec(0.0,0.0,0.0),config.qmin});
        ComponentStats out=
            integrate_local_png_multicenter_qmc(
                [&P_L,&config,kvec](const Vec3& qv) {
                    std::vector<Vec3> z3;
                    z3.push_back(kvec);
                    z3.push_back(qv);
                    z3.push_back(neg_vec(qv));
                    return
                        zrec_kernel_vec(z3,config)
                        *p_safe_vec(P_L,qv);
                },
                centers,
                config);
        cache[key]=out;
        return out;
    }
    ComponentStats out = integrate_loop_component_eps(
        [&P_L, &config, kvec, k](real r, real u, real o) {
            const real q = k * r;
            const Vec3 qv = q_vec(q, u, o);
            std::vector<Vec3> z3;
            z3.push_back(kvec);
            z3.push_back(qv);
            z3.push_back(neg_vec(qv));
            return pow2(r) * zrec_kernel_vec(z3, config) * p_safe_vec(P_L, qv);
        },
        k,
        config,
        config.p13_epsrel);
    cache[key] = out;
    return out;
}

ComponentStats post_i3_loop(
    const PowerSpectrum& P_L,
    real k,
    const NativeConfig& config) {
    return post_i3_loop_vec(
        P_L,make_vec(0.0,0.0,k),config);
}

real post_b123i_from_i3(
    const PowerSpectrum& P_L,
    const std::vector<Vec3>& vecs,
    const ComponentStats& i3_0,
    const ComponentStats& i3_1,
    const ComponentStats& i3_2,
    const NativeConfig& config) {
    const ComponentStats i3[3] = {i3_0, i3_1, i3_2};
    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    real total = 0.0;
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> z2;
        z2.push_back(vecs[i]);
        z2.push_back(vecs[j]);
        total += zrec_kernel_vec(z2, config)
                 * p_safe_vec(P_L, vecs[i])
                 * p_safe_vec(P_L, vecs[j])
                 * (i3[i].value + i3[j].value);
    }
    return 6.0 * total;
}

real damping_ir(real k, const NativeConfig& config) {
    return std::exp(-0.5 * k * k * config.ir_sigma2);
}

real p_wiggle_safe_vec(const PowerSpectrum& P_L, const PowerSpectrum& P_nw, const Vec3& vec) {
    return p_safe_vec(P_L, vec) - p_safe_vec(P_nw, vec);
}

real pair_bgg112_coeff(const Vec3& a, const Vec3& b, const NativeConfig& config) {
    std::vector<Vec3> z2;
    z2.push_back(a);
    z2.push_back(b);
    return 2.0 * zrec_kernel_vec(z2, config);
}

ComponentStats pair_bgg114_coeff(
    const PowerSpectrum& P_nw,
    const Vec3& a,
    const Vec3& b,
    real qscale,
    const NativeConfig& config) {
    return integrate_loop_component(
        [&P_nw, &config, a, b, qscale](real r, real u, real o) {
            const Vec3 qv = q_vec(qscale * r, u, o);
            std::vector<Vec3> z4;
            z4.push_back(a);
            z4.push_back(b);
            z4.push_back(qv);
            z4.push_back(neg_vec(qv));
            return pow2(r) * 12.0 * zrec_kernel_vec(z4, config) * p_safe_vec(P_nw, qv);
        },
        qscale,
        config);
}

real pair_bgg123i_coeff_from_i3(
    const Vec3& a,
    const Vec3& b,
    const ComponentStats& i3_a,
    const ComponentStats& i3_b,
    const NativeConfig& config) {
    std::vector<Vec3> z2;
    z2.push_back(a);
    z2.push_back(b);
    return 6.0 * zrec_kernel_vec(z2, config) * (i3_a.value + i3_b.value);
}

ComponentStats pair_bgm123ii_coeff(
    const PowerSpectrum& P_nw,
    const Vec3& linear,
    const Vec3& split,
    real qscale,
    const NativeConfig& config) {
    return integrate_loop_component(
        [&P_nw, &config, linear, split, qscale](real r, real u, real o) {
            const Vec3 qv = q_vec(qscale * r, u, o);
            const Vec3 split_minus_q = sub_vec(split, qv);
            if (norm_vec(split_minus_q) <= config.singular_floor) return 0.0;
            std::vector<Vec3> z2;
            z2.push_back(qv);
            z2.push_back(split_minus_q);
            std::vector<Vec3> z3;
            z3.push_back(linear);
            z3.push_back(qv);
            z3.push_back(split_minus_q);
            return pow2(r)
                   * 6.0
                   * zrec_kernel_vec(z2, config)
                   * zrec_kernel_vec(z3, config)
                   * p_safe_vec(P_nw, qv)
                   * p_safe_vec(P_nw, split_minus_q);
        },
        qscale,
        config);
}

}  // namespace

real local_png_ir_cutoff(const NativeConfig& config) {
    return config.png_ir_cutoff > 0.0
        ? config.png_ir_cutoff
        : config.qmin;
}

bool local_png_primordial_triplet_is_resolved(
    real k1,
    real k2,
    real k3,
    const NativeConfig& config) {
    const real cutoff=local_png_ir_cutoff(config);
    return k1>=cutoff &&k2>=cutoff &&k3>=cutoff;
}

ComponentStats audit_local_png_multicenter_qmc_excision_volume(
    const std::array<real,3>& shifted_center,
    real shifted_cutoff,
    const NativeConfig& config) {
    if (!(shifted_cutoff>0.0)) return ComponentStats();
    const Vec3 center=
        make_vec(
            shifted_center[0],
            shifted_center[1],
            shifted_center[2]);
    std::vector<LocalPngQmcCenter> centers;
    centers.push_back(
        LocalPngQmcCenter{
            make_vec(0.0,0.0,0.0),config.qmin});
    centers.push_back(
        LocalPngQmcCenter{center,shifted_cutoff});
    return
        integrate_local_png_multicenter_qmc(
            [center,shifted_cutoff](const Vec3& qv) {
                return
                    norm_vec(sub_vec(qv,center))
                            >=shifted_cutoff
                        ?1.0
                        :0.0;
            },
            centers,
            config);
}

real triangle_k3(const Triangle& tri) {
    return std::sqrt(pow2(tri.k1) + pow2(tri.k2) + 2.0 * tri.k1 * tri.k2 * tri.mu12);
}

real alpha(real k1, real k2, real mu) {
    if (k1 <= 0.0) return 0.0;
    return 1.0 + k2 * mu / k1;
}

real alphas(real k1, real k2, real mu) {
    return 0.5 * (alpha(k1, k2, mu) + alpha(k2, k1, mu));
}

real beta1(real k1, real k2, real mu) {
    if (k1 <= 0.0 || k2 <= 0.0) return 0.0;
    return mu * (pow2(k1) + pow2(k2) + 2.0 * k1 * k2 * mu) / (2.0 * k1 * k2);
}

real F2eds(real k1, real k2, real mu) {
    return 5.0 / 7.0 * alphas(k1, k2, mu) + 2.0 / 7.0 * beta1(k1, k2, mu);
}

real G2eds(real k1, real k2, real mu) {
    return 3.0 / 7.0 * alphas(k1, k2, mu) + 4.0 / 7.0 * beta1(k1, k2, mu);
}

real F3eds(real k1, real k2, real k3, real x23, real x12, real x13) {
    const real k23 = std::sqrt(pow2(k2) + pow2(k3) + 2.0 * k2 * k3 * x23);
    const real k13 = std::sqrt(pow2(k1) + pow2(k3) + 2.0 * k1 * k3 * x13);
    const real k12 = std::sqrt(pow2(k1) + pow2(k2) + 2.0 * k1 * k2 * x12);
    return F3edsb(k1, k2, k3, k23, k12, k13, x23, x12, x13);
}

real G3eds(real k1, real k2, real k3, real x23, real x12, real x13) {
    const real k23 = std::sqrt(pow2(k2) + pow2(k3) + 2.0 * k2 * k3 * x23);
    const real k13 = std::sqrt(pow2(k1) + pow2(k3) + 2.0 * k1 * k3 * x13);
    const real k12 = std::sqrt(pow2(k1) + pow2(k2) + 2.0 * k1 * k2 * x12);
    return G3edsb(k1, k2, k3, k23, k12, k13, x23, x12, x13);
}

real F3edsb(real k1, real k2, real k3, real k23, real k12, real k13, real x23, real x12, real x13) {
    if (std::min(std::min(k1, k2), std::min(std::min(k3, k23), std::min(k12, k13))) <= 0.0) return 0.0;
    return (2.0 / 63.0 * 2.0 * beta1(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 6.0 / 4.0 * alphas(k2, k3, x23))
            + 1.0 / 18.0 * alpha(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 10.0 / 2.0 * alphas(k2, k3, x23))
            + 1.0 / 9.0 * alpha(k23, k1, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 6.0 / 4.0 * alphas(k2, k3, x23))
            + 2.0 / 63.0 * 2.0 * beta1(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 6.0 / 4.0 * alphas(k1, k2, x12))
            + 1.0 / 18.0 * alpha(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 10.0 / 2.0 * alphas(k1, k2, x12))
            + 1.0 / 9.0 * alpha(k12, k3, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 6.0 / 4.0 * alphas(k1, k2, x12))
            + 2.0 / 63.0 * 2.0 * beta1(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 6.0 / 4.0 * alphas(k1, k3, x13))
            + 1.0 / 18.0 * alpha(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 10.0 / 2.0 * alphas(k1, k3, x13))
            + 1.0 / 9.0 * alpha(k13, k2, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 6.0 / 4.0 * alphas(k1, k3, x13)))
           / 3.0;
}

real G3edsb(real k1, real k2, real k3, real k23, real k12, real k13, real x23, real x12, real x13) {
    if (std::min(std::min(k1, k2), std::min(std::min(k3, k23), std::min(k12, k13))) <= 0.0) return 0.0;
    return (2.0 / 21.0 * 2.0 * beta1(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 6.0 / 4.0 * alphas(k2, k3, x23))
            + 1.0 / 42.0 * alpha(k1, k23, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 10.0 / 2.0 * alphas(k2, k3, x23))
            + 1.0 / 21.0 * alpha(k23, k1, (k2 * x12 + k3 * x13) / k23) * (2.0 * beta1(k2, k3, x23) + 6.0 / 4.0 * alphas(k2, k3, x23))
            + 2.0 / 21.0 * 2.0 * beta1(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 6.0 / 4.0 * alphas(k1, k2, x12))
            + 1.0 / 42.0 * alpha(k3, k12, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 10.0 / 2.0 * alphas(k1, k2, x12))
            + 1.0 / 21.0 * alpha(k12, k3, (k1 * x13 + k2 * x23) / k12) * (2.0 * beta1(k1, k2, x12) + 6.0 / 4.0 * alphas(k1, k2, x12))
            + 2.0 / 21.0 * 2.0 * beta1(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 6.0 / 4.0 * alphas(k1, k3, x13))
            + 1.0 / 42.0 * alpha(k2, k13, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 10.0 / 2.0 * alphas(k1, k3, x13))
            + 1.0 / 21.0 * alpha(k13, k2, (k1 * x12 + k3 * x23) / k13) * (2.0 * beta1(k1, k3, x13) + 6.0 / 4.0 * alphas(k1, k3, x13)))
           / 3.0;
}

real F4edsb(real k1, real k2, real k3, real k4, real x23, real x24, real x12, real x13, real x14) {
    const real x34 = kReactXmin;
    const real k1s = pow2(k1);
    const real k2s = pow2(k2);
    const real k3s = pow2(k3);
    const real k4s = pow2(k4);
    const real k1234 = k1s + k2s + k3s + k4s + 2.0 * k1 * k2 * x12 + 2.0 * k1 * k3 * x13
                       + 2.0 * k2 * k3 * x23 + 2.0 * k1 * k4 * x14 + 2.0 * k2 * k4 * x24 + 2.0 * k3 * k4 * x34;
    const real k123 = k1s + k2s + k3s + 2.0 * k1 * k2 * x12 + 2.0 * k1 * k3 * x13 + 2.0 * k2 * k3 * x23;
    const real k234 = k2s + k3s + k4s + 2.0 * k2 * k3 * x23 + 2.0 * k2 * k4 * x24 + 2.0 * k3 * k4 * x34;
    const real k341 = k3s + k4s + k1s + 2.0 * k3 * k4 * x34 + 2.0 * k3 * k1 * x13 + 2.0 * k4 * k1 * x14;
    const real k412 = k4s + k1s + k2s + 2.0 * k4 * k1 * x14 + 2.0 * k4 * k2 * x24 + 2.0 * k1 * k2 * x12;
    const real k12 = k1s + k2s + 2.0 * k1 * k2 * x12;
    const real k23 = k2s + k3s + 2.0 * k2 * k3 * x23;
    const real k34 = k3s + k4s + 2.0 * k3 * k4 * x34;
    const real k41 = k4s + k1s + 2.0 * k4 * k1 * x14;
    const real k13 = k1s + k3s + 2.0 * k1 * k3 * x13;
    const real k24 = k2s + k4s + 2.0 * k2 * k4 * x24;
    if (std::min(std::min(std::min(k123, k234), std::min(k341, k412)), std::min(std::min(k12, k23), std::min(std::min(k34, k41), std::min(k13, k24)))) <= kTiny) {
        return 0.0;
    }

    const real k12rt = sqrt_nonneg(k12);
    const real k23rt = sqrt_nonneg(k23);
    const real k34rt = sqrt_nonneg(k34);
    const real k41rt = sqrt_nonneg(k41);
    const real k13rt = sqrt_nonneg(k13);
    const real k24rt = sqrt_nonneg(k24);

    return (27.0 * (k1s + k1 * k2 * x12 + k1 * k3 * x13 + k1 * k4 * x14) / k1s * F3edsb(k2, k3, k4, k34rt, k23rt, k24rt, x34, x23, x24)
            + (27.0 * (k234 + k1 * (k2 * x12 + k3 * x13 + k4 * x14)) / k234 + 6.0 * k1234 * k1 * (k2 * x12 + k3 * x13 + k4 * x14) / k1s / k234) * G3edsb(k2, k3, k4, k34rt, k23rt, k24rt, x34, x23, x24)
            + 27.0 * (k2s + k2 * k3 * x23 + k2 * k4 * x24 + k2 * k1 * x12) / k2s * F3edsb(k3, k4, k1, k41rt, k34rt, k13rt, x14, x34, x13)
            + (27.0 * (k341 + k2 * (k3 * x23 + k4 * x24 + k1 * x12)) / k341 + 6.0 * k1234 * k2 * (k3 * x23 + k4 * x24 + k1 * x12) / k2s / k341) * G3edsb(k3, k4, k1, k41rt, k34rt, k13rt, x14, x34, x13)
            + 27.0 * (k3s + k3 * k4 * x34 + k3 * k1 * x13 + k3 * k2 * x23) / k3s * F3edsb(k4, k1, k2, k12rt, k41rt, k24rt, x12, x14, x24)
            + (27.0 * (k412 + k3 * (k4 * x34 + k1 * x13 + k2 * x23)) / k412 + 6.0 * k1234 * k3 * (k4 * x34 + k1 * x13 + k2 * x23) / k3s / k412) * G3edsb(k4, k1, k2, k12rt, k41rt, k24rt, x12, x14, x24)
            + 27.0 * (k4s + k4 * k1 * x14 + k4 * k2 * x24 + k4 * k3 * x34) / k4s * F3edsb(k1, k2, k3, k23rt, k12rt, k13rt, x23, x12, x13)
            + (27.0 * (k123 + k4 * (k1 * x14 + k2 * x24 + k3 * x34)) / k123 + 6.0 * k1234 * k4 * (k1 * x14 + k2 * x24 + k3 * x34) / k4s / k123) * G3edsb(k1, k2, k3, k23rt, k12rt, k13rt, x23, x12, x13)
            + 18.0 * (k12 + k3 * k1 * x13 + k3 * k2 * x23 + k4 * k1 * x14 + k4 * k2 * x24) / k12 * G2eds(k1, k2, x12) * F2eds(k3, k4, x34)
            + 18.0 * (k23 + k4 * k2 * x24 + k4 * k3 * x34 + k1 * k2 * x12 + k1 * k3 * x13) / k23 * G2eds(k2, k3, x23) * F2eds(k4, k1, x14)
            + 18.0 * (k34 + k1 * k3 * x13 + k1 * k4 * x14 + k2 * k3 * x23 + k2 * k4 * x24) / k34 * G2eds(k3, k4, x34) * F2eds(k1, k2, x12)
            + 18.0 * (k13 + k1 * k2 * x12 + k1 * k4 * x14 + k2 * k3 * x23 + k3 * k4 * x34) / k13 * G2eds(k1, k3, x13) * F2eds(k2, k4, x24)
            + 18.0 * (k24 + k2 * k3 * x23 + k2 * k1 * x12 + k4 * k1 * x14 + k3 * k4 * x34) / k24 * G2eds(k2, k4, x24) * F2eds(k1, k3, x13)
            + 18.0 * (k41 + k1 * k2 * x12 + k1 * k3 * x13 + k4 * k2 * x24 + k3 * k4 * x34) / k41 * G2eds(k1, k4, x14) * F2eds(k2, k3, x23)
            + 4.0 * k1234 * (k1 * k3 * x13 + k1 * k4 * x14 + k2 * k3 * x23 + k2 * k4 * x24) / k12 / k34 * G2eds(k1, k2, x12) * G2eds(k3, k4, x34)
            + 4.0 * k1234 * (k2 * k4 * x24 + k2 * k1 * x12 + k3 * k4 * x34 + k3 * k1 * x13) / k23 / k41 * G2eds(k2, k3, x23) * G2eds(k4, k1, x14)
            + 4.0 * k1234 * (k2 * k1 * x12 + k2 * k3 * x23 + k4 * k1 * x14 + k4 * k3 * x34) / k24 / k13 * G2eds(k2, k4, x24) * G2eds(k1, k3, x13))
           / 396.0;
}

ComponentStats P13_dd(const PowerSpectrum& P_L, real k, const NativeConfig& config) {
    typedef std::tuple<const PowerSpectrum*, real, real, real, real, real> CacheKey;
    static std::map<CacheKey, ComponentStats> cache;
    const CacheKey key(&P_L, k, config.qmin, config.qmax, config.p13_epsrel, config.epsabs);
    std::map<CacheKey, ComponentStats>::const_iterator found = cache.find(key);
    if (found != cache.end()) return found->second;

    real abserr = 0.0;
    int neval = 0;
    const real kmin = config.qmin / k;
    const real kmax = config.qmax / k;
    const real raw = Integrate<ExpSub>(
        [&P_L, k](real r) {
            return p_safe(P_L, k * r) * p13_shape(r);
        },
        kmin,
        kmax,
        config.p13_epsrel,
        config.epsabs,
        &abserr,
        &neval);
    ComponentStats out;
    const real prefactor = pow3(k) * p_safe(P_L, k) / (252.0 * 4.0 * kPi * kPi);
    out.value = prefactor * raw;
    out.abserr = std::fabs(prefactor) * abserr;
    out.neval = neval;
    cache[key] = out;
    return out;
}

ComponentResult compute_pre_recon_gaussian(const PowerSpectrum& P_L, const Triangle& tri, const NativeConfig& config) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);

    const real F2A = F2eds(tri.k1, tri.k2, tri.mu12);
    const real F2B = F2eds(tri.k1, out.k3, -(tri.k2 * tri.mu12 + tri.k1) / out.k3);
    const real F2C = F2eds(tri.k2, out.k3, -(tri.k1 * tri.mu12 + tri.k2) / out.k3);

    out.Btree = 2.0 * (p_safe(P_L, tri.k1) * p_safe(P_L, tri.k2) * F2A
                       + p_safe(P_L, out.k3) * p_safe(P_L, tri.k1) * F2B
                       + p_safe(P_L, tri.k2) * p_safe(P_L, out.k3) * F2C);

    out.P13_k1_stats = P13_dd(P_L, tri.k1, config);
    out.P13_k2_stats = P13_dd(P_L, tri.k2, config);
    out.P13_k3_stats = P13_dd(P_L, out.k3, config);
    out.B321II = F2A * (p_safe(P_L, tri.k1) * out.P13_k2_stats.value + p_safe(P_L, tri.k2) * out.P13_k1_stats.value)
                 + F2B * (p_safe(P_L, out.k3) * out.P13_k1_stats.value + p_safe(P_L, tri.k1) * out.P13_k3_stats.value)
                 + F2C * (p_safe(P_L, out.k3) * out.P13_k2_stats.value + p_safe(P_L, tri.k2) * out.P13_k3_stats.value);

    out.B222_stats = integrate_loop_component(
        [&P_L, &tri, &out](real r, real u, real o) {
            return B222_integrand(P_L, tri.k1, tri.k2, tri.mu12, out.k3, r, u, o);
        },
        tri.k1,
        config);
    out.B321I_stats = integrate_loop_component(
        [&P_L, &tri, &out](real r, real u, real o) {
            return B321I_integrand(P_L, tri.k1, tri.k2, tri.mu12, out.k3, r, u, o);
        },
        tri.k1,
        config);
    out.B411_stats = integrate_loop_component(
        [&P_L, &tri, &out](real r, real u, real o) {
            return B411_integrand(P_L, tri.k1, tri.k2, tri.mu12, out.k3, r, u, o);
        },
        tri.k1,
        config);

    out.B222 = out.B222_stats.value;
    out.B321I = out.B321I_stats.value;
    out.B411 = out.B411_stats.value;
    out.Bloopterms = out.B222 + out.B321I + out.B411;
    out.B1loop = out.Bloopterms + out.B321II;
    out.Btotal = out.Btree + out.B1loop;
    return out;
}

ComponentResult compute_post_recon_gaussian(const PowerSpectrum& P_L, const Triangle& tri, const NativeConfig& config) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const std::vector<Vec3> vecs = triangle_vecs(tri);

    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    out.Btree = 0.0;
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> z2;
        z2.push_back(vecs[i]);
        z2.push_back(vecs[j]);
        out.Btree += 2.0 * zrec_kernel_vec(z2, config) * p_safe_vec(P_L, vecs[i]) * p_safe_vec(P_L, vecs[j]);
    }

    out.P13_k1_stats = post_i3_loop(P_L, norm_vec(vecs[0]), config);
    out.P13_k2_stats = post_i3_loop(P_L, norm_vec(vecs[1]), config);
    out.P13_k3_stats = post_i3_loop(P_L, norm_vec(vecs[2]), config);
    out.B321II = post_b123i_from_i3(P_L, vecs, out.P13_k1_stats, out.P13_k2_stats, out.P13_k3_stats, config);

    out.B222_stats = integrate_loop_component(
        [&P_L, &vecs, &config, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * post_b222_integrand(P_L, vecs, qv, config);
        },
        tri.k1,
        config);
    out.B321I_stats = integrate_loop_component(
        [&P_L, &vecs, &config, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * post_b321i_integrand(P_L, vecs, qv, config);
        },
        tri.k1,
        config);
    out.B411_stats = integrate_loop_component(
        [&P_L, &vecs, &config, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * post_b114_integrand(P_L, vecs, qv, config);
        },
        tri.k1,
        config);

    out.B222 = out.B222_stats.value;
    out.B321I = out.B321I_stats.value;
    out.B411 = out.B411_stats.value;
    out.Bloopterms = out.B222 + out.B321I + out.B411;
    out.B1loop = out.Bloopterms + out.B321II;
    out.Btotal = out.Btree + out.B1loop;
    return out;
}

/*
 * halo bias-v1 的 pre-reconstruction Gaussian 入口。
 * P_L、tri、config 分别控制线性谱、外部三角形与 loop 积分；bias 提供共享的
 * b1/b2/bK2。返回 tree 与四类 one-loop diagram 的逐分量结果。
 */
ComponentResult compute_pre_recon_halo_bias_v1_gaussian(
    const PowerSpectrum& P_L,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const GaussianKernelFunction kernel = [&bias](const std::vector<Vec3>& vectors) {
        return halo_bias_v1_kernel_vec(vectors, bias);
    };
    const Triangle routed = canonical_halo_v1_loop_triangle(tri);
    ComponentResult out = compute_gaussian_with_kernel(P_L, routed, config, kernel);
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    return out;
}

/*
 * halo bias-v1 的 post-reconstruction Gaussian 入口。
 * 它先用与 pre 相同的 b1/b2/bK2 构造 K_n，再按 config 中的 R、b_rec 与
 * mesh window 映射到 Z_rec,n；返回值字段与 pre 入口一致。
 */
ComponentResult compute_post_recon_halo_bias_v1_gaussian(
    const PowerSpectrum& P_L,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const GaussianKernelFunction kernel = [&config, &bias](const std::vector<Vec3>& vectors) {
        return halo_bias_v1_zrec_kernel_vec(vectors, config, bias);
    };
    const Triangle routed = canonical_halo_v1_loop_triangle(tri);
    ComponentResult out = compute_gaussian_with_kernel(P_L, routed, config, kernel);
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    return out;
}

ComponentResult compute_post_recon_gaussian_ir_nowiggle(
    const PowerSpectrum& P_L,
    const PowerSpectrum& P_nw,
    const Triangle& tri,
    const NativeConfig& config) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const std::vector<Vec3> vecs = triangle_vecs(tri);
    const real qscale = tri.k1;

    const ComponentStats i3[3] = {
        post_i3_loop(P_nw, norm_vec(vecs[0]), config),
        post_i3_loop(P_nw, norm_vec(vecs[1]), config),
        post_i3_loop(P_nw, norm_vec(vecs[2]), config),
    };
    out.P13_k1_stats = i3[0];
    out.P13_k2_stats = i3[1];
    out.P13_k3_stats = i3[2];

    const int pairs[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        const int ell = pairs[row][2];
        const real di = damping_ir(norm_vec(vecs[i]), config);
        const real dj = damping_ir(norm_vec(vecs[j]), config);
        const real dell = damping_ir(norm_vec(vecs[ell]), config);
        const real dprod = di * dj * dell;
        const real di2 = di * di;
        const real dj2 = dj * dj;
        const real pnwi = p_safe_vec(P_nw, vecs[i]);
        const real pnwj = p_safe_vec(P_nw, vecs[j]);
        const real pwi = p_wiggle_safe_vec(P_L, P_nw, vecs[i]);
        const real pwj = p_wiggle_safe_vec(P_L, P_nw, vecs[j]);
        const real bgg112 = pair_bgg112_coeff(vecs[i], vecs[j], config);

        const ComponentStats bgg114_stats = pair_bgg114_coeff(P_nw, vecs[i], vecs[j], qscale, config);
        const real bgg123i = pair_bgg123i_coeff_from_i3(vecs[i], vecs[j], i3[i], i3[j], config);
        const ComponentStats bgm_stats = pair_bgm123ii_coeff(P_nw, vecs[i], vecs[j], qscale, config);
        const ComponentStats bmg_stats = pair_bgm123ii_coeff(P_nw, vecs[j], vecs[i], qscale, config);

        const real bgg114 = bgg114_stats.value;
        const real bgm = bgm_stats.value;
        const real bmg = bmg_stats.value;
        const real gg_loop = bgg114 + bgg123i;
        const real dprod_log = std::log(std::max<real>(dprod, 1.0e-300));
        const real di2_log = std::log(std::max<real>(di2, 1.0e-300));
        const real dj2_log = std::log(std::max<real>(dj2, 1.0e-300));

        const real tree_pair = bgg112 * (dprod * pwi * pwj + di2 * pwi * pnwj + dj2 * pnwi * pwj + pnwi * pnwj);
        const real bgg114_pair = bgg114 * (dprod * pwi * pwj + di2 * pwi * pnwj + dj2 * pnwi * pwj + pnwi * pnwj);
        const real bgg123i_pair = bgg123i * (dprod * pwi * pwj + di2 * pwi * pnwj + dj2 * pnwi * pwj + pnwi * pnwj);
        const real bgm_bmg_pair = bgm * (di2 * pwi + pnwi) + bmg * (dj2 * pwj + pnwj);
        const real total_pair =
            ((1.0 - dprod_log) * bgg112 + gg_loop) * dprod * pwi * pwj
            + ((1.0 - di2_log) * bgg112 + gg_loop + bgm / std::max<real>(pnwj, 1.0e-300)) * di2 * pwi * pnwj
            + ((1.0 - dj2_log) * bgg112 + gg_loop + bmg / std::max<real>(pnwi, 1.0e-300)) * dj2 * pnwi * pwj
            + bgg112 * pnwi * pnwj
            + gg_loop * pnwi * pnwj
            + bgm * pnwi
            + bmg * pnwj;

        out.Btree += tree_pair;
        out.B411 += bgg114_pair;
        out.B321II += bgg123i_pair;
        out.B321I += bgm_bmg_pair;
        out.Btotal += total_pair;
        out.B411_stats = add_stats(out.B411_stats, bgg114_stats);
        out.B321I_stats = add_stats(out.B321I_stats, bgm_stats);
        out.B321I_stats = add_stats(out.B321I_stats, bmg_stats);
    }

    out.B222_stats = integrate_loop_component(
        [&P_nw, &vecs, &config, &tri](real r, real u, real o) {
            const Vec3 qv = q_vec(tri.k1 * r, u, o);
            return pow2(r) * post_b222_integrand(P_nw, vecs, qv, config);
        },
        tri.k1,
        config);
    out.B222 = out.B222_stats.value;
    out.Btotal += out.B222;
    out.Bloopterms = out.B222 + out.B321I + out.B411;
    out.B1loop = out.Btotal - out.Btree;
    return out;
}

real local_png_tree_dfNL(const PowerSpectrum& P_L, const PowerSpectrum& transfer_m, const Triangle& tri, real b1) {
    const real k3 = triangle_k3(tri);
    return local_png_b0_dfNL(P_L, transfer_m, tri.k1, tri.k2, k3, b1);
}

/*
 * 严格实现 Barreira 2022 Appendix B.5/B.6 的 pre-recon tree 模型。
 *
 * P_L 与 transfer_m 分别提供 P_mm(k,z) 和 M(k,z)，tri 是闭合三角形，bias
 * 包含 b1/b2/bK2/bphi/bphidelta。返回 Gaussian tree、线性 fNL 响应的逐项
 * 分解，以及尚未乘 alpha3/nbar、alpha4/nbar^2 的 stochastic basis。
 * 本函数不调用 loop integration，也不会把 primordial B111 重复写入 K2_phi。
 */
ComponentResult compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const HaloBiasV1Params& bias) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const std::vector<Vec3> vecs = triangle_vecs(tri);
    const int pairs[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    real p[3] = {p_safe_vec(P_L, vecs[0]), p_safe_vec(P_L, vecs[1]), p_safe_vec(P_L, vecs[2])};
    real m[3] = {p_safe_vec(transfer_m, vecs[0]), p_safe_vec(transfer_m, vecs[1]), p_safe_vec(transfer_m, vecs[2])};
    real b0hat = 0.0;
    real inv_m_all = 0.0;
    for (int index = 0; index < 3; ++index) {
        if (m[index] <= 0.0) return out;
        inv_m_all += 1.0 / m[index];
        out.stochastic_alpha3_basis += bias.b1 * bias.b1 * p[index];
        out.dBdfNL_stochastic_alpha3_basis += bias.b1 * bias.bphi * p[index] / m[index];
        out.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis +=
            pow2(bias.bphi) * p[index] / pow2(m[index]);
    }
    out.stochastic_alpha4_basis = 1.0;

    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        const int k = pairs[row][2];
        std::vector<Vec3> k1_i(1, vecs[i]);
        std::vector<Vec3> k1_j(1, vecs[j]);
        std::vector<Vec3> k2_args;
        k2_args.push_back(vecs[i]);
        k2_args.push_back(vecs[j]);
        out.Btree += 2.0
                     * halo_bias_v1_kernel_vec(k1_i, bias)
                     * halo_bias_v1_kernel_vec(k1_j, bias)
                     * halo_bias_v1_kernel_vec(k2_args, bias)
                     * p[i] * p[j];

        const real inv_m_sum = 1.0 / m[i] + 1.0 / m[j];
        const real f2 = f2_vec(vecs[i], vecs[j]);
        const real s2 = tidal_s2_vec(vecs[i], vecs[j]);
        const real mu = cosine_vecs(vecs[i], vecs[j]);
        const real ki = norm_vec(vecs[i]);
        const real kj = norm_vec(vecs[j]);
        const real pp = p[i] * p[j];
        const real inv_m_product = 1.0 / (m[i] * m[j]);

        const real b0hat_pair = 2.0 * pp * m[k] / (m[i] * m[j]);
        b0hat += b0hat_pair;
        out.dBdfNL_local_primordial += pow3(bias.b1) * b0hat_pair;
        out.dBdfNL_local_bphi_f2 += pow2(bias.b1) * bias.bphi * pp * 2.0 * f2 * inv_m_sum;
        out.dBdfNL_local_bphi_advection +=
            pow2(bias.b1) * bias.bphi * pp * mu
            * (ki / (kj * m[i]) + kj / (ki * m[j]));
        out.dBdfNL_local_bphidelta += pow2(bias.b1) * bias.bphidelta * pp * inv_m_sum;
        out.dBdfNL_local_bphi_b2 += bias.b1 * bias.b2 * bias.bphi * pp * inv_m_sum;
        out.dBdfNL_local_bphi_bK2 += 2.0 * bias.b1 * bias.bK2 * bias.bphi * s2 * pp * inv_m_sum;

        /*
         * Fixed-bias tree-level finite-local-PNG coefficient from
         * Dizgah et al. (2020), Eq. (2.58).  The bphi*B0 term multiplies
         * the full cyclic B0hat and is assembled below.
         */
        out.Bhalo_tree_fNL2_bphi_sq_advection +=
            bias.b1 * pow2(bias.bphi) * pp * inv_m_sum * mu
            * (ki / (kj * m[i]) + kj / (ki * m[j]));
        out.Bhalo_tree_fNL2_bphi_sq_F2 +=
            2.0 * pow2(bias.bphi) * inv_m_product
            * bias.b1 * f2 * pp;
        out.Bhalo_tree_fNL2_bphi_sq_b2 +=
            pow2(bias.bphi) * bias.b2 * inv_m_product * pp;
        out.Bhalo_tree_fNL2_bphi_sq_bK2 +=
            2.0 * pow2(bias.bphi) * bias.bK2 * s2
            * inv_m_product * pp;
        out.Bhalo_tree_fNL2_bphi_bphidelta +=
            bias.b1 * bias.bphi * bias.bphidelta
            * pow2(inv_m_sum) * pp;
        out.Bhalo_tree_fNL2_bphi2_operator +=
            pow2(bias.b1) * bias.bphi2 * inv_m_product * pp;
    }

    out.Bhalo_tree_fNL2_bphi_B0 =
        pow2(bias.b1) * bias.bphi * inv_m_all * b0hat;
    out.Bhalo_tree_fNL2_deterministic =
        out.Bhalo_tree_fNL2_bphi_B0
        + out.Bhalo_tree_fNL2_bphi_sq_advection
        + out.Bhalo_tree_fNL2_bphi_sq_F2
        + out.Bhalo_tree_fNL2_bphi_sq_b2
        + out.Bhalo_tree_fNL2_bphi_sq_bK2
        + out.Bhalo_tree_fNL2_bphi_bphidelta
        + out.Bhalo_tree_fNL2_bphi2_operator;

    out.Btotal = out.Btree;
    out.dBdfNL_local_tree =
        out.dBdfNL_local_primordial
        + out.dBdfNL_local_bphi_f2
        + out.dBdfNL_local_bphi_advection
        + out.dBdfNL_local_bphidelta
        + out.dBdfNL_local_bphi_b2
        + out.dBdfNL_local_bphi_bK2;
    out.dBdfNL_local_total = out.dBdfNL_local_tree;
    return out;
}

/*
 * 将 Barreira K1/K2 的线性 PNG 响应传播到标准 post-reconstruction tree。
 *
 * 与 pre 入口相比，多出的 config 指定 R、b_rec 与 mesh window。函数对
 * Z1^G/Z1^phi 和 Z2^G/Z2^phi 做显式一阶展开，所以不会用 finite fNL 差分，
 * 也不会引入论文未定义的 K3^phi/K4^phi。返回 reconstruction 专属的 bphi
 * 分量，便于与 R->infinity 的 pre 极限逐项核对。
 */
static ComponentResult
compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors_impl(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const std::vector<Vec3>& vecs,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const int pairs[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    real p[3] = {p_safe_vec(P_L, vecs[0]), p_safe_vec(P_L, vecs[1]), p_safe_vec(P_L, vecs[2])};
    real m[3] = {p_safe_vec(transfer_m, vecs[0]), p_safe_vec(transfer_m, vecs[1]), p_safe_vec(transfer_m, vecs[2])};
    for (int index = 0; index < 3; ++index) {
        if (m[index] <= 0.0) return out;
        out.stochastic_alpha3_basis += bias.b1 * bias.b1 * p[index];
        out.dBdfNL_stochastic_alpha3_basis += bias.b1 * bias.bphi * p[index] / m[index];
    }
    out.stochastic_alpha4_basis = 1.0;

    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        const int k = pairs[row][2];
        const real ki = norm_vec(vecs[i]);
        const real kj = norm_vec(vecs[j]);
        const real mu = cosine_vecs(vecs[i], vecs[j]);
        const real f2 = f2_vec(vecs[i], vecs[j]);
        const real s2 = tidal_s2_vec(vecs[i], vecs[j]);
        const real pp = p[i] * p[j];
        const real inv_m_sum = 1.0 / m[i] + 1.0 / m[j];
        const real k1phi_i = bias.bphi / m[i];
        const real k1phi_j = bias.bphi / m[j];

        std::vector<Vec3> pair_vectors;
        pair_vectors.push_back(vecs[i]);
        pair_vectors.push_back(vecs[j]);
        const Vec3 total_vec = sum_vecs(pair_vectors);
        std::vector<Vec3> single_i(1, vecs[i]);
        std::vector<Vec3> single_j(1, vecs[j]);
        const real shift_i = shift_factor_post(total_vec, single_i, config);
        const real shift_j = shift_factor_post(total_vec, single_j, config);
        const real shift_sum = shift_i + shift_j;

        const real z2g_f2 = bias.b1 * f2;
        const real z2g_b2 = 0.5 * bias.b2;
        const real z2g_bK2 = bias.bK2 * s2;
        const real z2g_recon = 0.5 * bias.b1 * bias.b1 * shift_sum;
        const real z2g_total = z2g_f2 + z2g_b2 + z2g_bK2 + z2g_recon;
        const real z2phi_bphidelta = 0.5 * bias.bphidelta * inv_m_sum;
        const real z2phi_advection = 0.5 * bias.bphi * mu
                                       * (ki / (kj * m[i]) + kj / (ki * m[j]));
        const real z2phi_recon = 0.5 * bias.b1 * (k1phi_i + k1phi_j) * shift_sum;

        out.Btree += 2.0 * bias.b1 * bias.b1 * z2g_total * pp;
        out.dBdfNL_local_primordial += 2.0 * pow3(bias.b1) * pp * m[k] / (m[i] * m[j]);

        const real external_prefactor = 2.0 * bias.b1 * (k1phi_i + k1phi_j) * pp;
        out.dBdfNL_local_bphi_f2 += external_prefactor * z2g_f2;
        out.dBdfNL_local_bphi_b2 += external_prefactor * z2g_b2;
        out.dBdfNL_local_bphi_bK2 += external_prefactor * z2g_bK2;
        out.dBdfNL_local_bphi_reconstruction += external_prefactor * z2g_recon;

        const real second_order_prefactor = 2.0 * bias.b1 * bias.b1 * pp;
        out.dBdfNL_local_bphidelta += second_order_prefactor * z2phi_bphidelta;
        out.dBdfNL_local_bphi_advection += second_order_prefactor * z2phi_advection;
        out.dBdfNL_local_bphi_reconstruction += second_order_prefactor * z2phi_recon;

        /*
         * Exact fNL^2 coefficient of
         *
         *   2 P_i P_j Z1_i Z1_j Z2_ij
         *
         * after applying the standard-reconstruction product rule to K1 and
         * K2.  The two sources are (i) the fNL^2 part of the reconstructed
         * Z2 kernel and (ii) products of the external K1 responses with the
         * linear reconstructed Z2 response.  Their sum simplifies to the
         * symmetric expression below.
         */
        out.Bhalo_tree_fNL2_bphi_sq_reconstruction +=
            pp * shift_sum * pow2(bias.b1) * pow2(bias.bphi)
            * (1.0 / pow2(m[i])
               + 4.0 / (m[i] * m[j])
               + 1.0 / pow2(m[j]));
    }

    /*
     * All non-reconstruction finite-fNL tree terms are identical to the
     * pre-reconstruction Dizgah et al. basis.  Evaluate that already
     * source-audited decomposition once, then add the genuinely new
     * finite-R component above.
     */
    const ComponentResult pre_quadratic =
        compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(
            P_L, transfer_m, tri, bias);
    out.Bhalo_tree_fNL2_bphi_B0 =
        pre_quadratic.Bhalo_tree_fNL2_bphi_B0;
    out.Bhalo_tree_fNL2_bphi_sq_advection =
        pre_quadratic.Bhalo_tree_fNL2_bphi_sq_advection;
    out.Bhalo_tree_fNL2_bphi_sq_F2 =
        pre_quadratic.Bhalo_tree_fNL2_bphi_sq_F2;
    out.Bhalo_tree_fNL2_bphi_sq_b2 =
        pre_quadratic.Bhalo_tree_fNL2_bphi_sq_b2;
    out.Bhalo_tree_fNL2_bphi_sq_bK2 =
        pre_quadratic.Bhalo_tree_fNL2_bphi_sq_bK2;
    out.Bhalo_tree_fNL2_bphi_bphidelta =
        pre_quadratic.Bhalo_tree_fNL2_bphi_bphidelta;
    out.Bhalo_tree_fNL2_bphi2_operator =
        pre_quadratic.Bhalo_tree_fNL2_bphi2_operator;
    out.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis =
        pre_quadratic.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis;
    out.Bhalo_tree_fNL2_deterministic =
        pre_quadratic.Bhalo_tree_fNL2_deterministic
        +out.Bhalo_tree_fNL2_bphi_sq_reconstruction;

    out.Btotal = out.Btree;
    out.dBdfNL_local_tree =
        out.dBdfNL_local_primordial
        + out.dBdfNL_local_bphi_f2
        + out.dBdfNL_local_bphi_advection
        + out.dBdfNL_local_bphidelta
        + out.dBdfNL_local_bphi_b2
        + out.dBdfNL_local_bphi_bK2
        + out.dBdfNL_local_bphi_reconstruction;
    out.dBdfNL_local_total = out.dBdfNL_local_tree;
    return out;
}

ComponentResult compute_post_recon_halo_bias_v1_local_png_tree_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    return
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors_impl(
            P_L,
            transfer_m,
            tri,
            triangle_vecs(tri),
            config,
            bias);
}

ComponentResult
compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const std::vector<Vec3> vecs=triangle_vecs(vectors);
    validate_closed_triangle_vecs(vecs,config.singular_floor);
    const Triangle tri=triangle_from_vecs(vecs);
    return
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors_impl(
            P_L,
            transfer_m,
            tri,
            vecs,
            config,
            bias);
}

AdaptiveBrecFiniteTreeBasis
compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& config) {
    const std::vector<Vec3> vecs=triangle_vecs(vectors);
    validate_closed_triangle_vecs(vecs,config.singular_floor);
    const int pairs[3][3]={{0,1,2},{1,2,0},{2,0,1}};
    real p[3]={
        p_safe_vec(P_L,vecs[0]),
        p_safe_vec(P_L,vecs[1]),
        p_safe_vec(P_L,vecs[2])};
    real m[3]={
        p_safe_vec(transfer_m,vecs[0]),
        p_safe_vec(transfer_m,vecs[1]),
        p_safe_vec(transfer_m,vecs[2])};
    for (int index=0;index<3;++index) {
        if (!(m[index]>0.0)) return {};
    }

    AdaptiveBrecFiniteTreeBasis basis;
    for (int row=0;row<3;++row) {
        const int i=pairs[row][0];
        const int j=pairs[row][1];
        std::vector<Vec3> pair_vectors{vecs[i],vecs[j]};
        const Vec3 total_vec=sum_vecs(pair_vectors);
        const real shift_i=shift_factor_post(
            total_vec,std::vector<Vec3>{vecs[i]},config);
        const real shift_j=shift_factor_post(
            total_vec,std::vector<Vec3>{vecs[j]},config);
        const real pp=p[i]*p[j];
        const real inverse_m_sum=1.0/m[i]+1.0/m[j];
        const real shift_over_m=
            shift_i/m[i]+shift_j/m[j];
        basis.gaussian_linear-=pp*shift_over_m;
        basis.gaussian_quadratic+=
            pp*(shift_i/(m[i]*m[i])+shift_j/(m[j]*m[j]));
        basis.png_linear_cross-=
            2.0*pp*inverse_m_sum*shift_over_m;
    }
    return basis;
}

real
compute_post_recon_halo_local_png_brec_denominator_tree_basis_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& config) {
    return
        compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(
            P_L,transfer_m,vectors,config).gaussian_linear;
}

/*
 * 把 response-only 结果及严格 tree 物理分解合并进 Gaussian 一环结果。
 * 分开的 helper 可确保 pre/post 两个入口不会在字段语义上漂移。
 */
void merge_halo_bias_v1_local_png_response(
    ComponentResult& out,
    const ComponentResult& response,
    const ComponentResult& exact_tree) {
    out.dBdfNL_local_tree = response.dBdfNL_local_tree;
    out.dBdfNL_local_B222 = response.dBdfNL_local_B222;
    out.dBdfNL_local_B321I = response.dBdfNL_local_B321I;
    out.dBdfNL_local_B321II = response.dBdfNL_local_B321II;
    out.dBdfNL_local_B411 = response.dBdfNL_local_B411;
    out.dBdfNL_local_1loop = response.dBdfNL_local_1loop;
    out.dBdfNL_local_total = response.dBdfNL_local_total;
    out.dBdfNL_local_B222_stats = response.dBdfNL_local_B222_stats;
    out.dBdfNL_local_B321I_stats = response.dBdfNL_local_B321I_stats;
    out.dBdfNL_local_B321II_stats = response.dBdfNL_local_B321II_stats;
    out.dBdfNL_local_B411_stats = response.dBdfNL_local_B411_stats;
    out.dP13dfNL_local_k1_stats = response.dP13dfNL_local_k1_stats;
    out.dP13dfNL_local_k2_stats = response.dP13dfNL_local_k2_stats;
    out.dP13dfNL_local_k3_stats = response.dP13dfNL_local_k3_stats;

    /* 以下字段是 tree-level 可解释分解；loop 只按 diagram 分解。 */
    out.dBdfNL_local_primordial = exact_tree.dBdfNL_local_primordial;
    out.dBdfNL_local_bphi_f2 = exact_tree.dBdfNL_local_bphi_f2;
    out.dBdfNL_local_bphi_advection = exact_tree.dBdfNL_local_bphi_advection;
    out.dBdfNL_local_bphidelta = exact_tree.dBdfNL_local_bphidelta;
    out.dBdfNL_local_bphi_b2 = exact_tree.dBdfNL_local_bphi_b2;
    out.dBdfNL_local_bphi_bK2 = exact_tree.dBdfNL_local_bphi_bK2;
    out.dBdfNL_local_bphi_reconstruction = exact_tree.dBdfNL_local_bphi_reconstruction;
    out.stochastic_alpha3_basis = exact_tree.stochastic_alpha3_basis;
    out.stochastic_alpha4_basis = exact_tree.stochastic_alpha4_basis;
    out.dBdfNL_stochastic_alpha3_basis = exact_tree.dBdfNL_stochastic_alpha3_basis;
    out.Bhalo_tree_fNL2_bphi_B0 =
        exact_tree.Bhalo_tree_fNL2_bphi_B0;
    out.Bhalo_tree_fNL2_bphi_sq_advection =
        exact_tree.Bhalo_tree_fNL2_bphi_sq_advection;
    out.Bhalo_tree_fNL2_bphi_sq_F2 =
        exact_tree.Bhalo_tree_fNL2_bphi_sq_F2;
    out.Bhalo_tree_fNL2_bphi_sq_b2 =
        exact_tree.Bhalo_tree_fNL2_bphi_sq_b2;
    out.Bhalo_tree_fNL2_bphi_sq_bK2 =
        exact_tree.Bhalo_tree_fNL2_bphi_sq_bK2;
    out.Bhalo_tree_fNL2_bphi_sq_reconstruction =
        exact_tree.Bhalo_tree_fNL2_bphi_sq_reconstruction;
    out.Bhalo_tree_fNL2_bphi_bphidelta =
        exact_tree.Bhalo_tree_fNL2_bphi_bphidelta;
    out.Bhalo_tree_fNL2_bphi2_operator =
        exact_tree.Bhalo_tree_fNL2_bphi2_operator;
    out.Bhalo_tree_fNL2_deterministic =
        exact_tree.Bhalo_tree_fNL2_deterministic;
    out.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis =
        exact_tree.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis;
}

/*
 * pre-reconstruction halo bias-v1 local-PNG：Gaussian tree+完整一环基底，
 * 并对同一 bispectrum 泛函逐图求 O(fNL) 响应。Barreira 只固定 K1/K2
 * tree 极限；K3/K4 PNG 在这里由上方明确写出的 A/C/IC 递推定义。
 */
ComponentResult compute_pre_recon_halo_bias_v1_local_png_1loop_truncated_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const GaussianKernelFunction gaussian_kernel = [&bias](const std::vector<Vec3>& vectors) {
        return halo_bias_v1_kernel_vec(vectors, bias);
    };
    const ResponseKernelFunction response_kernel =
        [&transfer_m, &bias](const std::vector<Vec3>& vectors) {
            return halo_bias_v1_local_png_kernel_response_vec(vectors, transfer_m, bias);
        };
    const Triangle routed = canonical_halo_v1_loop_triangle(tri);
    ComponentResult out = compute_gaussian_with_kernel(P_L, routed, config, gaussian_kernel);
    const ComponentResult response =
        compute_local_png_response_with_kernel(P_L, routed, config, response_kernel, out);
    const ComponentResult exact_tree =
        compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(P_L, transfer_m, routed, bias);
    merge_halo_bias_v1_local_png_response(out, response, exact_tree);
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    return out;
}

/*
 * post-reconstruction 对应入口。Gaussian 与 PNG response 都经过同一个
 * reconstruction partition；response 使用 product rule，b_rec/R/mesh window
 * 只作为固定设置。R->infinity 时所有 shift 因子消失并严格退化到 pre 入口。
 */
ComponentResult compute_post_recon_halo_bias_v1_local_png_1loop_truncated_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias) {
    const GaussianKernelFunction gaussian_kernel =
        [&config, &bias](const std::vector<Vec3>& vectors) {
            return halo_bias_v1_zrec_kernel_vec(vectors, config, bias);
        };
    const ResponseKernelFunction response_kernel =
        [&transfer_m, &config, &bias](const std::vector<Vec3>& vectors) {
            return halo_bias_v1_local_png_zrec_kernel_response_vec(
                vectors, transfer_m, config, bias);
        };
    const Triangle routed = canonical_halo_v1_loop_triangle(tri);
    ComponentResult out = compute_gaussian_with_kernel(P_L, routed, config, gaussian_kernel);
    const ComponentResult response =
        compute_local_png_response_with_kernel(P_L, routed, config, response_kernel, out);
    const ComponentResult exact_tree =
        compute_post_recon_halo_bias_v1_local_png_tree_dfNL(
            P_L, transfer_m, routed, config, bias);
    merge_halo_bias_v1_local_png_response(out, response, exact_tree);
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    return out;
}

ComponentStats P12_png_dfNL(const PowerSpectrum& P_L, const PowerSpectrum& transfer_m, real k, const NativeConfig& config, real b1) {
    typedef std::tuple<
        const PowerSpectrum*,const PowerSpectrum*,
        real,real,real,real,real,real,real> CacheKey;
    static std::map<CacheKey, ComponentStats> cache;
    const CacheKey key(
        &P_L,&transfer_m,k,config.qmin,config.qmax,
        local_png_ir_cutoff(config),
        config.epsrel,config.epsabs,b1);
    std::map<CacheKey, ComponentStats>::const_iterator found = cache.find(key);
    if (found != cache.end()) return found->second;

    ComponentStats out = integrate_loop_component(
        [&P_L, &transfer_m, &config, k, b1](
            real r, real u, real o) {
            return P12_png_integrand(
                P_L,transfer_m,k,config,b1,r,u,o);
        },
        k,
        config);
    cache[key] = out;
    return out;
}

ComponentStats P12_png_post_dfNL_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& kvec,
    const NativeConfig& config,
    real b1) {
    using NumericKey=
        std::pair<std::array<int,4>,std::array<real,14>>;
    using CacheKey=std::tuple<
        const PowerSpectrum*,const PowerSpectrum*,NumericKey>;
    static thread_local std::map<
        CacheKey,ComponentStats> cache;
    const CacheKey key{
        &P_L,
        &transfer_m,
        NumericKey{
            std::array<int,4>{{
                config.recon_cic_window_power,
                config.png_linear_multicenter_qmc ? 1 : 0,
                config.png_b112ii_qmc_power,
                config.png_b112ii_qmc_replicates}},
            std::array<real,14>{{
                kvec.x,kvec.y,kvec.z,
                config.qmin,config.qmax,
                local_png_ir_cutoff(config),
                config.epsrel,
                config.p13_epsrel,config.epsabs,
                config.smoothing_radius,config.bias_recon,
                config.recon_cellsize,config.singular_floor,b1}}}};
    std::map<CacheKey, ComponentStats>::const_iterator found =
        cache.find(key);
    if (found != cache.end()) return found->second;
    const real k=norm_vec(kvec);
    if (k<=config.singular_floor) return ComponentStats();
    if (config.png_linear_multicenter_qmc) {
        const real png_cutoff=local_png_ir_cutoff(config);
        std::vector<LocalPngQmcCenter> centers;
        centers.push_back(
            LocalPngQmcCenter{
                make_vec(0.0,0.0,0.0),png_cutoff});
        centers.push_back(
            LocalPngQmcCenter{kvec,png_cutoff});
        ComponentStats out=
            integrate_local_png_multicenter_qmc(
                [&P_L,&transfer_m,&kvec,&config,b1](
                    const Vec3& qv) {
                    return
                        P12_png_post_integrand_physical_vec(
                            P_L,
                            transfer_m,
                            kvec,
                            config,
                            b1,
                            qv);
                },
                centers,
                config);
        cache[key]=out;
        return out;
    }
    ComponentStats out = integrate_loop_component(
        [&P_L, &transfer_m, &config, kvec, k, b1](
            real r,real u,real o) {
            return P12_png_post_integrand_vec(
                P_L,transfer_m,kvec,k,config,b1,r,u,o);
        },
        k,
        config);
    cache[key] = out;
    return out;
}

ComponentStats P12_png_post_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real k,
    const NativeConfig& config,
    real b1) {
    return P12_png_post_dfNL_vec(
        P_L,transfer_m,make_vec(0.0,0.0,k),config,b1);
}

ComponentStats B122II_png_ordered_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1) {
    return integrate_loop_component(
        [&P_L, &transfer_m, &config, ka, kb, xab, b1](
            real r, real u, real o) {
            return B122II_png_ordered_integrand(
                P_L,transfer_m,ka,kb,xab,config,b1,r,u,o);
        },
        ka,
        config);
}

ComponentStats B122II_png_post_ordered_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1) {
    return integrate_loop_component(
        [&P_L, &transfer_m, &config, ka, kb, xab, b1](real r, real u, real o) {
            return B122II_png_post_ordered_integrand(P_L, transfer_m, ka, kb, xab, config, b1, r, u, o);
        },
        ka,
        config);
}

ComponentStats B122II_png_post_ordered_dfNL_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    const NativeConfig& config,
    real b1) {
    const real scale=norm_vec(avec);
    if (scale<=config.singular_floor) return ComponentStats();
    if (config.png_linear_multicenter_qmc) {
        const real png_cutoff=local_png_ir_cutoff(config);
        std::vector<LocalPngQmcCenter> centers;
        centers.push_back(
            LocalPngQmcCenter{
                make_vec(0.0,0.0,0.0),png_cutoff});
        centers.push_back(
            LocalPngQmcCenter{neg_vec(avec),png_cutoff});
        centers.push_back(
            LocalPngQmcCenter{bvec,config.qmin});
        return
            integrate_local_png_multicenter_qmc(
                [&P_L,&transfer_m,&avec,&bvec,&config,b1](
                    const Vec3& qv) {
                    return
                        B122II_png_post_ordered_integrand_physical_vec(
                            P_L,
                            transfer_m,
                            avec,
                            bvec,
                            config,
                            b1,
                            qv);
                },
                centers,
                config);
    }
    return integrate_loop_component(
        [&P_L,&transfer_m,&config,avec,bvec,scale,b1](
            real r,real u,real o) {
            return B122II_png_post_ordered_integrand_vec(
                P_L,transfer_m,avec,bvec,scale,
                config,b1,r,u,o);
        },
        scale,
        config);
}

ComponentStats B113II_png_ordered_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1) {
    return integrate_loop_component(
        [&P_L, &transfer_m, &config, ka, kb, xab, b1](
            real r, real u, real o) {
            return B113II_png_ordered_integrand(
                P_L,transfer_m,ka,kb,xab,config,b1,r,u,o);
        },
        ka,
        config);
}

ComponentStats B113II_png_post_ordered_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    real ka,
    real kb,
    real xab,
    const NativeConfig& config,
    real b1) {
    return integrate_loop_component(
        [&P_L, &transfer_m, &config, ka, kb, xab, b1](real r, real u, real o) {
            return B113II_png_post_ordered_integrand(P_L, transfer_m, ka, kb, xab, config, b1, r, u, o);
        },
        ka,
        config);
}

ComponentStats B113II_png_post_ordered_dfNL_vec(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Vec3& avec,
    const Vec3& bvec,
    const NativeConfig& config,
    real b1) {
    const real scale=norm_vec(avec);
    if (scale<=config.singular_floor) return ComponentStats();
    if (config.png_linear_multicenter_qmc) {
        const real png_cutoff=local_png_ir_cutoff(config);
        std::vector<LocalPngQmcCenter> centers;
        centers.push_back(
            LocalPngQmcCenter{
                make_vec(0.0,0.0,0.0),png_cutoff});
        centers.push_back(
            LocalPngQmcCenter{bvec,png_cutoff});
        return
            integrate_local_png_multicenter_qmc(
                [&P_L,&transfer_m,&avec,&bvec,&config,b1](
                    const Vec3& qv) {
                    return
                        B113II_png_post_ordered_integrand_physical_vec(
                            P_L,
                            transfer_m,
                            avec,
                            bvec,
                            config,
                            b1,
                            qv);
                },
                centers,
                config);
    }
    return integrate_loop_component(
        [&P_L,&transfer_m,&config,avec,bvec,scale,b1](
            real r,real u,real o) {
            return B113II_png_post_ordered_integrand_vec(
                P_L,transfer_m,avec,bvec,scale,
                config,b1,r,u,o);
        },
        scale,
        config);
}

ComponentResult compute_pre_recon_local_png_1loop_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    real b1) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const real k1 = tri.k1;
    const real k2 = tri.k2;
    const real k3 = out.k3;
    const real x12 = tri.mu12;
    const real x23 = -(k1 * x12 + k2) / k3;
    const real x31 = -(k1 + k2 * x12) / k3;

    const real f2_12 = F2eds(k1, k2, x12);
    const real f2_23 = F2eds(k2, k3, x23);
    const real f2_31 = F2eds(k3, k1, x31);
    out.Btree = 2.0 * (p_safe(P_L, k1) * p_safe(P_L, k2) * f2_12
                       + p_safe(P_L, k2) * p_safe(P_L, k3) * f2_23
                       + p_safe(P_L, k3) * p_safe(P_L, k1) * f2_31);
    out.P13_k1_stats = P13_dd(P_L, k1, config);
    out.P13_k2_stats = P13_dd(P_L, k2, config);
    out.P13_k3_stats = P13_dd(P_L, k3, config);

    out.dBdfNL_local_tree = local_png_tree_dfNL(P_L, transfer_m, tri, b1);

    out.P12_png_k1_stats = P12_png_dfNL(P_L, transfer_m, k1, config, b1);
    out.P12_png_k2_stats = P12_png_dfNL(P_L, transfer_m, k2, config, b1);
    out.P12_png_k3_stats = P12_png_dfNL(P_L, transfer_m, k3, config, b1);

    out.dBdfNL_local_B122I =
        f2_12 * (p_safe(P_L, k1) * out.P12_png_k2_stats.value + p_safe(P_L, k2) * out.P12_png_k1_stats.value)
        + f2_23 * (p_safe(P_L, k2) * out.P12_png_k3_stats.value + p_safe(P_L, k3) * out.P12_png_k2_stats.value)
        + f2_31 * (p_safe(P_L, k3) * out.P12_png_k1_stats.value + p_safe(P_L, k1) * out.P12_png_k3_stats.value);
    out.dBdfNL_local_B122I_stats.value =
        out.dBdfNL_local_B122I;
    out.dBdfNL_local_B122I_stats.abserr =
        std::fabs(f2_12)
            * (std::fabs(p_safe(P_L,k1))
                   *out.P12_png_k2_stats.abserr
               +std::fabs(p_safe(P_L,k2))
                   *out.P12_png_k1_stats.abserr)
        +std::fabs(f2_23)
            * (std::fabs(p_safe(P_L,k2))
                   *out.P12_png_k3_stats.abserr
               +std::fabs(p_safe(P_L,k3))
                   *out.P12_png_k2_stats.abserr)
        +std::fabs(f2_31)
            * (std::fabs(p_safe(P_L,k3))
                   *out.P12_png_k1_stats.abserr
               +std::fabs(p_safe(P_L,k1))
                   *out.P12_png_k3_stats.abserr);
    out.dBdfNL_local_B122I_stats.neval =
        out.P12_png_k1_stats.neval
        +out.P12_png_k2_stats.neval
        +out.P12_png_k3_stats.neval;

    const real safe_p1 = std::max<real>(p_safe(P_L, k1), kTiny);
    const real safe_p2 = std::max<real>(p_safe(P_L, k2), kTiny);
    const real safe_p3 = std::max<real>(p_safe(P_L, k3), kTiny);
    out.dBdfNL_local_B113I = 0.5 * out.dBdfNL_local_tree
                              * (out.P13_k1_stats.value / safe_p1
                                 + out.P13_k2_stats.value / safe_p2
                                 + out.P13_k3_stats.value / safe_p3);
    out.dBdfNL_local_B113I_stats.value =
        out.dBdfNL_local_B113I;
    out.dBdfNL_local_B113I_stats.abserr =
        0.5*std::fabs(out.dBdfNL_local_tree)
        *(out.P13_k1_stats.abserr/safe_p1
          +out.P13_k2_stats.abserr/safe_p2
          +out.P13_k3_stats.abserr/safe_p3);
    out.dBdfNL_local_B113I_stats.neval =
        out.P13_k1_stats.neval
        +out.P13_k2_stats.neval
        +out.P13_k3_stats.neval;

    ComponentStats b122ii_12 = B122II_png_ordered_dfNL(P_L, transfer_m, k1, k2, x12, config, b1);
    ComponentStats b122ii_23 = B122II_png_ordered_dfNL(P_L, transfer_m, k2, k3, x23, config, b1);
    ComponentStats b122ii_31 = B122II_png_ordered_dfNL(P_L, transfer_m, k3, k1, x31, config, b1);
    out.dBdfNL_local_B122II_stats = add_stats(add_stats(b122ii_12, b122ii_23), b122ii_31);
    out.dBdfNL_local_B122II = out.dBdfNL_local_B122II_stats.value;

    ComponentStats b113ii_12 = B113II_png_ordered_dfNL(P_L, transfer_m, k1, k2, x12, config, b1);
    ComponentStats b113ii_21 = B113II_png_ordered_dfNL(P_L, transfer_m, k2, k1, x12, config, b1);
    ComponentStats b113ii_23 = B113II_png_ordered_dfNL(P_L, transfer_m, k2, k3, x23, config, b1);
    ComponentStats b113ii_32 = B113II_png_ordered_dfNL(P_L, transfer_m, k3, k2, x23, config, b1);
    ComponentStats b113ii_31 = B113II_png_ordered_dfNL(P_L, transfer_m, k3, k1, x31, config, b1);
    ComponentStats b113ii_13 = B113II_png_ordered_dfNL(P_L, transfer_m, k1, k3, x31, config, b1);
    out.dBdfNL_local_B113II_stats =
        add_stats(add_stats(add_stats(b113ii_12, b113ii_21), add_stats(b113ii_23, b113ii_32)), add_stats(b113ii_31, b113ii_13));
    out.dBdfNL_local_B113II = out.dBdfNL_local_B113II_stats.value;

    out.dBdfNL_local_1loop = out.dBdfNL_local_B122I + out.dBdfNL_local_B122II + out.dBdfNL_local_B113I + out.dBdfNL_local_B113II;
    out.dBdfNL_local_total = out.dBdfNL_local_tree + out.dBdfNL_local_1loop;
    out.dBdfNL_local_1loop_stats =
        add_stats(
            add_stats(
                out.dBdfNL_local_B122I_stats,
                out.dBdfNL_local_B122II_stats),
            add_stats(
                out.dBdfNL_local_B113I_stats,
                out.dBdfNL_local_B113II_stats));
    out.dBdfNL_local_1loop_stats.value =
        out.dBdfNL_local_1loop;
    out.dBdfNL_local_total_stats =
        out.dBdfNL_local_1loop_stats;
    out.dBdfNL_local_total_stats.value =
        out.dBdfNL_local_total;
    return out;
}

static ComponentResult
compute_post_recon_local_png_1loop_dfNL_vectors_impl(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const std::vector<Vec3>& vecs,
    const NativeConfig& config,
    real b1) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);

    const int pairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    out.Btree = 0.0;
    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> z2;
        z2.push_back(vecs[i]);
        z2.push_back(vecs[j]);
        out.Btree += 2.0 * zrec_kernel_vec(z2, config) * p_safe_vec(P_L, vecs[i]) * p_safe_vec(P_L, vecs[j]);
    }

    out.dBdfNL_local_tree = local_png_b0_dfNL_vec(P_L, transfer_m, vecs[0], vecs[1], vecs[2], b1);

    out.P12_png_k1_stats = P12_png_post_dfNL_vec(
        P_L,transfer_m,vecs[0],config,b1);
    out.P12_png_k2_stats = P12_png_post_dfNL_vec(
        P_L,transfer_m,vecs[1],config,b1);
    out.P12_png_k3_stats = P12_png_post_dfNL_vec(
        P_L,transfer_m,vecs[2],config,b1);
    const ComponentStats p12[3] = {out.P12_png_k1_stats, out.P12_png_k2_stats, out.P12_png_k3_stats};

    for (int row = 0; row < 3; ++row) {
        const int i = pairs[row][0];
        const int j = pairs[row][1];
        std::vector<Vec3> z2;
        z2.push_back(vecs[i]);
        z2.push_back(vecs[j]);
        const real kernel=zrec_kernel_vec(z2,config);
        const real pi=p_safe_vec(P_L,vecs[i]);
        const real pj=p_safe_vec(P_L,vecs[j]);
        out.dBdfNL_local_B122I +=
            kernel*(pi*p12[j].value+pj*p12[i].value);
        out.dBdfNL_local_B122I_stats.abserr +=
            std::fabs(kernel)
            *(std::fabs(pi)*p12[j].abserr
              +std::fabs(pj)*p12[i].abserr);
    }
    out.dBdfNL_local_B122I_stats.value =
        out.dBdfNL_local_B122I;
    out.dBdfNL_local_B122I_stats.neval =
        p12[0].neval+p12[1].neval+p12[2].neval;

    out.P13_k1_stats = post_i3_loop_vec(P_L,vecs[0],config);
    out.P13_k2_stats = post_i3_loop_vec(P_L,vecs[1],config);
    out.P13_k3_stats = post_i3_loop_vec(P_L,vecs[2],config);
    out.dBdfNL_local_B113I = 3.0 * out.dBdfNL_local_tree
                              * (out.P13_k1_stats.value + out.P13_k2_stats.value + out.P13_k3_stats.value);
    out.dBdfNL_local_B113I_stats.value =
        out.dBdfNL_local_B113I;
    out.dBdfNL_local_B113I_stats.abserr =
        3.0*std::fabs(out.dBdfNL_local_tree)
        *(out.P13_k1_stats.abserr
          +out.P13_k2_stats.abserr
          +out.P13_k3_stats.abserr);
    out.dBdfNL_local_B113I_stats.neval =
        out.P13_k1_stats.neval
        +out.P13_k2_stats.neval
        +out.P13_k3_stats.neval;

    ComponentStats b122ii_12 =
        B122II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[0],vecs[1],config,b1);
    ComponentStats b122ii_23 =
        B122II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[1],vecs[2],config,b1);
    ComponentStats b122ii_31 =
        B122II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[2],vecs[0],config,b1);
    out.dBdfNL_local_B122II_stats = add_stats(add_stats(b122ii_12, b122ii_23), b122ii_31);
    out.dBdfNL_local_B122II = out.dBdfNL_local_B122II_stats.value;

    ComponentStats b113ii_12 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[0],vecs[1],config,b1);
    ComponentStats b113ii_21 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[1],vecs[0],config,b1);
    ComponentStats b113ii_23 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[1],vecs[2],config,b1);
    ComponentStats b113ii_32 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[2],vecs[1],config,b1);
    ComponentStats b113ii_31 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[2],vecs[0],config,b1);
    ComponentStats b113ii_13 =
        B113II_png_post_ordered_dfNL_vec(
            P_L,transfer_m,vecs[0],vecs[2],config,b1);
    out.dBdfNL_local_B113II_stats =
        add_stats(add_stats(add_stats(b113ii_12, b113ii_21), add_stats(b113ii_23, b113ii_32)), add_stats(b113ii_31, b113ii_13));
    out.dBdfNL_local_B113II = out.dBdfNL_local_B113II_stats.value;

    out.dBdfNL_local_1loop = out.dBdfNL_local_B122I + out.dBdfNL_local_B122II + out.dBdfNL_local_B113I + out.dBdfNL_local_B113II;
    out.dBdfNL_local_total = out.dBdfNL_local_tree + out.dBdfNL_local_1loop;
    out.dBdfNL_local_1loop_stats =
        add_stats(
            add_stats(
                out.dBdfNL_local_B122I_stats,
                out.dBdfNL_local_B122II_stats),
            add_stats(
                out.dBdfNL_local_B113I_stats,
                out.dBdfNL_local_B113II_stats));
    out.dBdfNL_local_1loop_stats.value =
        out.dBdfNL_local_1loop;
    out.dBdfNL_local_total_stats =
        out.dBdfNL_local_1loop_stats;
    out.dBdfNL_local_total_stats.value =
        out.dBdfNL_local_total;
    return out;
}

ComponentResult compute_post_recon_local_png_1loop_dfNL(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    real b1) {
    return compute_post_recon_local_png_1loop_dfNL_vectors_impl(
        P_L,
        transfer_m,
        tri,
        triangle_vecs(tri),
        config,
        b1);
}

ComponentResult compute_post_recon_local_png_1loop_dfNL_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& config,
    real b1) {
    const std::vector<Vec3> vecs=triangle_vecs(vectors);
    validate_closed_triangle_vecs(vecs,config.singular_floor);
    const Triangle tri=triangle_from_vecs(vecs);
    return compute_post_recon_local_png_1loop_dfNL_vectors_impl(
        P_L,transfer_m,tri,vecs,config,b1);
}

/*
 * finite-fNL B112II 的隔离诊断入口。
 *
 * 这里只计算已经提取 fNL^2 后的系数，不改写 Btree/B1loop/Btotal，也不与
 * 线性 dB/dfNL 字段相加。三个 ordered orientations 对应
 * (11|2)+(11|2) cyclic，并共享完全相同的 spherical q domain、primordial
 * propagator IR cutoff 和积分容差。
 */
static ComponentResult
compute_local_png_B112II_fNL2_coefficient_vectors_impl(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const std::vector<Vec3>& vecs,
    const NativeConfig& config,
    bool post_recon) {
    ComponentResult out;
    out.triangle = tri;
    out.k3 = triangle_k3(tri);
    const real ir_cutoff =
        config.png_ir_cutoff > 0.0 ? config.png_ir_cutoff : config.qmin;
    if (tri.k1 < ir_cutoff || tri.k2 < ir_cutoff || out.k3 < ir_cutoff) {
        return out;
    }
    if (config.png_b112ii_multicenter_qmc) {
        out.B112II_fNL2_coefficient_stats =
            local_png_B112II_fNL2_multicenter_qmc(
                P_L, transfer_m, vecs, config, post_recon);
    } else {
        const ComponentStats orientation_12_3 =
            local_png_B112II_fNL2_ordered(
                P_L, transfer_m, vecs[0], vecs[1], vecs[2], config, post_recon);
        const ComponentStats orientation_23_1 =
            local_png_B112II_fNL2_ordered(
                P_L, transfer_m, vecs[1], vecs[2], vecs[0], config, post_recon);
        const ComponentStats orientation_31_2 =
            local_png_B112II_fNL2_ordered(
                P_L, transfer_m, vecs[2], vecs[0], vecs[1], config, post_recon);
        out.B112II_fNL2_coefficient_stats =
            add_stats(
                add_stats(orientation_12_3, orientation_23_1),
                orientation_31_2);
    }
    out.B112II_fNL2_coefficient =
        out.B112II_fNL2_coefficient_stats.value;
    return out;
}

ComponentResult compute_local_png_B112II_fNL2_coefficient_impl(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    bool post_recon) {
    if (config.png_b112ii_multicenter_qmc) {
        const Triangle routed=
            canonical_halo_v1_loop_triangle(tri);
        return
            compute_local_png_B112II_fNL2_coefficient_vectors_impl(
                P_L,
                transfer_m,
                routed,
                triangle_vecs(routed),
                config,
                post_recon);
    }
    return
        compute_local_png_B112II_fNL2_coefficient_vectors_impl(
            P_L,
            transfer_m,
            tri,
            triangle_vecs(tri),
            config,
            post_recon);
}

ComponentResult compute_pre_recon_local_png_B112II_fNL2_coefficient(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config) {
    return compute_local_png_B112II_fNL2_coefficient_impl(
        P_L, transfer_m, tri, config, false);
}

ComponentResult compute_post_recon_local_png_B112II_fNL2_coefficient(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config) {
    return compute_local_png_B112II_fNL2_coefficient_impl(
        P_L, transfer_m, tri, config, true);
}

ComponentResult
compute_post_recon_local_png_B112II_fNL2_coefficient_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& config) {
    const std::vector<Vec3> vecs=triangle_vecs(vectors);
    validate_closed_triangle_vecs(vecs,config.singular_floor);
    const Triangle tri=triangle_from_vecs(vecs);
    return
        compute_local_png_B112II_fNL2_coefficient_vectors_impl(
            P_L,transfer_m,tri,vecs,config,true);
}

namespace {

/*
 * 将已经验证过的线性 response 与 B112II 二次系数接到 Gaussian DM 基线。
 * 这里只组合字段，不重新定义任何旧入口：Btree/B1loop/各 Gaussian diagram
 * 始终来自 Gaussian 计算，dBdfNL 字段始终是 fNL=0 的一阶系数。
 */
void merge_dm_local_png_finite_terms(
    ComponentResult& out,
    const ComponentResult& response,
    const ComponentResult& quadratic,
    real fNL) {
    out.dBdfNL_local_tree = response.dBdfNL_local_tree;
    out.dBdfNL_local_B122I = response.dBdfNL_local_B122I;
    out.dBdfNL_local_B122II = response.dBdfNL_local_B122II;
    out.dBdfNL_local_B113I = response.dBdfNL_local_B113I;
    out.dBdfNL_local_B113II = response.dBdfNL_local_B113II;
    out.dBdfNL_local_1loop = response.dBdfNL_local_1loop;
    out.dBdfNL_local_total = response.dBdfNL_local_total;
    out.P12_png_k1_stats = response.P12_png_k1_stats;
    out.P12_png_k2_stats = response.P12_png_k2_stats;
    out.P12_png_k3_stats = response.P12_png_k3_stats;
    out.dBdfNL_local_B122II_stats =
        response.dBdfNL_local_B122II_stats;
    out.dBdfNL_local_B113II_stats =
        response.dBdfNL_local_B113II_stats;

    out.B112II_fNL2_coefficient =
        quadratic.B112II_fNL2_coefficient;
    out.B112II_fNL2_coefficient_stats =
        quadratic.B112II_fNL2_coefficient_stats;

    const real gaussian_total = out.Btotal;
    out.Btotal =
        gaussian_total
        + fNL * out.dBdfNL_local_total
        + fNL * fNL * out.B112II_fNL2_coefficient;
}

ComponentResult compute_local_png_finite_1loop_impl(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    real fNL,
    bool post_recon) {
    ComponentResult out = post_recon
        ? compute_post_recon_gaussian(P_L, tri, config)
        : compute_pre_recon_gaussian(P_L, tri, config);
    const ComponentResult response = post_recon
        ? compute_post_recon_local_png_1loop_dfNL(
              P_L, transfer_m, tri, config, 1.0)
        : compute_pre_recon_local_png_1loop_dfNL(
              P_L, transfer_m, tri, config, 1.0);
    const ComponentResult quadratic =
        compute_local_png_B112II_fNL2_coefficient_impl(
            P_L, transfer_m, tri, config, post_recon);
    merge_dm_local_png_finite_terms(out, response, quadratic, fNL);
    return out;
}

}  // namespace

ComponentResult compute_pre_recon_local_png_finite_1loop(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    real fNL) {
    return compute_local_png_finite_1loop_impl(
        P_L, transfer_m, tri, config, fNL, false);
}

ComponentResult compute_post_recon_local_png_finite_1loop(
    const PowerSpectrum& P_L,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    real fNL) {
    return compute_local_png_finite_1loop_impl(
        P_L, transfer_m, tri, config, fNL, true);
}

}  // namespace marisa_b_native
