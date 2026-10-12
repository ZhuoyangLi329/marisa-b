#ifndef MARISA_B_TREE_PNG_IR_H
#define MARISA_B_TREE_PNG_IR_H

#include "marisa_b_native.h"
#include <cstddef>

namespace marisa_b_native {
// 模块大纲：逐腿谱输入；完整 tree 原子项；严格单 wiggle IR；native 结果适配。
// FixedOrder 仅用于同投影基准；StrictLo 是受控默认；FactorizedClosure 为系统学。
enum class TreeIrScheme { FixedOrder = 0, StrictLo = 1, FactorizedClosure = 2 };
enum TreeIrTerm : std::size_t {
    GPre, GReconstruction, LPrimordial, LPhiF2, LPhiAdvection, LPhiDelta,
    LPhiB2, LPhiK2, LPhiReconstruction, QPhiB0, QAdvection, QF2, QB2, QK2,
    QReconstruction, QPhiPhiDelta, QPhi2, SG, SL, SQ, SConstant,
    GF2, GB2, GK2, TreeIrTermCount
};
using TreeIrTerms = std::array<double, TreeIrTermCount>;

// 新 post 入口复用原子项；保持旧 24 项 ABI 不变，另外附加三个 adaptive 源项。
constexpr std::size_t PostTreeTermCount = TreeIrTermCount + 3;
using PostTreeTerms = std::array<double, PostTreeTermCount>;
struct PostTreeSourceExpansion {
    PostTreeTerms fixed_order{}, no_wiggle{};
    std::array<PostTreeTerms, 3> wiggle{};
};
PostTreeSourceExpansion evaluate_post_tree_source_expansion(
    const ClosedTriangleVectors&, const std::array<double, 3>& power,
    const std::array<double, 3>& no_wiggle, const std::array<double, 3>& transfer,
    const HaloBiasV1Params&, const NativeConfig&, double bphi_recon, double eta);

TreeIrTerms evaluate_halo_tree_ir_legs(
    const ClosedTriangleVectors& vectors,
    const std::array<double, 3>& power,
    const std::array<double, 3>& no_wiggle_power,
    const std::array<double, 3>& transfer,
    const HaloBiasV1Params& bias,
    const NativeConfig& config,
    TreeIrScheme scheme = TreeIrScheme::StrictLo,
    bool post = false);

ComponentResult tree_ir_component_result(const ClosedTriangleVectors&, const TreeIrTerms&);
ComponentResult compute_pre_recon_halo_bias_v1_local_png_tree_ir(
    const PowerSpectrum&, const PowerSpectrum& no_wiggle, const PowerSpectrum& transfer,
    const Triangle&, const NativeConfig&, const HaloBiasV1Params&,
    TreeIrScheme = TreeIrScheme::StrictLo);
ComponentResult compute_post_recon_halo_bias_v1_local_png_tree_ir_vectors(
    const PowerSpectrum&, const PowerSpectrum& no_wiggle, const PowerSpectrum& transfer,
    const ClosedTriangleVectors&, const NativeConfig&, const HaloBiasV1Params&,
    TreeIrScheme = TreeIrScheme::StrictLo);
}

// 批量 Python binding：逐实际三角形计算后再投影；错误通过返回值传递，不跨 ABI 抛异常。
extern "C" int marisa_tree_ir_batch(
    std::size_t count, const double* vectors, const double* power,
    const double* no_wiggle, const double* transfer, const double* biases,
    double sigma2, int scheme, int post, double radius, double brec,
    double cell_size, int cic_power, double* output);
extern "C" const char* marisa_tree_ir_last_error();
#endif
