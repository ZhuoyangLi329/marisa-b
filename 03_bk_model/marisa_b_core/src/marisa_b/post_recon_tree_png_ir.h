#ifndef MARISA_B_POST_RECON_TREE_PNG_IR_H
#define MARISA_B_POST_RECON_TREE_PNG_IR_H
#include "tree_png_ir.h"
#include <vector>

namespace marisa_b_native {
// 模块大纲：有限盒 IR 截断；各向同性/笛卡尔 CIC 位移；逐腿阻尼；源项与指数混合分离。
struct PostReconIrSettings {
    double r_bao=110., k_ir=0.15, q_min=0.;
    int mesh_damping=0, angular_mu_order=32, angular_phi_order=64;
    // mesh_damping: 0 自动，1 强制各向同性（活动 CIC 时拒绝），2 笛卡尔。
};
struct PostReconIrGrid {
    std::vector<double> q, power, no_wiggle, transfer, shift_noise;
};
using PostLegCoefficients=std::array<std::array<double,3>,3>; // [f 阶数][腿]
struct PostReconTreeIrResult {
    PostTreeTerms source{}, no_wiggle{}, fixed_order{};
    PostLegCoefficients variance{}, damping{}; // damping 行顺序 D0,d1,d2
    std::array<double,3> mixing{}, stochastic_mixing{}, deterministic{};
};
PostLegCoefficients post_recon_displacement_variance(
    const ClosedTriangleVectors&, const PostReconIrGrid&, const HaloBiasV1Params&,
    const NativeConfig&, const PostReconIrSettings&, double bphi_recon, double eta);
PostReconTreeIrResult dress_post_tree_source(
    const ClosedTriangleVectors&, const PostTreeSourceExpansion&, const PostLegCoefficients&);
PostReconTreeIrResult compute_post_recon_tree_png_ir(
    const ClosedTriangleVectors&, const std::array<double,3>& power,
    const std::array<double,3>& no_wiggle, const std::array<double,3>& transfer,
    const HaloBiasV1Params&, const NativeConfig&, const PostReconIrGrid&,
    const PostReconIrSettings&, double bphi_recon, double eta);
}

// 输出每行 108 列：source27,nw27,fixed27,variance9,damping9,mix3,stoch_mix3,G/L/Q3。
extern "C" int marisa_post_tree_ir_batch(
    std::size_t count,const double* vectors,const double* power,const double* no_wiggle,
    const double* transfer,const double* bias,std::size_t nq,const double* q,
    const double* qpower,const double* qnw,const double* qm,const double* noise,
    const double* config,const int* quadrature,double* output);
extern "C" const char* marisa_post_tree_ir_last_error();
#endif
