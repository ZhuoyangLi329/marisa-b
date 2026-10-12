#ifndef MARISA_B_TREE_PNG_RSD_H
#define MARISA_B_TREE_PNG_RSD_H
// 模块大纲：复用 MARISA-B bias-v1 合同；有向 pre/post 核；独立 RSD 批量 ABI。
#include "marisa_b_native.h"
#include "finite_png_poly2.h"
#include <cstddef>
namespace marisa_b_native {
using RsdVector=std::array<double,3>;
enum class RsdReconstructionMethod {same_shift_no_linear_rsd_removal=0,kaiser_divided_same_shift=1,rec_iso=2,rec_sym=3};
struct RsdTreeConfig {
    double f_growth=0.; RsdVector los{{0.,0.,1.}};
    int selection_model=0,velocity_bias_model=0;
};
struct RsdReconstructionConfig {
    RsdReconstructionMethod method=RsdReconstructionMethod::same_shift_no_linear_rsd_removal;
    NativeConfig source{};
    double f_nl_recon_ratio=0.,bphi_recon=0.;
    // 默认保留旧 fNL_rec=eta*fNL 路径；固定 catalog 必须显式选 fixed。
    bool fixed_reconstruction_parameter=false;
    double f_nl_recon_fixed=0.,denominator_floor=0.;
    // 可选冻结的 W/D；允许实际 alpha_rec 表与物理 1/M 不同。
    bool tabulated_shift_response=false;
    std::array<double,3> shift_response{};
};
struct RsdFinitePngTreeResult {
    FnlPoly2 total{},gravitational_and_bias{},primordial{};
    std::array<FnlPoly2,3> cyclic_pairs{},z1{},z2{};
    // 不含 nbar 和 nuisance；顺序 Bshot,Pshot,Ashot。
    std::array<double,3> gaussian_stochastic{};
};
RsdFinitePngTreeResult compute_pre_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const ClosedTriangleVectors&,const std::array<double,3>&,const std::array<double,3>&,
    const HaloBiasV1Params&,const RsdTreeConfig&);
RsdFinitePngTreeResult compute_post_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const ClosedTriangleVectors&,const std::array<double,3>&,const std::array<double,3>&,
    const HaloBiasV1Params&,const RsdTreeConfig&,const RsdReconstructionConfig&);
}
// 每行 39 列：total3,gravity3,primordial3,pairs9,Z1_9,Z2_9,stochastic3。
extern "C" int marisa_tree_rsd_batch(std::size_t,const double*,const double*,const double*,
    const double*,const double*,const double*,int,int,double*);
// 独立 ABI：recon[5] 必须为零；固定 fNL_rec 与拟合 fNL 不联动。
extern "C" int marisa_tree_rsd_batch_fixed(std::size_t,const double*,const double*,const double*,
    const double*,const double*,const double*,int,double,double,double*);
extern "C" int marisa_tree_rsd_batch_response(std::size_t,const double*,const double*,const double*,
    const double*,const double*,const double*,double*);
extern "C" const char* marisa_tree_rsd_last_error();
#endif
