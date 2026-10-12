// 模块大纲：复用已验证的 native source jets 和方差；仅增加独立批量读出 ABI。
#include "tree_png_ir.h"
#include "post_recon_tree_png_ir.h"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace { thread_local std::string bridge_error; }
extern "C" const char* paware_ir_error() { return bridge_error.c_str(); }
extern "C" int paware_source_batch(std::size_t n, const double* v,
    const double* p,const double* nw,const double* m,const double* bias,
    const double* cfg,double* out) {
    try {
        using namespace marisa_b_native;
        HaloBiasV1Params b{bias[0],bias[1],bias[2],bias[3],bias[4],bias[5]};
        NativeConfig c; c.smoothing_radius=cfg[0];c.bias_recon=cfg[1];
        c.recon_cellsize=cfg[2];c.recon_cic_window_power=int(cfg[3]);
        for(std::size_t j=0;j<n;++j) {
            ClosedTriangleVectors vectors{}; std::array<double,3> pp{},nn{},mm{};
            for(int i=0;i<3;++i) {
                pp[i]=p[3*j+i];nn[i]=nw[3*j+i];mm[i]=m[3*j+i];
                for(int a=0;a<3;++a) vectors[i][a]=v[9*j+3*i+a];
            }
            const auto x=evaluate_post_tree_source_expansion(vectors,pp,nn,mm,b,c,cfg[4],cfg[5]);
            double* target=out+135*j;
            target=std::copy(x.fixed_order.begin(),x.fixed_order.end(),target);
            target=std::copy(x.no_wiggle.begin(),x.no_wiggle.end(),target);
            for(const auto& leg:x.wiggle) target=std::copy(leg.begin(),leg.end(),target);
        }
        bridge_error.clear();return 0;
    } catch(const std::exception& e) {bridge_error=e.what();return 1;}
}
