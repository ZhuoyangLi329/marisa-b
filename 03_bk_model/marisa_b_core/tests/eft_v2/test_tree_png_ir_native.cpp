// 模块大纲：与当前 native 逐字段回归；独立有限差分；IR 极限；非法配置拒绝。
#include "tree_png_ir.h"
#include "PowerSpectrum.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <algorithm>
namespace n=marisa_b_native;
namespace {
class Spectrum:public PowerSpectrum {
public:
    bool transfer;
    explicit Spectrum(bool t):transfer(t){}
    real Evaluate(real k)const override{return transfer?0.35+4*k*k:1200*std::exp(-3*k*k)/(1+2*k);}
    const Cosmology& GetCosmology()const override{throw std::logic_error("no cosmology");}
};
int checks=0;
void close(double a,double b,double tolerance=1.e-12) {
    ++checks;
    if (std::fabs(a-b)>tolerance*std::max({1.,std::fabs(a),std::fabs(b)}))
        throw std::runtime_error("native tree IR mismatch "+std::to_string(a)+" vs "+std::to_string(b));
}
}
int main() {
    try {
        Spectrum p(false),m(true);
        std::mt19937_64 rng(20260930);std::uniform_real_distribution<double> u(0.,1.);
        const double n::ComponentResult::* fields[]={
            &n::ComponentResult::Btree,&n::ComponentResult::dBdfNL_local_tree,
            &n::ComponentResult::dBdfNL_local_primordial,&n::ComponentResult::dBdfNL_local_bphi_f2,
            &n::ComponentResult::dBdfNL_local_bphi_advection,&n::ComponentResult::dBdfNL_local_bphidelta,
            &n::ComponentResult::dBdfNL_local_bphi_b2,&n::ComponentResult::dBdfNL_local_bphi_bK2,
            &n::ComponentResult::dBdfNL_local_bphi_reconstruction,
            &n::ComponentResult::Bhalo_tree_fNL2_bphi_B0,&n::ComponentResult::Bhalo_tree_fNL2_bphi_sq_advection,
            &n::ComponentResult::Bhalo_tree_fNL2_bphi_sq_F2,&n::ComponentResult::Bhalo_tree_fNL2_bphi_sq_b2,
            &n::ComponentResult::Bhalo_tree_fNL2_bphi_sq_bK2,&n::ComponentResult::Bhalo_tree_fNL2_bphi_sq_reconstruction,
            &n::ComponentResult::Bhalo_tree_fNL2_bphi_bphidelta,&n::ComponentResult::Bhalo_tree_fNL2_bphi2_operator,
            &n::ComponentResult::Bhalo_tree_fNL2_deterministic,&n::ComponentResult::stochastic_alpha3_basis,
            &n::ComponentResult::dBdfNL_stochastic_alpha3_basis,
            &n::ComponentResult::Bhalo_tree_fNL2_stochastic_alpha3PNG_basis,
            &n::ComponentResult::stochastic_alpha4_basis};
        for (int trial=0;trial<200;++trial) {
            const double k1=0.01+0.2*u(rng),k2=0.01+0.2*u(rng),mu=-0.99+1.98*u(rng);
            n::Triangle tri{k1,k2,mu};
            const double transverse=k2*std::sqrt(1-mu*mu);
            n::ClosedTriangleVectors v{{{{0,0,k1}},{{transverse,0,k2*mu}},{{-transverse,0,-k1-k2*mu}}}};
            std::array<double,3> pv{},nw{},mv{},norm{};
            for(int i=0;i<3;++i){norm[i]=std::sqrt(v[i][0]*v[i][0]+v[i][2]*v[i][2]);pv[i]=p(norm[i]);mv[i]=m(norm[i]);nw[i]=pv[i]*(1+0.06*std::sin(110*norm[i]));}
            n::HaloBiasV1Params b{1+2*u(rng),-1+2*u(rng),-0.5+u(rng),2+u(rng),-2+u(rng),-1+3*u(rng)};
            n::NativeConfig c;c.ir_sigma2=20;c.smoothing_radius=10;c.bias_recon=1.9;
            for(bool post:{false,true}) {
                const auto fixed=n::evaluate_halo_tree_ir_legs(v,pv,nw,mv,b,c,n::TreeIrScheme::FixedOrder,post);
                const auto r=n::tree_ir_component_result(v,fixed);
                const auto old=post?n::compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(p,m,v,c,b):n::compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(p,m,tri,b);
                for(const auto field:fields) close(r.*field,old.*field);
                auto zero=c;zero.ir_sigma2=0;
                const auto fact=n::evaluate_halo_tree_ir_legs(v,pv,nw,mv,b,zero,n::TreeIrScheme::FactorizedClosure,post);
                for(std::size_t j=0;j<fixed.size();++j) close(fact[j],fixed[j]);
                const auto strict=n::evaluate_halo_tree_ir_legs(v,pv,nw,mv,b,c,n::TreeIrScheme::StrictLo,post);
                close(strict[n::SQ],fixed[n::SQ]);
                std::array<double,3> mnw{},pp{},pm{},mp{},mm{};
                constexpr double h=2.e-4;
                for(int i=0;i<3;++i){
                    mnw[i]=mv[i]*std::sqrt(nw[i]/pv[i]);
                    const double dw=std::exp(-norm[i]*norm[i]*c.ir_sigma2)*(pv[i]-nw[i]);
                    pp[i]=nw[i]+h*dw;pm[i]=nw[i]-h*dw;
                    mp[i]=mnw[i]*std::sqrt(pp[i]/nw[i]);mm[i]=mnw[i]*std::sqrt(pm[i]/nw[i]);
                }
                const auto smooth=n::evaluate_halo_tree_ir_legs(v,nw,nw,mnw,b,c,n::TreeIrScheme::FixedOrder,post);
                const auto plus=n::evaluate_halo_tree_ir_legs(v,pp,pp,mp,b,c,n::TreeIrScheme::FixedOrder,post);
                const auto minus=n::evaluate_halo_tree_ir_legs(v,pm,pm,mm,b,c,n::TreeIrScheme::FixedOrder,post);
                for(std::size_t j=0;j<fixed.size();++j) close(strict[j],smooth[j]+(plus[j]-minus[j])/(2*h),3.e-8);
            }
            // 当前 native adaptive 原始基底交叉核对，包含 CIC 开关与非单位 eta。
            for (bool cic:{false,true}) {
                c.recon_cellsize=cic?8.:0.;c.recon_cic_window_power=cic?4:0;
                const double eta=0.73,bpr=3.14,ratio=bpr/c.bias_recon;
                const auto jets=n::evaluate_post_tree_source_expansion(v,pv,nw,mv,b,c,bpr,eta);
                const auto old=n::compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(p,m,v,c);
                close(jets.fixed_order[24],eta*std::pow(b.b1,4)*ratio*old.gaussian_linear);
                close(jets.fixed_order[25],eta*eta*std::pow(b.b1,4)*ratio*ratio*old.gaussian_quadratic);
                close(jets.fixed_order[26],eta*std::pow(b.b1,3)*b.bphi*ratio*old.png_linear_cross);
                const auto fixed=n::evaluate_halo_tree_ir_legs(v,pv,nw,mv,b,c,n::TreeIrScheme::FixedOrder,true);
                for(std::size_t j=0;j<fixed.size();++j) close(jets.fixed_order[j],fixed[j]);
            }
        }
        std::cout<<"tree PNG IR native checks passed: "<<checks<<'\n';
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
