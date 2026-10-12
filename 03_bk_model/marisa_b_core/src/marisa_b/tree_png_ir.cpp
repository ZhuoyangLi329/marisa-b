#include "tree_png_ir.h"
#include "PowerSpectrum.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace marisa_b_native {
namespace {
// 一阶自动微分：tangent 同时注入各腿 D_i P_w,i，线性性保证等于三次逐腿导数之和。
struct Dual1 {
    double value, tangent;
    Dual1(double v = 0., double t = 0.) : value(v), tangent(t) {}
    Dual1& operator+=(const Dual1& b) { value += b.value; tangent += b.tangent; return *this; }
};
Dual1 operator+(Dual1 a, Dual1 b) { return {a.value+b.value,a.tangent+b.tangent}; }
Dual1 operator*(Dual1 a, Dual1 b) { return {a.value*b.value,a.tangent*b.value+a.value*b.tangent}; }
Dual1 operator/(Dual1 a, Dual1 b) {
    return {a.value/b.value,(a.tangent-(a.value/b.value)*b.tangent)/b.value};
}
double dot(const std::array<double,3>& a, const std::array<double,3>& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
double shift(const std::array<double,3>& total, const std::array<double,3>& q,
             const NativeConfig& c) {
    const double q2=dot(q,q);
    double window=std::exp(-0.5*q2*c.smoothing_radius*c.smoothing_radius);
    if (c.recon_cellsize>0 && c.recon_cic_window_power>0) {
        double sinc=1.;
        for (double x:q) {
            const double a=0.5*x*c.recon_cellsize;
            sinc*=std::fabs(a)<1.e-12 ? 1. : std::sin(a)/a;
        }
        window*=std::pow(sinc,c.recon_cic_window_power);
    }
    return -dot(total,q)*window/(q2*c.bias_recon);
}

template<class S> std::array<S,TreeIrTermCount> atomic(
    const ClosedTriangleVectors& v, const std::array<double,3>& k,
    const std::array<S,3>& p, const std::array<S,3>& m,
    const HaloBiasV1Params& b, const NativeConfig& c, bool post) {
    std::array<S,TreeIrTermCount> x{};
    std::array<S,3> u{{S(1.)/m[0],S(1.)/m[1],S(1.)/m[2]}};
    const double b1=b.b1, bp=b.bphi;
    S b0{};
    x[SConstant]=S(1.);
    for (int i=0;i<3;++i) {
        x[SG]+=b1*b1*p[i];
        x[SL]+=b1*bp*p[i]*u[i];
        x[SQ]+=bp*bp*p[i]*u[i]*u[i];
        const int j=(i+1)%3, ell=(i+2)%3;
        const double mu=dot(v[i],v[j])/(k[i]*k[j]);
        const double f2=5./7.+0.5*mu*(k[i]/k[j]+k[j]/k[i])+2./7.*mu*mu;
        const double s2=mu*mu-1./3.;
        const S pp=p[i]*p[j], us=u[i]+u[j], up=u[i]*u[j];
        const S adv=mu*(k[i]/k[j]*u[i]+k[j]/k[i]*u[j]);
        x[GF2]+=2.*b1*b1*b1*f2*pp;
        x[GB2]+=b1*b1*b.b2*pp;
        x[GK2]+=2.*b1*b1*b.bK2*s2*pp;
        x[LPhiF2]+=2.*b1*b1*bp*f2*pp*us;
        x[LPhiAdvection]+=b1*b1*bp*pp*adv;
        x[LPhiDelta]+=b1*b1*b.bphidelta*pp*us;
        x[LPhiB2]+=b1*b.b2*bp*pp*us;
        x[LPhiK2]+=2.*b1*b.bK2*bp*s2*pp*us;
        x[QAdvection]+=b1*bp*bp*pp*us*adv;
        x[QF2]+=2.*b1*bp*bp*f2*pp*up;
        x[QB2]+=bp*bp*b.b2*pp*up;
        x[QK2]+=2.*bp*bp*b.bK2*s2*pp*up;
        x[QPhiPhiDelta]+=b1*bp*b.bphidelta*pp*us*us;
        x[QPhi2]+=b1*b1*b.bphi2*pp*up;
        if (post) {
            std::array<double,3> total{};
            for (int a=0;a<3;++a) total[a]=v[i][a]+v[j][a];
            const double s=shift(total,v[i],c)+shift(total,v[j],c);
            x[GReconstruction]+=b1*b1*b1*b1*s*pp;
            x[LPhiReconstruction]+=2.*b1*b1*b1*bp*s*pp*us;
            x[QReconstruction]+=b1*b1*bp*bp*s*pp*(u[i]*u[i]+4.*up+u[j]*u[j]);
        }
        b0+=2.*pp*m[ell]*up;
    }
    x[GPre]=x[GF2]+x[GB2]+x[GK2];
    x[LPrimordial]=b1*b1*b1*b0;
    x[QPhiB0]=b1*b1*bp*(u[0]+u[1]+u[2])*b0;
    return x;
}

template<class S> std::array<S,PostTreeTermCount> post_atomic(
    const ClosedTriangleVectors& v, const std::array<double,3>& k,
    const std::array<S,3>& p, const std::array<S,3>& m,
    const HaloBiasV1Params& b, const NativeConfig& c, double bpr, double eta) {
    std::array<S,PostTreeTermCount> x{};
    const auto base=atomic(v,k,p,m,b,c,true);
    std::copy(base.begin(),base.end(),x.begin());
    const double ratio=bpr/c.bias_recon;
    for (int i=0;i<3;++i) {
        const int j=(i+1)%3;
        std::array<double,3> total{};
        for (int a=0;a<3;++a) total[a]=v[i][a]+v[j][a];
        const double si=shift(total,v[i],c), sj=shift(total,v[j],c);
        const S ui=S(1.)/m[i], uj=S(1.)/m[j], pp=p[i]*p[j];
        // f_rec=eta*f；二阶项已经是 Taylor 系数，不能再除以 2。
        x[24]+=-eta*std::pow(b.b1,4)*ratio*pp*(si*ui+sj*uj);
        x[25]+=eta*eta*std::pow(b.b1,4)*ratio*ratio*pp*(si*ui*ui+sj*uj*uj);
        x[26]+=-2.*eta*std::pow(b.b1,3)*b.bphi*ratio*pp*(ui+uj)*(si*ui+sj*uj);
    }
    return x;
}

std::array<double,3> validate(const ClosedTriangleVectors& v) {
    std::array<double,3> k{},sum{};
    for (int i=0;i<3;++i) {
        for (int a=0;a<3;++a) {
            if (!std::isfinite(v[i][a])) throw std::invalid_argument("non-finite triangle");
            sum[a]+=v[i][a];
        }
        k[i]=std::sqrt(dot(v[i],v[i]));
        if (!(k[i]>0)) throw std::invalid_argument("zero triangle leg");
    }
    if (std::sqrt(dot(sum,sum))>2.e-12*(*std::max_element(k.begin(),k.end())))
        throw std::invalid_argument("triangle is not closed");
    return k;
}
ClosedTriangleVectors vectors(const Triangle& t) {
    if (!(t.k1>0 && t.k2>0 && std::fabs(t.mu12)<=1))
        throw std::invalid_argument("invalid triangle");
    const double x=t.k2*std::sqrt(std::max(0.,1.-t.mu12*t.mu12));
    return {{{{0.,0.,t.k1}},{{x,0.,t.k2*t.mu12}},{{-x,0.,-t.k1-t.k2*t.mu12}}}};
}
ComponentResult spectra(const PowerSpectrum& p, const PowerSpectrum& nw,
                        const PowerSpectrum& m, const ClosedTriangleVectors& v,
                        const NativeConfig& c, const HaloBiasV1Params& b,
                        TreeIrScheme s, bool post) {
    const auto k=validate(v);
    std::array<double,3> pv{},nv{},mv{};
    for (int i=0;i<3;++i) { pv[i]=p(k[i]);nv[i]=nw(k[i]);mv[i]=m(k[i]); }
    return tree_ir_component_result(v,evaluate_halo_tree_ir_legs(v,pv,nv,mv,b,c,s,post));
}
}

PostTreeSourceExpansion evaluate_post_tree_source_expansion(
    const ClosedTriangleVectors& v,const std::array<double,3>& p,
    const std::array<double,3>& nw,const std::array<double,3>& m,
    const HaloBiasV1Params& b,const NativeConfig& c,double bpr,double eta) {
    // 借用固定阶入口的输入检查，但不走只允许各向同性的旧标量 IR 分支。
    evaluate_halo_tree_ir_legs(v,p,nw,m,b,c,TreeIrScheme::FixedOrder,true);
    if (!(std::isfinite(bpr) && std::isfinite(eta) &&
          std::isfinite(c.recon_cellsize) && c.recon_cellsize>=0 &&
          c.recon_cic_window_power>=0)) throw std::invalid_argument("invalid post IR configuration");
    const auto k=validate(v);
    std::array<double,3> mnw{};
    for (int i=0;i<3;++i) mnw[i]=m[i]*std::sqrt(nw[i]/p[i]);
    PostTreeSourceExpansion result;
    result.fixed_order=post_atomic(v,k,p,m,b,c,bpr,eta);
    result.no_wiggle=post_atomic(v,k,nw,mnw,b,c,bpr,eta);
    for (int leg=0;leg<3;++leg) {
        std::array<Dual1,3> pd{},md{};
        for (int i=0;i<3;++i) {
            const double pw=i==leg ? p[i]-nw[i] : 0.;
            pd[i]={nw[i],pw};md[i]={mnw[i],0.5*mnw[i]*pw/nw[i]};
        }
        const auto dual=post_atomic(v,k,pd,md,b,c,bpr,eta);
        for (std::size_t j=0;j<PostTreeTermCount;++j) result.wiggle[leg][j]=dual[j].tangent;
    }
    return result;
}

TreeIrTerms evaluate_halo_tree_ir_legs(
    const ClosedTriangleVectors& v, const std::array<double,3>& p,
    const std::array<double,3>& nw, const std::array<double,3>& m,
    const HaloBiasV1Params& b, const NativeConfig& c, TreeIrScheme scheme, bool post) {
    const auto k=validate(v);
    for (double bias:{b.b1,b.b2,b.bK2,b.bphi,b.bphidelta,b.bphi2})
        if (!std::isfinite(bias)) throw std::invalid_argument("non-finite halo bias");
    if (!(std::isfinite(c.ir_sigma2) && c.ir_sigma2>=0))
        throw std::invalid_argument("invalid tree IR Sigma^2");
    if (post && !(std::isfinite(c.bias_recon) && c.bias_recon!=0 &&
                  std::isfinite(c.smoothing_radius) && c.smoothing_radius>=0))
        throw std::invalid_argument("invalid reconstruction parameters");
    if (post && scheme!=TreeIrScheme::FixedOrder &&
        c.recon_cellsize>0 && c.recon_cic_window_power>0)
        throw std::invalid_argument("scalar tree IR does not support Cartesian CIC covariance");
    for (int i=0;i<3;++i)
        if (!(p[i]>0 && nw[i]>0 && m[i]>0 && std::isfinite(p[i]) &&
              std::isfinite(nw[i]) && std::isfinite(m[i])))
            throw std::invalid_argument("invalid leg spectrum");
    if (scheme==TreeIrScheme::FixedOrder) return atomic(v,k,p,m,b,c,post);
    std::array<double,3> pir{},mir{};
    std::array<Dual1,3> pd{},md{};
    for (int i=0;i<3;++i) {
        const double mnw=m[i]*std::sqrt(nw[i]/p[i]);
        const double dw=std::exp(-k[i]*k[i]*c.ir_sigma2)*(p[i]-nw[i]);
        pd[i]={nw[i],dw};md[i]={mnw,0.5*mnw*dw/nw[i]};
        pir[i]=nw[i]+dw;mir[i]=m[i]*std::sqrt(pir[i]/p[i]);
    }
    if (scheme==TreeIrScheme::FactorizedClosure) return atomic(v,k,pir,mir,b,c,post);
    if (scheme!=TreeIrScheme::StrictLo) throw std::invalid_argument("unknown tree IR scheme");
    const auto dual=atomic(v,k,pd,md,b,c,post);
    TreeIrTerms out{};
    for (std::size_t j=0;j<out.size();++j) out[j]=dual[j].value+dual[j].tangent;
    return out;
}

ComponentResult tree_ir_component_result(const ClosedTriangleVectors& v,const TreeIrTerms& x) {
    const auto k=validate(v);
    ComponentResult r;
    r.triangle={k[0],k[1],dot(v[0],v[1])/(k[0]*k[1])};r.k3=k[2];
    r.Btree=x[GPre]+x[GReconstruction];r.Btotal=r.Btree;
    r.dBdfNL_local_primordial=x[LPrimordial];r.dBdfNL_local_bphi_f2=x[LPhiF2];
    r.dBdfNL_local_bphi_advection=x[LPhiAdvection];r.dBdfNL_local_bphidelta=x[LPhiDelta];
    r.dBdfNL_local_bphi_b2=x[LPhiB2];r.dBdfNL_local_bphi_bK2=x[LPhiK2];
    r.dBdfNL_local_bphi_reconstruction=x[LPhiReconstruction];
    for (std::size_t j=LPrimordial;j<=LPhiReconstruction;++j) r.dBdfNL_local_tree+=x[j];
    r.dBdfNL_local_total=r.dBdfNL_local_tree;
    r.Bhalo_tree_fNL2_bphi_B0=x[QPhiB0];r.Bhalo_tree_fNL2_bphi_sq_advection=x[QAdvection];
    r.Bhalo_tree_fNL2_bphi_sq_F2=x[QF2];r.Bhalo_tree_fNL2_bphi_sq_b2=x[QB2];
    r.Bhalo_tree_fNL2_bphi_sq_bK2=x[QK2];r.Bhalo_tree_fNL2_bphi_sq_reconstruction=x[QReconstruction];
    r.Bhalo_tree_fNL2_bphi_bphidelta=x[QPhiPhiDelta];r.Bhalo_tree_fNL2_bphi2_operator=x[QPhi2];
    for (std::size_t j=QPhiB0;j<=QPhi2;++j) r.Bhalo_tree_fNL2_deterministic+=x[j];
    r.stochastic_alpha3_basis=x[SG];r.dBdfNL_stochastic_alpha3_basis=x[SL];
    r.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis=x[SQ];r.stochastic_alpha4_basis=x[SConstant];
    return r;
}
ComponentResult compute_pre_recon_halo_bias_v1_local_png_tree_ir(
    const PowerSpectrum& p,const PowerSpectrum& nw,const PowerSpectrum& m,
    const Triangle& t,const NativeConfig& c,const HaloBiasV1Params& b,TreeIrScheme s) {
    return spectra(p,nw,m,vectors(t),c,b,s,false);
}
ComponentResult compute_post_recon_halo_bias_v1_local_png_tree_ir_vectors(
    const PowerSpectrum& p,const PowerSpectrum& nw,const PowerSpectrum& m,
    const ClosedTriangleVectors& v,const NativeConfig& c,const HaloBiasV1Params& b,TreeIrScheme s) {
    return spectra(p,nw,m,v,c,b,s,true);
}
}

namespace { thread_local std::string last_error; }
extern "C" const char* marisa_tree_ir_last_error() { return last_error.c_str(); }
extern "C" int marisa_tree_ir_batch(
    std::size_t n,const double* v,const double* p,const double* nw,const double* m,
    const double* bias,double sigma2,int scheme,int post,double radius,double brec,
    double cell_size,int cic_power,double* out) {
    try {
        using namespace marisa_b_native;
        if (!v || !p || !nw || !m || !bias || !out) throw std::invalid_argument("null batch pointer");
        HaloBiasV1Params b{bias[0],bias[1],bias[2],bias[3],bias[4],bias[5]};
        NativeConfig c;c.ir_sigma2=sigma2;c.smoothing_radius=radius;c.bias_recon=brec;
        c.recon_cellsize=cell_size;c.recon_cic_window_power=cic_power;
        for (std::size_t row=0;row<n;++row) {
            ClosedTriangleVectors vec{};std::array<double,3> pp{},nn{},mm{};
            for (int i=0;i<3;++i) {
                pp[i]=p[3*row+i];nn[i]=nw[3*row+i];mm[i]=m[3*row+i];
                for (int a=0;a<3;++a) vec[i][a]=v[9*row+3*i+a];
            }
            const auto x=evaluate_halo_tree_ir_legs(vec,pp,nn,mm,b,c,static_cast<TreeIrScheme>(scheme),post!=0);
            std::copy(x.begin(),x.end(),out+row*TreeIrTermCount);
        }
        last_error.clear();return 0;
    } catch (const std::exception& e) { last_error=e.what();return 1; }
}
