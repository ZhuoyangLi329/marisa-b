#include "post_recon_tree_png_ir.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>

namespace marisa_b_native {
namespace {
constexpr double pi=3.1415926535897932384626433832795;
double dot(const std::array<double,3>& a,const std::array<double,3>& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
bool positive(double x) { return std::isfinite(x) && x>0.; }
std::array<double,3> components(const PostTreeTerms& x) {
    std::array<double,3> out{{x[GPre]+x[GReconstruction],x[24],x[25]+x[26]}};
    for (int i=LPrimordial;i<=LPhiReconstruction;++i) out[1]+=x[i];
    for (int i=QPhiB0;i<=QPhi2;++i) out[2]+=x[i];
    return out;
}
struct BoundedGrid { std::vector<double> q,p,m,noise,weights; };
double interpolate(const std::vector<double>& q,const std::vector<double>& y,double x) {
    if (x<=q.front()) return y.front();
    if (x>=q.back()) return y.back();
    auto it=std::upper_bound(q.begin(),q.end(),x);
    const std::size_t j=it-q.begin(),i=j-1;
    return y[i]+(x-q[i])/(q[j]-q[i])*(y[j]-y[i]);
}
BoundedGrid prepare(const PostReconIrGrid& g,const PostReconIrSettings& s) {
    const std::size_t n=g.q.size();
    if (n<3 || g.power.size()!=n || g.no_wiggle.size()!=n || g.transfer.size()!=n ||
        (!g.shift_noise.empty() && g.shift_noise.size()!=n))
        throw std::invalid_argument("invalid IR grid shape");
    if (!(positive(s.r_bao) && positive(s.k_ir) && positive(s.q_min) && s.q_min<s.k_ir))
        throw std::invalid_argument("positive q_min < k_ir and r_bao required");
    if (s.mesh_damping<0 || s.mesh_damping>2 || s.angular_mu_order<4 || s.angular_phi_order<4)
        throw std::invalid_argument("invalid post IR quadrature");
    std::vector<double> mnw(n),noise(n,0.);
    for (std::size_t j=0;j<n;++j) {
        if (!(positive(g.q[j]) && positive(g.power[j]) && positive(g.no_wiggle[j]) &&
              positive(g.transfer[j]) && (j==0 || g.q[j]>g.q[j-1])))
            throw std::invalid_argument("invalid IR grid values");
        // 必须先在原网格构造 M_nw，再插入积分端点，保持与 oracle 相同的插值语义。
        mnw[j]=g.transfer[j]*std::sqrt(g.no_wiggle[j]/g.power[j]);
        if (!g.shift_noise.empty()) noise[j]=g.shift_noise[j];
        if (!(std::isfinite(noise[j]) && noise[j]>=0.))
            throw std::invalid_argument("shift noise must be finite and nonnegative");
    }
    if (g.q.front()>s.q_min || g.q.back()<s.k_ir)
        throw std::invalid_argument("IR grid must bracket q_min and k_ir");
    BoundedGrid b;b.q.push_back(s.q_min);
    for (double q:g.q) if (q>s.q_min && q<s.k_ir) b.q.push_back(q);
    b.q.push_back(s.k_ir);
    const std::size_t nb=b.q.size();
    if (nb<3) throw std::invalid_argument("need at least three bounded IR nodes");
    for (double q:b.q) {
        b.p.push_back(interpolate(g.q,g.no_wiggle,q));
        b.m.push_back(interpolate(g.q,mnw,q));
        b.noise.push_back(interpolate(g.q,noise,q));
    }
    // 非均匀 Simpson；偶数节点的最后一段使用 SciPy 的 Cartwright 校正。
    b.weights.assign(nb,0.);
    for (std::size_t i=0;i+2<nb;i+=2) {
        const double h0=b.q[i+1]-b.q[i],h1=b.q[i+2]-b.q[i+1],h=h0+h1;
        b.weights[i]+=h/6.*(2.-h1/h0);
        b.weights[i+1]+=h*h*h/(6.*h0*h1);
        b.weights[i+2]+=h/6.*(2.-h0/h1);
    }
    if (nb%2==0) {
        const double h0=b.q[nb-2]-b.q[nb-3],h1=b.q[nb-1]-b.q[nb-2];
        b.weights[nb-1]+=(2.*h1*h1+3.*h0*h1)/(6.*(h0+h1));
        b.weights[nb-2]+=(h1*h1+3.*h0*h1)/(6.*h0);
        b.weights[nb-3]-=h1*h1*h1/(6.*h0*(h0+h1));
    }
    return b;
}
std::array<std::array<double,3>,3> directions(const ClosedTriangleVectors& v) {
    auto dir=v;std::array<double,3> sum{};double scale=0.;
    for (int i=0;i<3;++i) {
        for (double x:v[i]) if (!std::isfinite(x)) throw std::invalid_argument("non-finite triangle");
        const double k=std::sqrt(dot(v[i],v[i]));
        if (!positive(k)) throw std::invalid_argument("zero triangle leg");
        scale=std::max(scale,k);
        for (int a=0;a<3;++a) { dir[i][a]/=k;sum[a]+=v[i][a]; }
    }
    if (std::sqrt(dot(sum,sum))>2.e-12*scale) throw std::invalid_argument("triangle is not closed");
    return dir;
}
std::array<double,3> density(double W,double M,double P,double noise,
    const HaloBiasV1Params& b,const NativeConfig& c,double bpr,double eta) {
    const double a=b.b1/c.bias_recon,n=b.bphi/(c.bias_recon*M),d=bpr/(c.bias_recon*M);
    const double r0=1.-W*a,r1=-W*(n-eta*a*d),r2=W*(eta*n*d-eta*eta*a*d*d);
    const double e0=W*W/(c.bias_recon*c.bias_recon);
    return {{P*r0*r0+noise*e0,
        2.*P*r0*r1-2.*noise*eta*d*e0,
        P*(r1*r1+2.*r0*r2)+3.*noise*eta*eta*d*d*e0}};
}
PostLegCoefficients variance(const ClosedTriangleVectors& v,const BoundedGrid& g,
    const HaloBiasV1Params& b,const NativeConfig& c,const PostReconIrSettings& s,double bpr,double eta) {
    if (!(std::isfinite(b.b1) && std::isfinite(b.bphi) && std::isfinite(bpr) && std::isfinite(eta) &&
          std::isfinite(c.bias_recon) && c.bias_recon!=0. &&
          std::isfinite(c.smoothing_radius) && c.smoothing_radius>=0. &&
          std::isfinite(c.recon_cellsize) && c.recon_cellsize>=0. && c.recon_cic_window_power>=0))
        throw std::invalid_argument("invalid displacement parameters");
    const auto dir=directions(v);
    const bool cic=c.recon_cellsize>0 && c.recon_cic_window_power>0;
    if (cic && s.mesh_damping==1) throw std::invalid_argument("cannot discard active Cartesian CIC");
    const bool cartesian=s.mesh_damping==2 || (s.mesh_damping==0 && cic);
    PostLegCoefficients result{};
    std::vector<std::array<double,3>> qhat;
    std::vector<double> angular_weights;
    if (cartesian) {
        std::unique_ptr<gsl_integration_glfixed_table,decltype(&gsl_integration_glfixed_table_free)>
            table(gsl_integration_glfixed_table_alloc(s.angular_mu_order),gsl_integration_glfixed_table_free);
        if (!table) throw std::bad_alloc();
        for (int j=0;j<s.angular_mu_order;++j) {
            double mu,w;gsl_integration_glfixed_point(-1.,1.,j,&mu,&w,table.get());
            for (int a=0;a<s.angular_phi_order;++a) {
                const double phi=2.*pi*a/s.angular_phi_order,t=std::sqrt(1.-mu*mu);
                qhat.push_back({{t*std::cos(phi),t*std::sin(phi),mu}});
                angular_weights.push_back(w*2.*pi/s.angular_phi_order);
            }
        }
    }
    for (std::size_t j=0;j<g.q.size();++j) {
        const double q=g.q[j],x=q*s.r_bao,W0=std::exp(-0.5*q*q*c.smoothing_radius*c.smoothing_radius);
        if (!cartesian) {
            const auto f=density(W0,g.m[j],g.p[j],g.noise[j],b,c,bpr,eta);
            const double window=1.-gsl_sf_bessel_j0(x)+2.*gsl_sf_bessel_j2(x);
            for (int m=0;m<3;++m) for (int leg=0;leg<3;++leg)
                result[m][leg]+=g.weights[j]*window/(6.*pi*pi)*f[m];
        } else {
            for (std::size_t a=0;a<qhat.size();++a) {
                double W=W0;
                if (cic) {
                    double prod=1.;
                    for (double n:qhat[a]) {
                        const double z=0.5*q*c.recon_cellsize*n;
                        prod*=std::fabs(z)<1.e-12 ? 1. : std::sin(z)/z;
                    }
                    W*=std::pow(prod,c.recon_cic_window_power);
                }
                const auto f=density(W,g.m[j],g.p[j],g.noise[j],b,c,bpr,eta);
                for (int leg=0;leg<3;++leg) {
                    const double mu=dot(dir[leg],qhat[a]);
                    const double weight=g.weights[j]*angular_weights[a]*mu*mu*(1.-std::cos(x*mu))/(8.*pi*pi*pi);
                    for (int m=0;m<3;++m) result[m][leg]+=weight*f[m];
                }
            }
        }
    }
    for (int leg=0;leg<3;++leg) {
        if (result[0][leg]<-1.e-12) throw std::runtime_error("negative variance: refine radial grid");
        result[0][leg]=std::max(0.,result[0][leg]);
        for (int m=0;m<3;++m) if (!std::isfinite(result[m][leg])) throw std::runtime_error("non-finite variance");
    }
    return result;
}
}
PostLegCoefficients post_recon_displacement_variance(const ClosedTriangleVectors& v,
    const PostReconIrGrid& g,const HaloBiasV1Params& b,const NativeConfig& c,
    const PostReconIrSettings& s,double bpr,double eta) { return variance(v,prepare(g,s),b,c,s,bpr,eta); }

PostReconTreeIrResult dress_post_tree_source(const ClosedTriangleVectors& v,
    const PostTreeSourceExpansion& x,const PostLegCoefficients& sig) {
    directions(v);
    PostReconTreeIrResult r;r.source=x.no_wiggle;r.no_wiggle=x.no_wiggle;
    r.fixed_order=x.fixed_order;r.variance=sig;
    for (int i=0;i<3;++i) {
        for (int m=0;m<3;++m) if (!std::isfinite(sig[m][i])) throw std::invalid_argument("non-finite variance");
        if (sig[0][i]<0.) throw std::invalid_argument("negative variance");
        const double k2=dot(v[i],v[i]),D=std::exp(-k2*sig[0][i]);
        const double d1=-k2*sig[1][i],d2=0.5*k2*k2*sig[1][i]*sig[1][i]-k2*sig[2][i];
        r.damping[0][i]=D;r.damping[1][i]=d1;r.damping[2][i]=d2;
        for (std::size_t j=0;j<PostTreeTermCount;++j) r.source[j]+=D*x.wiggle[i][j];
        const auto w=components(x.wiggle[i]);
        r.mixing[0]+=D*d1*w[0];r.mixing[1]+=D*d1*w[1];r.mixing[2]+=D*d2*w[0];
        r.stochastic_mixing[0]+=D*d1*x.wiggle[i][SG];
        r.stochastic_mixing[1]+=D*d1*x.wiggle[i][SL];
        r.stochastic_mixing[2]+=D*d2*x.wiggle[i][SG];
    }
    r.deterministic=components(r.source);
    r.deterministic[1]+=r.mixing[0];r.deterministic[2]+=r.mixing[1]+r.mixing[2];
    return r;
}
PostReconTreeIrResult compute_post_recon_tree_png_ir(const ClosedTriangleVectors& v,
    const std::array<double,3>& p,const std::array<double,3>& nw,const std::array<double,3>& m,
    const HaloBiasV1Params& b,const NativeConfig& c,const PostReconIrGrid& g,
    const PostReconIrSettings& s,double bpr,double eta) {
    return dress_post_tree_source(v,evaluate_post_tree_source_expansion(v,p,nw,m,b,c,bpr,eta),
        post_recon_displacement_variance(v,g,b,c,s,bpr,eta));
}
}
namespace { thread_local std::string post_error; }
extern "C" const char* marisa_post_tree_ir_last_error() { return post_error.c_str(); }
extern "C" int marisa_post_tree_ir_batch(std::size_t n,const double* v,const double* p,const double* nw,
    const double* m,const double* bias,std::size_t nq,const double* q,const double* qp,const double* qnw,
    const double* qm,const double* noise,const double* cfg,const int* quad,double* out) {
    try {
        using namespace marisa_b_native;
        if (!v || !p || !nw || !m || !bias || !q || !qp || !qnw || !qm || !cfg || !quad || !out)
            throw std::invalid_argument("null post batch pointer");
        HaloBiasV1Params b{bias[0],bias[1],bias[2],bias[3],bias[4],bias[5]};
        NativeConfig c;c.smoothing_radius=cfg[0];c.bias_recon=cfg[1];c.recon_cellsize=cfg[2];c.recon_cic_window_power=quad[0];
        PostReconIrSettings s{cfg[3],cfg[4],cfg[5],quad[1],quad[2],quad[3]};
        PostReconIrGrid grid;grid.q.assign(q,q+nq);grid.power.assign(qp,qp+nq);
        grid.no_wiggle.assign(qnw,qnw+nq);grid.transfer.assign(qm,qm+nq);
        if (noise) grid.shift_noise.assign(noise,noise+nq);
        const auto bounded=prepare(grid,s);
        for (std::size_t row=0;row<n;++row) {
            ClosedTriangleVectors vec{};std::array<double,3> pp{},nn{},mm{};
            for (int i=0;i<3;++i) {
                pp[i]=p[3*row+i];nn[i]=nw[3*row+i];mm[i]=m[3*row+i];
                for (int a=0;a<3;++a) vec[i][a]=v[9*row+3*i+a];
            }
            const auto source=evaluate_post_tree_source_expansion(vec,pp,nn,mm,b,c,cfg[6],cfg[7]);
            const auto r=dress_post_tree_source(vec,source,variance(vec,bounded,b,c,s,cfg[6],cfg[7]));
            double* ptr=out+108*row;
            for (const auto* values:{&r.source,&r.no_wiggle,&r.fixed_order})
                ptr=std::copy(values->begin(),values->end(),ptr);
            for (const auto* values:{&r.variance,&r.damping})
                for (const auto& order:*values) ptr=std::copy(order.begin(),order.end(),ptr);
            for (const auto* values:{&r.mixing,&r.stochastic_mixing,&r.deterministic})
                ptr=std::copy(values->begin(),values->end(),ptr);
            for (const double* x=out+108*row;x<ptr;++x)
                if (!std::isfinite(*x)) throw std::runtime_error("non-finite post output");
        }
        post_error.clear();return 0;
    } catch (const std::exception& e) { post_error=e.what();return 1; }
}
