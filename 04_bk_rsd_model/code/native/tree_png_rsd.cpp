#include "tree_png_rsd.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
namespace marisa_b_native {
namespace {
// 模块大纲：closed triangle/LOS 验证；标准 Z1/Z2；same-shift 二阶 partition；G/L/Q。
double dot(const RsdVector& a,const RsdVector& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(const RsdVector& a) {return std::sqrt(dot(a,a));}
RsdFinitePngTreeResult evaluate(const ClosedTriangleVectors& v,const std::array<double,3>& p,
    const std::array<double,3>& m,const HaloBiasV1Params& b,const RsdTreeConfig& rsd,
    const RsdReconstructionConfig* rec) {
    if(rsd.selection_model!=0||rsd.velocity_bias_model!=0) throw std::invalid_argument("selection/velocity bias not implemented");
    for(double x:{b.b1,b.b2,b.bK2,b.bphi,b.bphidelta,b.bphi2,rsd.f_growth})
        if(!std::isfinite(x)) throw std::invalid_argument("nonfinite bias/growth");
    RsdVector n=rsd.los,sum{};
    for(double x:n) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite LOS");
    const double nn=norm(n);
    if(!(nn>0.)) throw std::invalid_argument("zero LOS");
    for(double& x:n) x/=nn;
    std::array<double,3> k{},mu{},u{};
    for(int i=0;i<3;++i) {
        for(int a=0;a<3;++a) {if(!std::isfinite(v[i][a])) throw std::invalid_argument("nonfinite vector");sum[a]+=v[i][a];}
        k[i]=norm(v[i]);
        if(!(k[i]>0.)||!(p[i]>0.)||!(m[i]>0.)||!std::isfinite(p[i])||!std::isfinite(m[i]))
            throw std::invalid_argument("zero leg or invalid spectrum");
        mu[i]=dot(v[i],n)/k[i];u[i]=1./m[i];
    }
    if(norm(sum)>2.e-11*(*std::max_element(k.begin(),k.end()))) throw std::invalid_argument("triangle does not close");
    if(rec) {
        if(rec->method!=RsdReconstructionMethod::same_shift_no_linear_rsd_removal) throw std::invalid_argument("reconstruction method not implemented");
        const auto& c=rec->source;
        for(double x:{c.smoothing_radius,c.bias_recon,c.recon_cellsize,rec->bphi_recon,rec->f_nl_recon_ratio})
            if(!std::isfinite(x)) throw std::invalid_argument("nonfinite reconstruction");
        if(c.smoothing_radius<0.||c.bias_recon==0.||c.recon_cellsize<0.||c.recon_cic_window_power<0)
            throw std::invalid_argument("invalid reconstruction");
        if(rec->fixed_reconstruction_parameter) {
            if(rec->f_nl_recon_ratio!=0.||!std::isfinite(rec->f_nl_recon_fixed)
                ||!std::isfinite(rec->denominator_floor)||rec->denominator_floor<0.)
                throw std::invalid_argument("invalid fixed reconstruction parameter path");
        }
        if(rec->tabulated_shift_response)
            for(double x:rec->shift_response)
                if(!std::isfinite(x)||x<0.) throw std::invalid_argument("invalid frozen shift response");
    }
    RsdFinitePngTreeResult out;
    out.gaussian_stochastic[2]=1.;
    for(int i=0;i<3;++i) {
        out.z1[i]={b.b1+rsd.f_growth*mu[i]*mu[i],b.bphi*u[i],0.};
        out.gaussian_stochastic[0]+=p[i]*b.b1*out.z1[i].c0;
        out.gaussian_stochastic[1]+=p[i]*rsd.f_growth*mu[i]*mu[i]*out.z1[i].c0;
    }
    double b0=0.;
    for(int i=0;i<3;++i) {
        const int j=(i+1)%3,ell=(i+2)%3;
        RsdVector total{}; for(int a=0;a<3;++a) total[a]=v[i][a]+v[j][a];
        const double kn=norm(total),par=dot(total,n);
        if(!(kn>0.)) throw std::invalid_argument("singular pair");
        const double nu=std::clamp(dot(v[i],v[j])/(k[i]*k[j]),-1.,1.);
        const double common=.5*nu*(k[i]/k[j]+k[j]/k[i]);
        const double f2=5./7.+common+2./7.*nu*nu;
        const double g2=3./7.+common+4./7.*nu*nu;
        const double adv=nu*(k[i]/k[j]*u[i]+k[j]/k[i]*u[j]);
        FnlPoly2 z2{b.b1*f2+.5*b.b2+b.bK2*(nu*nu-1./3.)+rsd.f_growth*par*par/(kn*kn)*g2,
            .5*(b.bphi*adv+b.bphidelta*(u[i]+u[j])),.5*b.bphi2*u[i]*u[j]};
        // nonlinear mapping 必须使用完整 finite-PNG Z1，包含 RSD-bphi 交叉项。
        z2=z2+.5*rsd.f_growth*par*((mu[i]/k[i])*out.z1[j]+(mu[j]/k[j])*out.z1[i]);
        if(rec) {
            const auto& c=rec->source;
            FnlPoly2 shifts{};
            for(int a:{i,j}) {
                if(rec->tabulated_shift_response) {
                    shifts=shifts+FnlPoly2{-dot(total,v[a])*rec->shift_response[a]/(k[a]*k[a]),0.,0.};
                    continue;
                }
                double w=std::exp(-.5*k[a]*k[a]*c.smoothing_radius*c.smoothing_radius);
                if(c.recon_cellsize>0.&&c.recon_cic_window_power>0) {
                    double prod=1.;
                    for(double x:v[a]) {double y=.5*x*c.recon_cellsize;prod*=std::abs(y)<1.e-12?1.:std::sin(y)/y;}
                    w*=std::pow(prod,c.recon_cic_window_power);
                }
                FnlPoly2 denom{c.bias_recon,rec->f_nl_recon_ratio*rec->bphi_recon*u[a],0.};
                if(rec->fixed_reconstruction_parameter) {
                    // catalog 参数冻结：不对 fNL_rec 作 Taylor 展开，不跟随试探 fNL。
                    denom={c.bias_recon+rec->f_nl_recon_fixed*rec->bphi_recon*u[a],0.,0.};
                    if(!std::isfinite(denom.c0)||denom.c0<=rec->denominator_floor)
                        throw std::invalid_argument("fixed reconstruction denominator is nonpositive or below floor");
                }
                shifts=shifts+FnlPoly2{-dot(total,v[a])*w/(k[a]*k[a]),0.,0.}/denom;
            }
            // 当前 partition map 的精确二阶项；不改变旧 real-space API。
            z2=z2+.5*shifts*out.z1[i]*out.z1[j];
        }
        out.z2[i]=z2;
        out.cyclic_pairs[i]=(2.*p[i]*p[j])*out.z1[i]*out.z1[j]*z2;
        out.gravitational_and_bias=out.gravitational_and_bias+out.cyclic_pairs[i];
        b0+=2.*p[i]*p[j]*m[ell]*u[i]*u[j];
    }
    out.primordial=FnlPoly2{0.,b0,0.}*out.z1[0]*out.z1[1]*out.z1[2];
    out.total=out.gravitational_and_bias+out.primordial;
    for(double x:{out.total.c0,out.total.c1,out.total.c2}) if(!std::isfinite(x)) throw std::overflow_error("nonfinite result");
    return out;
}
}
RsdFinitePngTreeResult compute_pre_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const ClosedTriangleVectors& v,const std::array<double,3>& p,const std::array<double,3>& m,
    const HaloBiasV1Params& b,const RsdTreeConfig& r) {return evaluate(v,p,m,b,r,nullptr);}
RsdFinitePngTreeResult compute_post_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const ClosedTriangleVectors& v,const std::array<double,3>& p,const std::array<double,3>& m,
    const HaloBiasV1Params& b,const RsdTreeConfig& r,const RsdReconstructionConfig& c) {return evaluate(v,p,m,b,r,&c);}
}
namespace {thread_local std::string rsd_error;}
extern "C" const char* marisa_tree_rsd_last_error() {return rsd_error.c_str();}
namespace {
int batch_impl(std::size_t count,const double* vectors,const double* power,
    const double* transfer,const double* bias,const double* config,const double* recon,int post,int method,
    bool fixed,double fnl_rec,double denominator_floor,double* output,const double* response=nullptr) {
    try {
        using namespace marisa_b_native;
        if(count==0) throw std::invalid_argument("empty triangle batch");
        if(config[4]!=0.||config[5]!=0.) throw std::invalid_argument("selection/velocity bias not implemented");
        HaloBiasV1Params b{bias[0],bias[1],bias[2],bias[3],bias[4],bias[5]};
        RsdTreeConfig r;r.f_growth=config[0];r.los={config[1],config[2],config[3]};r.selection_model=int(config[4]);r.velocity_bias_model=int(config[5]);
        RsdReconstructionConfig c;c.method=static_cast<RsdReconstructionMethod>(method);
        c.source.smoothing_radius=recon[0];c.source.bias_recon=recon[1];c.source.recon_cellsize=recon[2];
        if(!std::isfinite(recon[3])||recon[3]<0.||recon[3]>std::numeric_limits<int>::max()||recon[3]!=std::floor(recon[3]))
            throw std::invalid_argument("CIC power must be finite nonnegative integer");
        c.source.recon_cic_window_power=int(recon[3]);c.bphi_recon=recon[4];c.f_nl_recon_ratio=recon[5];
        c.fixed_reconstruction_parameter=fixed;c.f_nl_recon_fixed=fnl_rec;c.denominator_floor=denominator_floor;
        c.tabulated_shift_response=response!=nullptr;
        if(post!=0&&post!=1) throw std::invalid_argument("post must be 0/1");
        for(std::size_t row=0;row<count;++row) {
            ClosedTriangleVectors v{};std::array<double,3> p{},m{};
            for(int i=0;i<3;++i) {p[i]=power[3*row+i];m[i]=transfer[3*row+i];for(int a=0;a<3;++a)v[i][a]=vectors[9*row+3*i+a];}
            if(response)for(int i=0;i<3;++i)c.shift_response[i]=response[3*row+i];
            const auto x=post?compute_post_recon_halo_bias_v1_local_png_tree_rsd_vectors(v,p,m,b,r,c):compute_pre_recon_halo_bias_v1_local_png_tree_rsd_vectors(v,p,m,b,r);
            double* dst=output+39*row;
            auto put=[&dst](FnlPoly2 q){*dst++=q.c0;*dst++=q.c1;*dst++=q.c2;};
            put(x.total);put(x.gravitational_and_bias);put(x.primordial);
            for(auto q:x.cyclic_pairs)put(q);for(auto q:x.z1)put(q);for(auto q:x.z2)put(q);
            for(double q:x.gaussian_stochastic)*dst++=q;
        }
        rsd_error.clear();return 0;
    }catch(const std::exception& e){rsd_error=e.what();return 1;}
}
}
extern "C" int marisa_tree_rsd_batch(std::size_t count,const double* vectors,const double* power,
    const double* transfer,const double* bias,const double* config,const double* recon,int post,int method,double* output) {
    return batch_impl(count,vectors,power,transfer,bias,config,recon,post,method,false,0.,0.,output);
}
extern "C" int marisa_tree_rsd_batch_fixed(std::size_t count,const double* vectors,const double* power,
    const double* transfer,const double* bias,const double* config,const double* recon,int method,
    double fnl_rec,double denominator_floor,double* output) {
    return batch_impl(count,vectors,power,transfer,bias,config,recon,1,method,true,fnl_rec,denominator_floor,output);
}
extern "C" int marisa_tree_rsd_batch_response(std::size_t count,const double* vectors,const double* power,
    const double* transfer,const double* bias,const double* config,const double* response,double* output) {
    const double recon[6]={0.,1.,0.,0.,0.,0.};
    return batch_impl(count,vectors,power,transfer,bias,config,recon,1,0,true,0.,0.,output,response);
}
