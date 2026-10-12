#include "ir_resummation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "PowerSpectrum.h"
#include "diagram_assembler.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;

double spherical_j0(double x) {
    if (std::fabs(x)<1.0e-4) {
        const double x2=x*x;
        return 1.0-x2/6.0+x2*x2/120.0-x2*x2*x2/5040.0;
    }
    return std::sin(x)/x;
}

double spherical_j2(double x) {
    if (std::fabs(x)<1.0e-3) {
        const double x2=x*x;
        return x2/15.0-x2*x2/210.0+x2*x2*x2/7560.0;
    }
    return ((3.0/(x*x*x)-1.0/x)*std::sin(x)-3.0*std::cos(x)/(x*x));
}

}  // namespace

double eisenstein_hu_nowiggle_shape(
    double k,const IrResummationConfig& config) {
    if (!(k>0.0)) return 0.0;
    const double h=config.hubble;
    const double omega_m=config.omega_m;
    const double omega_b=config.omega_b;
    const double theta=config.cmb_temperature/2.7;
    if (!(h>0.0 && omega_m>0.0 && omega_b>=0.0 && omega_b<omega_m
          && theta>0.0)) {
        throw std::invalid_argument("invalid cosmology for Eisenstein-Hu no-wiggle shape");
    }
    const double baryon_fraction=omega_b/omega_m;
    const double omega_m_h2=omega_m*h*h;
    const double alpha_gamma=
        1.0-0.328*std::log(431.0*omega_m_h2)*baryon_fraction
        +0.38*std::log(22.3*omega_m_h2)*baryon_fraction*baryon_fraction;
    const double sound=44.5*std::log(9.83/omega_m_h2)
        /std::sqrt(1.0+10.0*std::pow(omega_b*h*h,0.75))*h;
    const double gamma_effective=omega_m*h*(
        alpha_gamma+(1.0-alpha_gamma)/(1.0+std::pow(0.43*k*sound,4)));
    const double q=k*theta*theta/gamma_effective;
    const double logarithm=std::log(2.0*std::exp(1.0)+1.8*q);
    const double denominator=logarithm+(14.2+731.0/(1.0+62.5*q))*q*q;
    const double transfer=logarithm/denominator;
    return std::pow(k,config.scalar_tilt)*transfer*transfer;
}

IrResummation::IrResummation(
    const PowerSpectrum& linear_power,IrResummationConfig config)
    : linear_power_(linear_power),config_(config),
      no_wiggle_(*this,PowerView::Kind::NoWiggle),
      loop_(*this,PowerView::Kind::Loop),tree_(*this,PowerView::Kind::TreeNlo) {
    if (!(config_.kmin>0.0 && config_.kmax>config_.kmin
          && config_.grid_size>=64 && config_.smoothing_lambda>0.0
          && config_.smoothing_truncate>=2.0 && config_.lambda_ir>0.0
          && config_.sound_horizon>0.0 && config_.hubble>0.0
          && config_.omega_m>0.0 && config_.omega_b>=0.0
          && config_.omega_b<config_.omega_m && config_.cmb_temperature>0.0)) {
        throw std::invalid_argument("invalid IR-resummation configuration");
    }
    log_k_.resize(config_.grid_size);
    std::vector<double> smooth_input(config_.grid_size);
    const double lower=std::log(config_.kmin);
    const double spacing=std::log(config_.kmax/config_.kmin)/(config_.grid_size-1);
    for (int index=0;index<config_.grid_size;++index) {
        log_k_[index]=lower+spacing*index;
        const double k=std::exp(log_k_[index]);
        const double p=linear_power_(k);
        if (!(p>0.0 && std::isfinite(p))) {
            throw std::invalid_argument("IR split sampled non-positive linear power");
        }
        if (config_.split==SmoothSplitPrescription::GaussianEisensteinHuRatio) {
            smooth_input[index]=p/eisenstein_hu_nowiggle_shape(k,config_);
        } else {
            smooth_input[index]=std::log(k*k*k*p);
        }
    }
    const int radius=static_cast<int>(std::ceil(
        config_.smoothing_truncate*config_.smoothing_lambda/spacing));
    std::vector<double> weights(2*radius+1);
    double weight_sum=0.0;
    for (int offset=-radius;offset<=radius;++offset) {
        const double displacement=spacing*offset/config_.smoothing_lambda;
        weights[offset+radius]=std::exp(-0.5*displacement*displacement);
        weight_sum+=weights[offset+radius];
    }
    log_p_nw_.resize(config_.grid_size);
    const double low_slope=smooth_input[1]-smooth_input[0];
    const double high_slope=smooth_input.back()-smooth_input[smooth_input.size()-2];
    for (int index=0;index<config_.grid_size;++index) {
        double smoothed=0.0;
        for (int offset=-radius;offset<=radius;++offset) {
            const int source=index+offset;
            double value=0.0;
            if (source<0) value=smooth_input.front()+source*low_slope;
            else if (source>=config_.grid_size) {
                value=smooth_input.back()+(source-(config_.grid_size-1))*high_slope;
            } else value=smooth_input[source];
            smoothed+=weights[offset+radius]*value;
        }
        smoothed/=weight_sum;
        if (config_.split==SmoothSplitPrescription::GaussianEisensteinHuRatio) {
            if (!(smoothed>0.0 && std::isfinite(smoothed))) {
                throw std::runtime_error("Gaussian EH-ratio split produced non-positive power");
            }
            // Store the smoothed, dimensionful P11/P_EH amplitude rather than
            // P_nw itself.  Multiplying by the analytic EH shape only after
            // interpolation makes an exactly EH-shaped input a fixed point of
            // the split, without interpolation curvature artifacts.
            log_p_nw_[index]=std::log(smoothed);
        } else {
            log_p_nw_[index]=smoothed-3.0*log_k_[index];
        }
    }
    sigma2_=std::isfinite(config_.sigma2_override)
        ?config_.sigma2_override:compute_sigma2();
    if (!(sigma2_>=0.0 && std::isfinite(sigma2_))) {
        throw std::invalid_argument("IR displacement variance must be finite and non-negative");
    }
}

double IrResummation::interpolate_log_nw(double k) const {
    if (!(k>0.0)) return -std::numeric_limits<double>::infinity();
    const double x=std::log(k);
    const auto extrapolate=[](
        double x,double x0,double x1,double y0,double y1) {
        return y0+(x-x0)*(y1-y0)/(x1-x0);
    };
    if (x<=log_k_.front()) {
        return extrapolate(x,log_k_[0],log_k_[1],log_p_nw_[0],log_p_nw_[1]);
    }
    if (x>=log_k_.back()) {
        const std::size_t last=log_k_.size()-1;
        return extrapolate(
            x,log_k_[last],log_k_[last-1],log_p_nw_[last],log_p_nw_[last-1]);
    }
    const auto right=std::upper_bound(log_k_.begin(),log_k_.end(),x);
    const std::size_t index=right-log_k_.begin();
    return extrapolate(
        x,log_k_[index-1],log_k_[index],log_p_nw_[index-1],log_p_nw_[index]);
}

double IrResummation::evaluate(PowerView::Kind kind,double k) const {
    if (!(k>0.0)) return 0.0;
    double smooth=std::exp(interpolate_log_nw(k));
    if (config_.split==SmoothSplitPrescription::GaussianEisensteinHuRatio) {
        smooth*=eisenstein_hu_nowiggle_shape(k,config_);
    }
    if (kind==PowerView::Kind::NoWiggle) return smooth;
    const double linear=linear_power_(k);
    const double x=k*k*sigma2_;
    const double damping=std::exp(-x);
    if (kind==PowerView::Kind::Loop) return smooth+damping*(linear-smooth);
    return smooth+(1.0+x)*damping*(linear-smooth);
}

real IrResummation::PowerView::Evaluate(real k) const {
    return owner_.evaluate(kind_,k);
}

const Cosmology& IrResummation::PowerView::GetCosmology() const {
    return owner_.linear_power_.GetCosmology();
}

double IrResummation::compute_sigma2() const {
    constexpr int intervals=4096;
    const double lower=std::log(std::min(config_.kmin,1.0e-6));
    const double upper=std::log(config_.lambda_ir);
    const double step=(upper-lower)/intervals;
    long double integral=0.0L;
    for (int index=0;index<=intervals;++index) {
        const double p=std::exp(lower+step*index);
        const double power=config_.sigma_power==SigmaPowerPrescription::FullP11
            ?linear_power_(p):evaluate(PowerView::Kind::NoWiggle,p);
        const double x=config_.sound_horizon*p;
        const double window=1.0-spherical_j0(x)+2.0*spherical_j2(x);
        const double value=p*power*window;
        const int weight=(index==0 || index==intervals)?1:(index%2==0?2:4);
        integral+=weight*value;
    }
    return static_cast<double>(integral*step/3.0)/(6.0*kPi*kPi);
}

std::string IrResummation::metadata_json() const {
    std::ostringstream output;
    output<<std::setprecision(17)
          <<"{\"schema\":\"marisa-b-eft-v2-ir-v1\","
          <<"\"split\":\""
          <<(config_.split==SmoothSplitPrescription::GaussianEisensteinHuRatio
              ?"Gaussian smoothing of P11/P_EH on log k"
              :"Gaussian smoothing of log(k^3 P11) on log k")<<"\","
          <<"\"kmin\":"<<config_.kmin<<",\"kmax\":"<<config_.kmax<<','
          <<"\"grid_size\":"<<config_.grid_size<<','
          <<"\"smoothing_lambda\":"<<config_.smoothing_lambda<<','
          <<"\"smoothing_truncate\":"<<config_.smoothing_truncate<<','
          <<"\"cosmology\":{"<<"\"h\":"<<config_.hubble
          <<",\"n_s\":"<<config_.scalar_tilt<<",\"Omega_m\":"<<config_.omega_m
          <<",\"Omega_b\":"<<config_.omega_b<<"},"
          <<"\"Lambda_IR\":"<<config_.lambda_ir<<','
          <<"\"r_s\":"<<config_.sound_horizon<<','
          <<"\"sigma_power\":\""
          <<(config_.sigma_power==SigmaPowerPrescription::FullP11?"P11":"P_nw")
          <<"\",\"sigma2\":"<<sigma2_<<"}";
    return output.str();
}

DiagramTemplates evaluate_ir_resummed(
    const IrResummation& resummation,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& integration) {
    DiagramTemplates result=evaluate_direct(
        resummation.loop_power(),closed_triangle,provider,integration);
    const DiagramAssembler tree(
        resummation.tree_power(),closed_triangle,provider);
    result.tree=tree.tree();
    result.total=result.tree+result.one_loop;
    return result;
}

}  // namespace marisa_b_eft_v2
