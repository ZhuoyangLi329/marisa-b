#include "uv_subtraction.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "PowerSpectrum.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;

Vec3 scale(const Vec3& value,double factor) {
    return Vec3{factor*value.x,factor*value.y,factor*value.z};
}

Vec3 unit(const Vec3& value) {
    const double length=marisa_b_halo_v1::norm(value);
    if (!(length>0.0)) throw std::invalid_argument("UV direction has zero length");
    return scale(value,1.0/length);
}

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int count,double lower,double upper) {
    if (count<1 || !(upper>lower)) {
        throw std::invalid_argument("invalid UV-restoration quadrature");
    }
    QuadratureRule result;
    result.nodes.resize(static_cast<std::size_t>(count));
    result.weights.resize(static_cast<std::size_t>(count));
    const int half=(count+1)/2;
    const double midpoint=0.5*(lower+upper);
    const double half_width=0.5*(upper-lower);
    for (int index=0;index<half;++index) {
        double root=std::cos(
            kPi*(static_cast<double>(index)+0.75)
            /(static_cast<double>(count)+0.5));
        double derivative=0.0;
        for (int iteration=0;iteration<100;++iteration) {
            double previous=1.0;
            double current=root;
            for (int order=2;order<=count;++order) {
                const double next=
                    ((2.0*order-1.0)*root*current
                     -(order-1.0)*previous)/order;
                previous=current;
                current=next;
            }
            const double pn=count==1?root:current;
            const double pnm1=count==1?1.0:previous;
            derivative=count*(root*pn-pnm1)/(root*root-1.0);
            const double update=pn/derivative;
            root-=update;
            if (std::fabs(update)
                <4.0*std::numeric_limits<double>::epsilon()) break;
        }
        const double weight=
            2.0/((1.0-root*root)*derivative*derivative);
        result.nodes[static_cast<std::size_t>(index)]=
            midpoint-half_width*root;
        result.nodes[static_cast<std::size_t>(count-1-index)]=
            midpoint+half_width*root;
        result.weights[static_cast<std::size_t>(index)]=half_width*weight;
        result.weights[static_cast<std::size_t>(count-1-index)]=
            half_width*weight;
    }
    return result;
}

}  // namespace

SparsePolynomial p13_uv_coefficient(
    const FieldKernelProvider& provider) {
    constexpr double inverse_315=1.0/315.0;
    if (provider.name()=="matter_spt") {
        return SparsePolynomial::constant(-61.0*inverse_315);
    }
    if (provider.name()!="eft_v2_cobra_bias") {
        throw std::invalid_argument(
            "analytic P13 UV coefficient supports only matter_spt and "
            "eft_v2_cobra_bias providers");
    }
    return inverse_315*(
        -61.0*SparsePolynomial::variable(ParameterId::B1)
        -576.0*SparsePolynomial::variable(ParameterId::Gamma2)
        +672.0*SparsePolynomial::variable(ParameterId::Gamma21));
}

double normalized_linear_power_uv_tail_moment(
    const PowerSpectrum& power,double lower,
    double sampled_upper,int quadrature_order) {
    if (!(lower>0.0 && sampled_upper>lower
          && std::isfinite(lower)
          && std::isfinite(sampled_upper)
          && quadrature_order>=16)) {
        throw std::invalid_argument(
            "invalid linear-power UV-tail moment configuration");
    }
    const double ratio=std::sqrt(2.0);
    double effective_upper=sampled_upper;
    double high_inner=power(effective_upper/ratio);
    double high_value=power(effective_upper);
    while (!(high_inner>0.0 && high_value>0.0
             && std::isfinite(high_inner)
             && std::isfinite(high_value))
           && effective_upper/ratio>lower*(1.0+1.0e-12)) {
        effective_upper/=ratio;
        high_inner=power(effective_upper/ratio);
        high_value=power(effective_upper);
    }
    if (!(high_inner>0.0 && high_value>0.0
          && std::isfinite(high_inner)
          && std::isfinite(high_value))) {
        throw std::invalid_argument(
            "linear-power UV-tail moment sampled invalid power");
    }
    const double high_slope=
        std::log(high_value/high_inner)/std::log(ratio);
    if (!(high_slope<-1.0)) {
        throw std::domain_error(
            "linear-power UV-tail moment has a non-convergent high-k tail");
    }
    const QuadratureRule logarithmic=gauss_legendre(
        quadrature_order,std::log(lower),std::log(effective_upper));
    long double sampled=0.0L;
    for (std::size_t index=0;index<logarithmic.nodes.size();++index) {
        const double q=std::exp(logarithmic.nodes[index]);
        const double value=power(q);
        if (!(value>0.0 && std::isfinite(value))) {
            throw std::invalid_argument(
                "linear-power UV-tail moment encountered invalid power");
        }
        sampled+=static_cast<long double>(logarithmic.weights[index])
                 *q*value;
    }
    const long double extrapolated=
        static_cast<long double>(high_value)*effective_upper
        /(-high_slope-1.0);
    const double normalized=static_cast<double>(
        (sampled+extrapolated)/(2.0L*kPi*kPi));
    if (!(normalized>=0.0 && std::isfinite(normalized))) {
        throw std::runtime_error(
            "linear-power UV-tail moment is invalid");
    }
    return normalized;
}

UvSubtraction::UvSubtraction(
    const DiagramAssembler& assembler,UvSubtractionConfig config)
    : assembler_(assembler),config_(config) {
    if (!(config_.mu_ren>0.0 && config_.asymptotic_factor>1.0)) {
        throw std::invalid_argument("invalid UV subtraction scale configuration");
    }
    if (config_.richardson_order!=2) {
        throw std::invalid_argument("EFT-v2 currently supports the audited q^-2 Richardson UV limit");
    }
    double external_scale=0.0;
    for (const Vec3& value:assembler_.external()) {
        external_scale=std::max(external_scale,marisa_b_halo_v1::norm(value));
    }
    reference_q_=config_.asymptotic_factor*std::max(external_scale,config_.mu_ren);
}

std::string UvSubtraction::direction_key(const Vec3& direction) const {
    const Vec3 normalized=unit(direction);
    std::ostringstream output;
    output<<std::fixed<<std::setprecision(12)
          <<normalized.x<<','<<normalized.y<<','<<normalized.z;
    return output.str();
}

UvSubtractionComponents UvSubtraction::compute_asymptotic(const Vec3& direction) const {
    const Vec3 normalized=unit(direction);
    const Vec3 q1=scale(normalized,reference_q_);
    const Vec3 q2=scale(normalized,2.0*reference_q_);
    UvSubtractionComponents result;
    // A(q)=A_inf+c/q^2+O(q^-4), hence A_inf=(4 A(2q)-A(q))/3.
    result.B321II=(1.0/3.0)*(
        4.0*assembler_.tadpole_coefficient(Diagram::B321II,q2)
        -assembler_.tadpole_coefficient(Diagram::B321II,q1));
    result.B411=(1.0/3.0)*(
        4.0*assembler_.tadpole_coefficient(Diagram::B411,q2)
        -assembler_.tadpole_coefficient(Diagram::B411,q1));
    result.total=result.B321II+result.B411;
    return result;
}

UvSubtractionComponents UvSubtraction::asymptotic_coefficients(
    const Vec3& unit_direction) const {
    const std::string key=direction_key(unit_direction);
    const auto found=cache_.find(key);
    if (found!=cache_.end()) return found->second;
    const UvSubtractionComponents value=compute_asymptotic(unit_direction);
    cache_.emplace(key,value);
    return value;
}

UvSubtractionComponents UvSubtraction::subleading_coefficients(
    const Vec3& unit_direction) const {
    const Vec3 direction=unit(unit_direction);
    const auto antipodal=[this,&direction](
        Diagram diagram,double radius) {
        const Vec3 positive=scale(direction,radius);
        const Vec3 negative=scale(direction,-radius);
        return 0.5*(
            assembler_.tadpole_coefficient(diagram,positive)
            +assembler_.tadpole_coefficient(diagram,negative));
    };
    const auto coefficient=[this,&antipodal](Diagram diagram) {
        const double q1=reference_q_;
        const SparsePolynomial value1=antipodal(diagram,q1);
        const SparsePolynomial value2=antipodal(diagram,2.0*q1);
        const SparsePolynomial value4=antipodal(diagram,4.0*q1);
        const SparsePolynomial estimate_q=
            (4.0/3.0)*q1*q1*(value1-value2);
        const SparsePolynomial estimate_2q=
            (16.0/3.0)*q1*q1*(value2-value4);
        // If C(q)=C0+C2/q^2+C4/q^4+..., the two estimates are
        // C2+5 C4/(4 q^2) and C2+5 C4/(16 q^2).  This combination
        // cancels C4 and leaves O(q^-4) relative contamination.
        return (1.0/3.0)*(4.0*estimate_2q-estimate_q);
    };
    UvSubtractionComponents result;
    result.B321II=coefficient(Diagram::B321II);
    result.B411=coefficient(Diagram::B411);
    result.total=result.B321II+result.B411;
    return result;
}

UvSubtractionComponents
UvSubtraction::angular_averaged_subleading_coefficients(
    int n_mu,int n_phi) const {
    const QuadratureRule mu=gauss_legendre(n_mu,-1.0,1.0);
    const QuadratureRule phi=gauss_legendre(n_phi,0.0,2.0*kPi);
    UvSubtractionComponents result;
    for (int imu=0;imu<n_mu;++imu) {
        const double z=mu.nodes[static_cast<std::size_t>(imu)];
        const double transverse=std::sqrt(std::max(0.0,1.0-z*z));
        for (int iphi=0;iphi<n_phi;++iphi) {
            const double angle=phi.nodes[static_cast<std::size_t>(iphi)];
            const Vec3 direction{
                transverse*std::cos(angle),
                transverse*std::sin(angle),z};
            const double weight=
                mu.weights[static_cast<std::size_t>(imu)]
                *phi.weights[static_cast<std::size_t>(iphi)]
                /(4.0*kPi);
            const UvSubtractionComponents value=
                subleading_coefficients(direction);
            result.B321II+=weight*value.B321II;
            result.B411+=weight*value.B411;
        }
    }
    result.total=result.B321II+result.B411;
    return result;
}

UvSubtractionComponents UvSubtraction::integrands(const Vec3& loop) const {
    const double radius=marisa_b_halo_v1::norm(loop);
    if (!(radius>0.0)) throw std::invalid_argument("UV subtraction loop has zero length");
    UvSubtractionComponents result=asymptotic_coefficients(scale(loop,1.0/radius));
    const double Pq=assembler_.loop_power(loop);
    result.B321II*=Pq;
    result.B411*=Pq;
    result.total=result.B321II+result.B411;
    return result;
}

}  // namespace marisa_b_eft_v2
