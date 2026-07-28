#include "stochastic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "PowerSpectrum.h"
#include "ir_safe_integrands.h"
#include "uv_subtraction.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int count,double lower,double upper) {
    if (count<1 || !(upper>lower)) throw std::invalid_argument("invalid stochastic quadrature");
    QuadratureRule result;
    result.nodes.resize(count);
    result.weights.resize(count);
    const int half=(count+1)/2;
    const double midpoint=0.5*(lower+upper);
    const double half_width=0.5*(upper-lower);
    for (int index=0;index<half;++index) {
        double root=std::cos(kPi*(index+0.75)/(count+0.5));
        double derivative=0.0;
        for (int iteration=0;iteration<100;++iteration) {
            double previous=1.0;
            double current=root;
            for (int order=2;order<=count;++order) {
                const double next=((2.0*order-1.0)*root*current
                                   -(order-1.0)*previous)/order;
                previous=current;
                current=next;
            }
            const double pn=count==1?root:current;
            const double pnm1=count==1?1.0:previous;
            derivative=count*(root*pn-pnm1)/(root*root-1.0);
            const double update=pn/derivative;
            root-=update;
            if (std::fabs(update)<4.0*std::numeric_limits<double>::epsilon()) break;
        }
        const double weight=2.0/((1.0-root*root)*derivative*derivative);
        result.nodes[index]=midpoint-half_width*root;
        result.nodes[count-1-index]=midpoint+half_width*root;
        result.weights[index]=half_width*weight;
        result.weights[count-1-index]=half_width*weight;
    }
    return result;
}

Vec3 scale(const Vec3& value,double factor) {
    return Vec3{factor*value.x,factor*value.y,factor*value.z};
}

SparsePolynomial linear_term(const SparsePolynomial& kernel,ParameterId id) {
    return kernel.coefficient(MonomialKey::variable(id))
           *SparsePolynomial::variable(id);
}

SparsePolynomial power_bias_kernel(const std::vector<Vec3>& momenta) {
    const SparsePolynomial full=deterministic_kernel(momenta);
    SparsePolynomial result;
    for (ParameterId id:{ParameterId::B1,ParameterId::B2,
                         ParameterId::Gamma2,ParameterId::Gamma21}) {
        result+=linear_term(full,id);
    }
    return result;
}

class PowerBiasKernelProvider final:
    public FieldKernelProvider {
public:
    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override {
        return power_bias_kernel(momenta);
    }

    std::string_view name() const noexcept override {
        return "eft_v2_renormalized_stochastic_density";
    }
};

class ZeroStochasticDirectionProvider final:
    public FieldKernelProvider {
public:
    SparsePolynomial deterministic(
        const std::vector<Vec3>&) const override {
        return {};
    }

    std::string_view name() const noexcept override {
        return "zero_stochastic_direction";
    }
};

class ZeroMarkedStochasticProvider final:
    public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>&,
        const Vec3&) const override {
        return {};
    }
    std::string_view name() const noexcept override {
        return "zero_marked_stochastic";
    }
};

/*
 * Direction with respect to Bshot_C-1 at fixed Pshot=0.  The raw COBRA
 * relation is 2 d1 (1+Pshot)=b1 Bshot_C, hence the marked epsilon-delta
 * kernel has derivative b1/2.  Its pure-epsilon K0 derivative is zero, so
 * K0 does not contribute to this *parameter direction*.  This statement
 * must not be confused with the fixed Poisson point K0=1: correlations
 * between an external density mark and internal reconstruction-shift noise
 * survive the estimator subtraction and are assembled separately in
 * poisson_reconstruction.cpp.
 */
class BshotResidualMarkedProvider final:
    public MarkedFieldKernelProvider {
public:
    explicit BshotResidualMarkedProvider(
        const FieldKernelProvider& base)
        :base_(base) {}

    SparsePolynomial marked(
        const std::vector<Vec3>& matter_momenta,
        const Vec3&) const override {
        if (matter_momenta.size()!=1) return {};
        return 0.5*base_.deterministic(matter_momenta);
    }
    std::string_view name() const noexcept override {
        return "Bshot_residual_marked_epsilon_delta";
    }

private:
    const FieldKernelProvider& base_;
};

std::array<double,4> stochastic_bias_projection(
    const SparsePolynomial& full) {
    const auto coefficient=[&full](ParameterId id) {
        return full.coefficient(MonomialKey::variable(id));
    };
    const double gamma21=coefficient(ParameterId::Gamma21);
    return {{
        coefficient(ParameterId::B1),
        coefficient(ParameterId::B2),
        coefficient(ParameterId::Gamma2)-(4.0/7.0)*gamma21,
        -(4.0/7.0)*gamma21}};
}

class StochasticDirectionProvider final:
    public FieldKernelProvider {
public:
    StochasticDirectionProvider(
        const FieldKernelProvider& base,
        std::size_t direction)
        :base_(base),direction_(direction) {
        if (direction_>=4) {
            throw std::invalid_argument(
                "stochastic direction index is out of range");
        }
    }

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override {
        const auto projection=
            stochastic_bias_projection(
                base_.deterministic(momenta));
        return SparsePolynomial::constant(
            projection[direction_]);
    }

    std::string_view name() const noexcept override {
        return "eft_v2_stochastic_direction";
    }

private:
    const FieldKernelProvider& base_;
    std::size_t direction_=0;
};

std::array<double,4> stochastic_bias_kernel(
    const std::vector<Vec3>& momenta) {
    return stochastic_bias_projection(deterministic_kernel(momenta));
}

SparsePolynomial reconstructed_leading_bshot_residual_generating_shape_impl(
    const PowerSpectrum& tree_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    bool stochastic_shift) {
    const BshotResidualMarkedProvider marked(base_provider);
    static const ZeroMarkedStochasticProvider zero_marked;
    const MarkedFieldKernelProvider& shift_marked=
        stochastic_shift
        ?static_cast<const MarkedFieldKernelProvider&>(marked)
        :static_cast<const MarkedFieldKernelProvider&>(zero_marked);
    SparsePolynomial result;
    for (std::size_t matter_leg=0;
         matter_leg<closed_triangle.size();
         ++matter_leg) {
        const Vec3 matter=scale(
            closed_triangle[matter_leg],-1.0);
        const double power=tree_power(
            marisa_b_halo_v1::norm(matter));
        const SparsePolynomial deterministic=
            base_provider.deterministic({matter});
        for (std::size_t noise_leg=0;
             noise_leg<closed_triangle.size();
             ++noise_leg) {
            if (noise_leg==matter_leg) continue;
            const Vec3 noise=scale(
                closed_triangle[noise_leg],-1.0);
            const SparsePolynomial marked_second_order=
                reconstructed_marked_field_kernel(
                    base_provider,marked,
                    base_provider,shift_marked,
                    reconstruction,{matter},noise);
            result+=power*deterministic
                    *marked_second_order;
        }
    }
    return result;
}

std::array<double,4> add(
    std::array<double,4> left,const std::array<double,4>& right,double factor=1.0) {
    for (std::size_t index=0;index<left.size();++index) left[index]+=factor*right[index];
    return left;
}

std::array<double,4> scale(
    std::array<double,4> value,double factor) {
    for (double& item:value) item*=factor;
    return value;
}

struct CrossIntegrand {
    std::array<SparsePolynomial,4> values;
    CrossIntegrand& operator+=(const CrossIntegrand& other) {
        for (std::size_t index=0;index<values.size();++index) {
            values[index]+=other.values[index];
        }
        return *this;
    }
};

MixedStochasticPowerTemplates mixed_cross_power_impl(
    const PowerSpectrum& power,double k,
    const StochasticIntegrationConfig& config) {
    if (!(k>0.0 && config.qmin>0.0 && config.qmax>config.qmin
          && config.mu_ren>0.0 && config.asymptotic_factor>1.0)) {
        throw std::invalid_argument("invalid mixed stochastic loop configuration");
    }
    const Vec3 external{0.0,0.0,k};
    const SparsePolynomial K1=power_bias_kernel({external});
    const double Pk=power(k);
    const SoftRegionMapper mapper({Vec3{},external},config.qmax);
    const double reference=config.asymptotic_factor*std::max(k,config.mu_ren);
    const QuadratureRule mu=gauss_legendre(config.n_mu,-1.0,1.0);
    MixedStochasticPowerTemplates result;
    const double measure=1.0/(4.0*kPi*kPi);
    for (int imu=0;imu<config.n_mu;++imu) {
        const double cosine=mu.nodes[imu];
        const Vec3 direction{std::sqrt(std::max(0.0,1.0-cosine*cosine)),0.0,cosine};
        const Vec3 q1=scale(direction,reference);
        const Vec3 q2=scale(direction,2.0*reference);
        const SparsePolynomial A1=power_bias_kernel({external,q1,scale(q1,-1.0)});
        const SparsePolynomial A2=power_bias_kernel({external,q2,scale(q2,-1.0)});
        const SparsePolynomial Ainf=(1.0/3.0)*(4.0*A2-A1);
        const auto D1=stochastic_bias_kernel({external,q1,scale(q1,-1.0)});
        const auto D2=stochastic_bias_kernel({external,q2,scale(q2,-1.0)});
        const auto Dinf=scale(add(scale(D2,4.0),D1,-1.0),1.0/3.0);

        std::vector<double> boundaries={config.qmin,config.qmax};
        for (double value:mapper.radial_boundaries(direction)) {
            if (value>config.qmin) boundaries.push_back(value);
        }
        std::sort(boundaries.begin(),boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(),boundaries.end(),[](double left,double right) {
            return std::fabs(left-right)<=1.0e-12*std::max({left,right,1.0});
        }),boundaries.end());
        for (std::size_t segment=0;segment+1<boundaries.size();++segment) {
            const QuadratureRule radial=gauss_legendre(
                config.n_radial,std::log(boundaries[segment]),
                std::log(boundaries[segment+1]));
            for (std::size_t iq=0;iq<radial.nodes.size();++iq) {
                const double radius=std::exp(radial.nodes[iq]);
                const Vec3 mapped=scale(direction,radius);
                const CrossIntegrand bare22=mapper.map(mapped,[&](const Vec3& q) {
                    const Vec3 shifted=marisa_b_halo_v1::subtract(external,q);
                    const SparsePolynomial KA=power_bias_kernel({q,shifted});
                    const auto KD=stochastic_bias_kernel({q,shifted});
                    CrossIntegrand value;
                    const double prefactor=2.0*power(marisa_b_halo_v1::norm(q))
                                          *power(marisa_b_halo_v1::norm(shifted));
                    for (std::size_t index=0;index<4;++index) {
                        value.values[index]=prefactor*KD[index]*KA;
                    }
                    return value;
                });
                CrossIntegrand subtraction22;
                CrossIntegrand bare13;
                CrossIntegrand subtraction13;
                if (radius<=config.qmax) {
                    const Vec3 minus_q=scale(mapped,-1.0);
                    const SparsePolynomial KA2zero=
                        power_bias_kernel({mapped,minus_q});
                    const auto KD2zero=
                        stochastic_bias_kernel({mapped,minus_q});
                    const SparsePolynomial KA3=
                        power_bias_kernel({external,mapped,minus_q});
                    const auto KD3=
                        stochastic_bias_kernel({external,mapped,minus_q});
                    const double Pq=power(radius);
                    const double prefactor22=2.0*Pq*Pq;
                    const double prefactor13=3.0*Pk*Pq;
                    for (std::size_t index=0;index<4;++index) {
                        subtraction22.values[index]=
                            prefactor22*KD2zero[index]*KA2zero;
                        bare13.values[index]=
                            prefactor13*(KD3[index]*K1);
                        subtraction13.values[index]=
                            prefactor13*(Dinf[index]*K1);
                    }
                    bare13.values[0]+=prefactor13*KA3;
                    subtraction13.values[0]+=prefactor13*Ainf;
                }
                const double weight=radial.weights[iq]*radius*radius*radius
                                    *mu.weights[imu]*measure;
                for (std::size_t index=0;index<4;++index) {
                    result.P22_bare[index]+=weight*bare22.values[index];
                    result.P13_bare[index]+=weight*bare13.values[index];
                    result.P22_zero_lag_subtraction[index]+=
                        weight*subtraction22.values[index];
                    result.P13_bias_subtraction[index]+=
                        weight*subtraction13.values[index];
                    result.P22[index]+=weight*(
                        bare22.values[index]-subtraction22.values[index]);
                    result.P13[index]+=weight*(
                        bare13.values[index]-subtraction13.values[index]);
                }
                ++result.integration_nodes;
            }
        }
    }
    if (config.restore_p13_uv_tail) {
        const EftBiasKernelProvider provider;
        const SparsePolynomial alphaA=p13_uv_coefficient(provider);
        const auto alphaD=stochastic_bias_projection(alphaA);
        const double moment=normalized_linear_power_uv_tail_moment(
            power,config.qmax,config.uv_tail_kmax,
            config.uv_tail_quadrature_order);
        const double prefactor=0.5*Pk*k*k*moment;
        for (std::size_t index=0;index<4;++index) {
            result.P13_uv_tail_restoration[index]=
                prefactor*alphaD[index]*K1;
            result.P13[index]+=
                result.P13_uv_tail_restoration[index];
        }
        result.P13_uv_tail_restoration[0]+=prefactor*alphaA;
        result.P13[0]+=prefactor*alphaA;
    }
    for (std::size_t index=0;index<4;++index) {
        result.value[index]=result.P22[index]+result.P13[index];
    }
    return result;
}

struct ReconstructedStochasticKernelSystem {
    const FieldKernelProvider& base;
    marisa_b_halo_v1::ReconstructionConfig reconstruction;
    ReconstructedStochasticMap map;
    PowerBiasKernelProvider density_base;
    std::array<StochasticDirectionProvider,4> direction;

    ReconstructedStochasticKernelSystem(
        const FieldKernelProvider& provider,
        const marisa_b_halo_v1::ReconstructionConfig& config,
        ReconstructedStochasticMap selected_map)
        :base(provider),
         reconstruction(config),
         map(selected_map),
         direction{{
             StochasticDirectionProvider(provider,0),
             StochasticDirectionProvider(provider,1),
             StochasticDirectionProvider(provider,2),
             StochasticDirectionProvider(provider,3)}} {}

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const {
        static const ZeroStochasticDirectionProvider zero;
        return reconstructed_field_kernel_variation_with_shift(
            density_base,zero,base,zero,
            reconstruction,momenta).value;
    }

    std::array<SparsePolynomial,4> stochastic(
        const std::vector<Vec3>& momenta) const {
        static const ZeroStochasticDirectionProvider zero;
        std::array<SparsePolynomial,4> result;
        for (std::size_t index=0;
             index<result.size();
             ++index) {
            const FieldKernelProvider& shift_direction=
                map==ReconstructedStochasticMap::
                        TiedDensityAndShift
                ?static_cast<const FieldKernelProvider&>(
                    direction[index])
                :static_cast<const FieldKernelProvider&>(zero);
            result[index]=
                reconstructed_field_kernel_variation_with_shift(
                    density_base,direction[index],
                    base,shift_direction,
                    reconstruction,momenta).direction;
        }
        return result;
    }
};

struct ReconstructedAsymptotic {
    SparsePolynomial deterministic;
    std::array<SparsePolynomial,4> stochastic;
};

ReconstructedAsymptotic reconstructed_asymptotic_limit(
    const ReconstructedStochasticKernelSystem& kernels,
    const Vec3& external,
    const Vec3& unit_direction,
    double reference) {
    const Vec3 q1=scale(unit_direction,reference);
    const Vec3 q2=scale(unit_direction,2.0*reference);
    const SparsePolynomial A1=
        kernels.deterministic(
            {external,q1,scale(q1,-1.0)});
    const SparsePolynomial A2=
        kernels.deterministic(
            {external,q2,scale(q2,-1.0)});
    const auto D1=kernels.stochastic(
        {external,q1,scale(q1,-1.0)});
    const auto D2=kernels.stochastic(
        {external,q2,scale(q2,-1.0)});
    ReconstructedAsymptotic result;
    result.deterministic=(1.0/3.0)*(4.0*A2-A1);
    for (std::size_t index=0;
         index<result.stochastic.size();
         ++index) {
        result.stochastic[index]=
            (1.0/3.0)*(4.0*D2[index]-D1[index]);
    }
    return result;
}

ReconstructedAsymptotic reconstructed_p22_zero_lag_limit(
    const ReconstructedStochasticKernelSystem& kernels,
    const Vec3& external,
    const Vec3& unit_direction,
    double reference) {
    const auto antipodal=[&](
        double radius,
        SparsePolynomial& deterministic,
        std::array<SparsePolynomial,4>& stochastic) {
        const Vec3 positive=
            scale(unit_direction,radius);
        const Vec3 negative=
            scale(unit_direction,-radius);
        const auto evaluate=[&](
            const Vec3& q,
            SparsePolynomial& A,
            std::array<SparsePolynomial,4>& D) {
            A=kernels.deterministic(
                {q,marisa_b_halo_v1::subtract(
                        external,q)});
            D=kernels.stochastic(
                {q,marisa_b_halo_v1::subtract(
                        external,q)});
        };
        SparsePolynomial positive_A;
        SparsePolynomial negative_A;
        std::array<SparsePolynomial,4> positive_D;
        std::array<SparsePolynomial,4> negative_D;
        evaluate(positive,positive_A,positive_D);
        evaluate(negative,negative_A,negative_D);
        deterministic=0.5*(positive_A+negative_A);
        for (std::size_t index=0;
             index<stochastic.size();
             ++index) {
            stochastic[index]=
                0.5*(positive_D[index]+negative_D[index]);
        }
    };
    SparsePolynomial A1,A2;
    std::array<SparsePolynomial,4> D1,D2;
    antipodal(reference,A1,D1);
    antipodal(2.0*reference,A2,D2);
    ReconstructedAsymptotic result;
    result.deterministic=
        (1.0/3.0)*(4.0*A2-A1);
    for (std::size_t index=0;
         index<result.stochastic.size();
         ++index) {
        result.stochastic[index]=
            (1.0/3.0)*(4.0*D2[index]-D1[index]);
    }
    return result;
}

ReconstructedAsymptotic reconstructed_subleading_coefficient(
    const ReconstructedStochasticKernelSystem& kernels,
    const Vec3& external,
    const Vec3& unit_direction,
    double reference) {
    const auto values=[&](
        double radius,
        SparsePolynomial& deterministic,
        std::array<SparsePolynomial,4>& stochastic) {
        const Vec3 positive=
            scale(unit_direction,radius);
        const Vec3 negative=
            scale(unit_direction,-radius);
        const auto evaluate=[&](
            const Vec3& q,
            SparsePolynomial& A,
            std::array<SparsePolynomial,4>& D) {
            A=kernels.deterministic(
                {external,q,scale(q,-1.0)});
            D=kernels.stochastic(
                {external,q,scale(q,-1.0)});
        };
        SparsePolynomial positive_A;
        SparsePolynomial negative_A;
        std::array<SparsePolynomial,4> positive_D;
        std::array<SparsePolynomial,4> negative_D;
        evaluate(positive,positive_A,positive_D);
        evaluate(negative,negative_A,negative_D);
        deterministic=0.5*(positive_A+negative_A);
        for (std::size_t index=0;
             index<stochastic.size();
             ++index) {
            stochastic[index]=
                0.5*(positive_D[index]+negative_D[index]);
        }
    };
    SparsePolynomial A1,A2,A4;
    std::array<SparsePolynomial,4> D1,D2,D4;
    values(reference,A1,D1);
    values(2.0*reference,A2,D2);
    values(4.0*reference,A4,D4);
    const auto coefficient=[reference](
        const SparsePolynomial& value1,
        const SparsePolynomial& value2,
        const SparsePolynomial& value4) {
        const SparsePolynomial estimate_q=
            (4.0/3.0)*reference*reference
            *(value1-value2);
        const SparsePolynomial estimate_2q=
            (16.0/3.0)*reference*reference
            *(value2-value4);
        return (1.0/3.0)*
            (4.0*estimate_2q-estimate_q);
    };
    ReconstructedAsymptotic result;
    result.deterministic=coefficient(A1,A2,A4);
    for (std::size_t index=0;
         index<result.stochastic.size();
         ++index) {
        result.stochastic[index]=
            coefficient(D1[index],D2[index],D4[index]);
    }
    return result;
}

MixedStochasticPowerTemplates
reconstructed_mixed_cross_power_impl(
    const PowerSpectrum& power,
    const Vec3& external,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& config,
    ReconstructedStochasticMap map) {
    const double k=marisa_b_halo_v1::norm(external);
    if (!(k>0.0 && reconstruction.enabled
          &&config.qmin>0.0
          &&config.qmax>config.qmin
          &&config.n_radial>0
          &&config.n_mu>0
          &&config.n_phi>0
          &&config.mu_ren>0.0
          &&config.asymptotic_factor>1.0)) {
        throw std::invalid_argument(
            "invalid reconstructed mixed stochastic configuration");
    }
    const ReconstructedStochasticKernelSystem kernels(
        base_provider,reconstruction,map);
    const SparsePolynomial K1=
        kernels.deterministic({external});
    const double Pk=power(k);
    const SoftRegionMapper mapper(
        {Vec3{},external},config.qmax);
    const double reference=
        config.asymptotic_factor
        *std::max(k,config.mu_ren);
    const QuadratureRule mu=
        gauss_legendre(config.n_mu,-1.0,1.0);
    const QuadratureRule phi=
        gauss_legendre(config.n_phi,0.0,2.0*kPi);
    const double measure=
        1.0/(8.0*kPi*kPi*kPi);
    MixedStochasticPowerTemplates result;
    ReconstructedAsymptotic averaged_subleading;
    for (int imu=0;imu<config.n_mu;++imu) {
        const double cosine=
            mu.nodes[static_cast<std::size_t>(imu)];
        const double transverse=
            std::sqrt(
                std::max(0.0,1.0-cosine*cosine));
        for (int iphi=0;
             iphi<config.n_phi;
             ++iphi) {
            const double azimuth=
                phi.nodes[static_cast<std::size_t>(iphi)];
            const Vec3 direction{
                transverse*std::cos(azimuth),
                transverse*std::sin(azimuth),
                cosine};
            const double angular_weight=
                mu.weights[static_cast<std::size_t>(imu)]
                *phi.weights[static_cast<std::size_t>(iphi)];
            const ReconstructedAsymptotic asymptotic=
                reconstructed_asymptotic_limit(
                    kernels,external,direction,reference);
            const ReconstructedAsymptotic zero_lag=
                reconstructed_p22_zero_lag_limit(
                    kernels,external,direction,reference);

            std::vector<double> boundaries={
                config.qmin,config.qmax};
            for (double value:
                 mapper.radial_boundaries(direction)) {
                if (value>config.qmin) {
                    boundaries.push_back(value);
                }
            }
            std::sort(
                boundaries.begin(),boundaries.end());
            boundaries.erase(
                std::unique(
                    boundaries.begin(),boundaries.end(),
                    [](double left,double right) {
                        return std::fabs(left-right)
                            <=1.0e-12
                              *std::max({
                                  left,right,1.0});
                    }),
                boundaries.end());
            for (std::size_t segment=0;
                 segment+1<boundaries.size();
                 ++segment) {
                const QuadratureRule radial=
                    gauss_legendre(
                        config.n_radial,
                        std::log(boundaries[segment]),
                        std::log(boundaries[segment+1]));
                for (std::size_t iq=0;
                     iq<radial.nodes.size();
                     ++iq) {
                    const double radius=
                        std::exp(radial.nodes[iq]);
                    const Vec3 mapped=
                        scale(direction,radius);
                    const CrossIntegrand bare22=
                        mapper.map(
                            mapped,[&](const Vec3& q) {
                                const Vec3 shifted=
                                    marisa_b_halo_v1::
                                        subtract(
                                            external,q);
                                const SparsePolynomial KA=
                                    kernels.deterministic(
                                        {q,shifted});
                                const auto KD=
                                    kernels.stochastic(
                                        {q,shifted});
                                CrossIntegrand value;
                                const double prefactor=
                                    2.0
                                    *power(
                                        marisa_b_halo_v1::
                                            norm(q))
                                    *power(
                                        marisa_b_halo_v1::
                                            norm(shifted));
                                for (std::size_t index=0;
                                     index<4;
                                     ++index) {
                                    value.values[index]=
                                        prefactor
                                        *KD[index]*KA;
                                }
                                return value;
                            });
                    CrossIntegrand subtraction22;
                    CrossIntegrand bare13;
                    CrossIntegrand subtraction13;
                    if (radius<=config.qmax) {
                        const Vec3 minus_q=
                            scale(mapped,-1.0);
                        const SparsePolynomial KA3=
                            kernels.deterministic(
                                {external,mapped,minus_q});
                        const auto KD3=
                            kernels.stochastic(
                                {external,mapped,minus_q});
                        const double Pq=power(radius);
                        const double prefactor22=
                            2.0*Pq*Pq;
                        const double prefactor13=
                            3.0*Pk*Pq;
                        for (std::size_t index=0;
                             index<4;
                             ++index) {
                            subtraction22.values[index]=
                                prefactor22
                                *zero_lag.stochastic[index]
                                *zero_lag.deterministic;
                            bare13.values[index]=
                                prefactor13
                                *KD3[index]*K1;
                            subtraction13.values[index]=
                                prefactor13
                                *asymptotic
                                    .stochastic[index]
                                *K1;
                        }
                        bare13.values[0]+=
                            prefactor13*KA3;
                        subtraction13.values[0]+=
                            prefactor13
                            *asymptotic.deterministic;
                    }
                    const double weight=
                        radial.weights[iq]
                        *radius*radius*radius
                        *angular_weight*measure;
                    for (std::size_t index=0;
                         index<4;
                         ++index) {
                        result.P22_bare[index]+=
                            weight*bare22.values[index];
                        result.P13_bare[index]+=
                            weight*bare13.values[index];
                        result
                            .P22_zero_lag_subtraction[index]+=
                            weight
                            *subtraction22.values[index];
                        result
                            .P13_bias_subtraction[index]+=
                            weight
                            *subtraction13.values[index];
                        result.P22[index]+=
                            weight*(
                                bare22.values[index]
                                -subtraction22.values[index]);
                        result.P13[index]+=
                            weight*(
                                bare13.values[index]
                                -subtraction13.values[index]);
                    }
                    ++result.integration_nodes;
                }
            }
            if (config.restore_p13_uv_tail) {
                const ReconstructedAsymptotic subleading=
                    reconstructed_subleading_coefficient(
                        kernels,external,direction,
                        reference);
                const double normalized_weight=
                    angular_weight/(4.0*kPi);
                averaged_subleading.deterministic+=
                    normalized_weight
                    *subleading.deterministic;
                for (std::size_t index=0;
                     index<4;
                     ++index) {
                    averaged_subleading.stochastic[index]+=
                        normalized_weight
                        *subleading.stochastic[index];
                }
            }
        }
    }
    if (config.restore_p13_uv_tail) {
        const double moment=
            normalized_linear_power_uv_tail_moment(
                power,config.qmax,
                config.uv_tail_kmax,
                config.uv_tail_quadrature_order);
        const double prefactor=3.0*Pk*moment;
        for (std::size_t index=0;
             index<4;
             ++index) {
            result.P13_uv_tail_restoration[index]=
                prefactor
                *averaged_subleading.stochastic[index]
                *K1;
            result.P13[index]+=
                result.P13_uv_tail_restoration[index];
        }
        result.P13_uv_tail_restoration[0]+=
            prefactor
            *averaged_subleading.deterministic;
        result.P13[0]+=
            prefactor
            *averaged_subleading.deterministic;
    }
    for (std::size_t index=0;
         index<4;
         ++index) {
        result.value[index]=
            result.P22[index]+result.P13[index];
    }
    return result;
}

}  // namespace

SparsePolynomial reconstructed_leading_bshot_residual_generating_shape(
    const PowerSpectrum& tree_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    bool stochastic_shift) {
    return reconstructed_leading_bshot_residual_generating_shape_impl(
        tree_power,closed_triangle,base_provider,
        reconstruction,stochastic_shift);
}

const std::array<ParameterId,kStochasticCount>& stochastic_parameter_ids() {
    static const std::array<ParameterId,kStochasticCount> ids={{
        ParameterId::Pshot,ParameterId::A0Power,
        ParameterId::AshotResidual,ParameterId::A1Pure,
        ParameterId::BshotResidual,ParameterId::D2,
        ParameterId::DG2,ParameterId::DGamma3,
        ParameterId::Abar0Mixed,ParameterId::A3Mixed,
        ParameterId::A4Mixed,ParameterId::A5Mixed}};
    return ids;
}

MixedStochasticPowerTemplates mixed_stochastic_cross_power(
    const PowerSpectrum& linear_power,double k,
    const StochasticIntegrationConfig& integration) {
    return mixed_cross_power_impl(linear_power,k,integration);
}

MixedStochasticPowerTemplates
reconstructed_mixed_stochastic_cross_power(
    const PowerSpectrum& linear_power,
    const Vec3& external,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& integration,
    ReconstructedStochasticMap map) {
    return reconstructed_mixed_cross_power_impl(
        linear_power,external,base_provider,
        reconstruction,integration,map);
}

std::array<double,kStochasticCount> StochasticTemplates::linear_shapes(
    const std::array<double,kParameterCount>& parameter_values,
    double number_density) const {
    if (!(number_density>0.0 && std::isfinite(number_density))) {
        throw std::invalid_argument("stochastic number density must be positive and finite");
    }
    std::array<double,kStochasticCount> result={};
    for (std::size_t index=0;index<result.size();++index) {
        result[index]=raw[index].evaluate(parameter_values);
        if (index==2 || index==3) result[index]/=(number_density*number_density);
        else result[index]/=number_density;
    }
    result[4]+=parameter_values[static_cast<std::size_t>(ParameterId::BNabla2Delta)]
               *bshot_bnabla2_cross.evaluate(parameter_values)/number_density;
    return result;
}

double StochasticTemplates::evaluate(
    const std::array<double,kParameterCount>& parameter_values,
    double number_density) const {
    const auto shapes=linear_shapes(parameter_values,number_density);
    double result=0.0;
    for (std::size_t index=0;index<shapes.size();++index) {
        result+=parameter_values[static_cast<std::size_t>(
                    stochastic_parameter_ids()[index])]*shapes[index];
    }
    return result;
}

StochasticTemplates stochastic_bispectrum(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    double k_nl,
    const StochasticIntegrationConfig& integration,
    const std::array<MixedStochasticPowerTemplates,3>* precomputed_cross_power,
    const PowerSpectrum* leading_tree_power) {
    if (!(k_nl>0.0 && std::isfinite(k_nl))) {
        throw std::invalid_argument("stochastic k_NL must be positive and finite");
    }
    const Vec3 closure=marisa_b_halo_v1::add(
        marisa_b_halo_v1::add(closed_triangle[0],closed_triangle[1]),closed_triangle[2]);
    if (marisa_b_halo_v1::norm(closure)>1.0e-12) {
        throw std::invalid_argument("stochastic bispectrum triangle does not close");
    }
    const SparsePolynomial b1=SparsePolynomial::variable(ParameterId::B1);
    std::array<double,3> k={};
    const PowerSpectrum& tree_power=leading_tree_power?*leading_tree_power:loop_power;
    std::array<double,3> P_loop={};
    std::array<double,3> P_tree={};
    for (int index=0;index<3;++index) {
        k[index]=marisa_b_halo_v1::norm(closed_triangle[index]);
        P_loop[index]=loop_power(k[index]);
        P_tree[index]=tree_power(k[index]);
    }
    StochasticTemplates result;
    result.pure[2]=SparsePolynomial::constant(1.0);
    result.pure[3]=SparsePolynomial::constant(
        (k[0]*k[0]+k[1]*k[1]+k[2]*k[2])/(k_nl*k_nl));
    result.mixed_tree[4]=(P_tree[0]+P_tree[1]+P_tree[2])*(b1*b1);
    for (int leg=0;leg<3;++leg) {
        const auto cross=precomputed_cross_power
            ?(*precomputed_cross_power)[leg]
            :mixed_cross_power_impl(loop_power,k[leg],integration);
        result.mixed_one_loop[4]+=b1*cross[0];
        result.mixed_one_loop[5]+=2.0*cross[1];
        result.mixed_one_loop[6]+=2.0*cross[2];
        result.mixed_one_loop[7]+=2.0*cross[3];
    }
    for (int leg=0;leg<3;++leg) {
        const int second=(leg+1)%3;
        const int third=(leg+2)%3;
        const double k1sq=k[leg]*k[leg];
        const double k2sq=k[second]*k[second];
        const double k3sq=k[third]*k[third];
        const double difference=k2sq-k3sq;
        const SparsePolynomial common=b1*P_loop[leg];
        result.derivative_mixed[8]+=common*(-(k1sq*(k2sq+k3sq)-difference*difference)
                                  /(2.0*k1sq*k_nl*k_nl));
        result.derivative_mixed[9]+=common*(-(k1sq*k1sq+difference*difference)
                                  /(2.0*k1sq*k_nl*k_nl));
        result.derivative_mixed[10]+=common*(-k1sq/(k_nl*k_nl));
        result.derivative_mixed[11]+=common*(-(k2sq+k3sq)/(k_nl*k_nl));
    }
    // COBRA Eq. (62): one term per external leg,
    // -(b1 Bshot) b_{nabla^2 delta} k_i^2 P11(k_i)/(nbar k_NL^2).
    // The displayed "+ 2 cyc." supplies the other two legs; it is not an
    // additional Wick-contraction factor of two.
    result.bshot_bnabla2_cross=
        -(P_loop[0]*k[0]*k[0]+P_loop[1]*k[1]*k[1]+P_loop[2]*k[2]*k[2])
        /(k_nl*k_nl)*b1;
    for (std::size_t index=0;index<kStochasticCount;++index) {
        result.raw[index]=result.pure[index]+result.mixed_tree[index]
                          +result.mixed_one_loop[index]
                          +result.derivative_mixed[index];
    }
    return result;
}

StochasticTemplates reconstructed_stochastic_bispectrum(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_nl,
    const StochasticIntegrationConfig& integration,
    const PowerSpectrum* leading_tree_power) {
    return reconstructed_stochastic_bispectrum_decomposition(
        loop_power,closed_triangle,base_provider,
        reconstruction,k_nl,integration,
        leading_tree_power).tied;
}

ReconstructedStochasticTemplates
reconstructed_stochastic_bispectrum_decomposition(
    const PowerSpectrum& loop_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_nl,
    const StochasticIntegrationConfig& integration,
    const PowerSpectrum* leading_tree_power) {
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "reconstructed stochastic bispectrum "
            "requires enabled reconstruction");
    }
    std::array<MixedStochasticPowerTemplates,3> density_cross;
    std::array<MixedStochasticPowerTemplates,3> tied_cross;
    for (std::size_t leg=0;leg<density_cross.size();++leg) {
        density_cross[leg]=
            reconstructed_mixed_stochastic_cross_power(
                loop_power,closed_triangle[leg],
                base_provider,reconstruction,
                integration,
                ReconstructedStochasticMap::DensityOnly);
        tied_cross[leg]=
            reconstructed_mixed_stochastic_cross_power(
                loop_power,closed_triangle[leg],
                base_provider,reconstruction,
                integration,
                ReconstructedStochasticMap::
                    TiedDensityAndShift);
    }
    ReconstructedStochasticTemplates result;
    result.density_only=stochastic_bispectrum(
        loop_power,closed_triangle,k_nl,
        integration,&density_cross,leading_tree_power);
    result.tied=stochastic_bispectrum(
        loop_power,closed_triangle,k_nl,
        integration,&tied_cross,leading_tree_power);
    /*
     * Replace the inherited pre-reconstruction leading mixed shape by the
     * marked-epsilon derivation.  This is only the derivative with respect to
     * the estimator-subtracted Bshot_C-1 residual coordinate.  Its K0
     * derivative vanishes.  The nonzero fixed K0=1 Poisson baseline is not a
     * residual-parameter direction and is added independently by the
     * role-resolved cumulant contractions in poisson_reconstruction.cpp.
     */
    const PowerSpectrum& tree_power=
        leading_tree_power?*leading_tree_power:loop_power;
    result.density_only.mixed_tree[4]=
        reconstructed_leading_bshot_residual_generating_shape(
            tree_power,closed_triangle,base_provider,
            reconstruction,false);
    result.tied.mixed_tree[4]=
        reconstructed_leading_bshot_residual_generating_shape(
            tree_power,closed_triangle,base_provider,
            reconstruction,true);
    result.density_only.raw[4]=
        result.density_only.pure[4]
        +result.density_only.mixed_tree[4]
        +result.density_only.mixed_one_loop[4]
        +result.density_only.derivative_mixed[4];
    result.tied.raw[4]=
        result.tied.pure[4]
        +result.tied.mixed_tree[4]
        +result.tied.mixed_one_loop[4]
        +result.tied.derivative_mixed[4];
    for (std::size_t index=0;
         index<kStochasticCount;
         ++index) {
        result.noisy_shift.pure[index]=
            result.tied.pure[index]
            -result.density_only.pure[index];
        result.noisy_shift.mixed_tree[index]=
            result.tied.mixed_tree[index]
            -result.density_only.mixed_tree[index];
        result.noisy_shift.mixed_one_loop[index]=
            result.tied.mixed_one_loop[index]
            -result.density_only.mixed_one_loop[index];
        result.noisy_shift.derivative_mixed[index]=
            result.tied.derivative_mixed[index]
            -result.density_only.derivative_mixed[index];
        result.noisy_shift.raw[index]=
            result.tied.raw[index]
            -result.density_only.raw[index];
    }
    result.noisy_shift.bshot_bnabla2_cross=
        result.tied.bshot_bnabla2_cross
        -result.density_only.bshot_bnabla2_cross;
    return result;
}

double analytic_poisson_bispectrum(
    const std::array<double,3>& tracer_tree_power,
    double number_density) {
    if (!(number_density>0.0 && std::isfinite(number_density))) {
        throw std::invalid_argument("Poisson number density must be positive and finite");
    }
    return (tracer_tree_power[0]+tracer_tree_power[1]+tracer_tree_power[2])
           /number_density+1.0/(number_density*number_density);
}

}  // namespace marisa_b_eft_v2
