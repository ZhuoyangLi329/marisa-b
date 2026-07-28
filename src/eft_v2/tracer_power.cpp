#include "tracer_power.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "PowerSpectrum.h"
#include "ir_safe_integrands.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int count,double lower,double upper) {
    if (count<1 || !(upper>lower)) throw std::invalid_argument("invalid power quadrature");
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

void append_boundaries(
    std::vector<double>& values,const SoftRegionMapper& mapper,
    const Vec3& direction,double qmin) {
    for (double value:mapper.radial_boundaries(direction)) {
        if (value>qmin) values.push_back(value);
    }
}

void accumulate(TracerPowerTemplates& target,const TracerPowerTemplates& source,double weight) {
    target.tree+=weight*source.tree;
    target.P22+=weight*source.P22;
    target.P13+=weight*source.P13;
    target.P22_bare+=weight*source.P22_bare;
    target.P13_bare+=weight*source.P13_bare;
    target.P22_stochastic_subtraction+=weight*source.P22_stochastic_subtraction;
    target.P13_bias_subtraction+=weight*source.P13_bias_subtraction;
    target.P13_uv_tail_restoration+=
        weight*source.P13_uv_tail_restoration;
    target.counterterm_bnabla2_delta+=weight*source.counterterm_bnabla2_delta;
    target.stochastic_pshot+=weight*source.stochastic_pshot;
    target.stochastic_a0+=weight*source.stochastic_a0;
    target.integration_nodes+=source.integration_nodes;
}

SparsePolynomial p13_uv_tail_restoration(
    const PowerSpectrum& power,double k,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration) {
    if (!integration.restore_p13_uv_tail) return {};
    const double moment=normalized_linear_power_uv_tail_moment(
        power,integration.qmax,integration.uv_tail_kmax,
        integration.uv_tail_quadrature_order);
    const Vec3 external{0.0,0.0,k};
    const SparsePolynomial K1=provider.deterministic({external});
    return moment*power(k)*k*k*K1*p13_uv_coefficient(provider);
}

}  // namespace

TracerPowerTemplates tracer_power_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    double k,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration,
    double k_nl) {
    if (!(k>0.0 && integration.qmin>0.0 && integration.qmax>k
          && integration.n_radial>0 && integration.n_mu>0 && k_nl>0.0)) {
        throw std::invalid_argument("invalid tracer-power configuration");
    }
    const Vec3 external{0.0,0.0,k};
    const SparsePolynomial K1=provider.deterministic({external});
    TracerPowerTemplates result;
    result.k=k;
    result.tree=tree_power(k)*(K1*K1);
    result.counterterm_bnabla2_delta=
        (-2.0*k*k/(k_nl*k_nl))*loop_power(k)*K1;
    result.stochastic_pshot=1.0;
    result.stochastic_a0=k*k/(k_nl*k_nl);

    const SoftRegionMapper p22_mapper({Vec3{},external},integration.qmax);
    const SoftRegionMapper p13_mapper({Vec3{}},integration.qmax);
    const QuadratureRule mu=gauss_legendre(integration.n_mu,-1.0,1.0);
    const double measure=1.0/(4.0*kPi*kPi);
    const double reference=integration.uv.asymptotic_factor
                           *std::max(k,integration.uv.mu_ren);
    for (int imu=0;imu<integration.n_mu;++imu) {
        const double cosine=mu.nodes[imu];
        const Vec3 direction{
            std::sqrt(std::max(0.0,1.0-cosine*cosine)),0.0,cosine};
        const Vec3 q1=scale(direction,reference);
        const Vec3 q2=scale(direction,2.0*reference);
        const SparsePolynomial K3q1=provider.deterministic({external,q1,scale(q1,-1.0)});
        const SparsePolynomial K3q2=provider.deterministic({external,q2,scale(q2,-1.0)});
        const SparsePolynomial K3inf=(1.0/3.0)*(4.0*K3q2-K3q1);
        std::vector<double> boundaries={integration.qmin};
        append_boundaries(boundaries,p22_mapper,direction,integration.qmin);
        append_boundaries(boundaries,p13_mapper,direction,integration.qmin);
        std::sort(boundaries.begin(),boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(),boundaries.end(),[](double left,double right) {
            return std::fabs(left-right)<=1.0e-12*std::max({left,right,1.0});
        }),boundaries.end());
        for (std::size_t segment=0;segment+1<boundaries.size();++segment) {
            const QuadratureRule radial=gauss_legendre(
                integration.n_radial,std::log(boundaries[segment]),
                std::log(boundaries[segment+1]));
            for (std::size_t iq=0;iq<radial.nodes.size();++iq) {
                const double radius=std::exp(radial.nodes[iq]);
                const Vec3 mapped=scale(direction,radius);
                const SparsePolynomial bare22=p22_mapper.map(mapped,[&](const Vec3& q) {
                    const Vec3 shifted=marisa_b_halo_v1::subtract(external,q);
                    const SparsePolynomial K2=provider.deterministic({q,shifted});
                    return 2.0*loop_power(marisa_b_halo_v1::norm(q))
                           *loop_power(marisa_b_halo_v1::norm(shifted))*(K2*K2);
                });
                const SparsePolynomial bare13=p13_mapper.map(mapped,[&](const Vec3& q) {
                    const SparsePolynomial K3=provider.deterministic({external,q,scale(q,-1.0)});
                    return 6.0*loop_power(k)*loop_power(marisa_b_halo_v1::norm(q))*K1*K3;
                });
                SparsePolynomial subtraction22;
                SparsePolynomial subtraction13;
                if (radius<=integration.qmax) {
                    const Vec3 minus_mapped=scale(mapped,-1.0);
                    const SparsePolynomial zero_lag=provider.deterministic({mapped,minus_mapped});
                    const double Pq=loop_power(radius);
                    subtraction22=2.0*Pq*Pq*(zero_lag*zero_lag);
                    subtraction13=6.0*loop_power(k)*Pq*K1*K3inf;
                }
                const double weight=radial.weights[iq]*radius*radius*radius
                                    *mu.weights[imu]*measure;
                result.P22_bare+=weight*bare22;
                result.P13_bare+=weight*bare13;
                result.P22_stochastic_subtraction+=weight*subtraction22;
                result.P13_bias_subtraction+=weight*subtraction13;
                result.P22+=weight*(bare22-subtraction22);
                result.P13+=weight*(bare13-subtraction13);
                ++result.integration_nodes;
            }
        }
    }
    result.P13_uv_tail_restoration=p13_uv_tail_restoration(
        loop_power,k,provider,integration);
    result.P13+=result.P13_uv_tail_restoration;
    result.one_loop=result.P22+result.P13;
    result.spt=result.tree+result.one_loop;
    return result;
}

TracerPowerTemplates tracer_power_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    double lower,
    double upper,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell,
    double k_nl) {
    const auto radial_rule=
        marisa_b_shell_v1::make_radial_shell_rule(
            lower,upper,shell);
    const auto& nodes=radial_rule.nodes;
    if (nodes.empty()) throw std::runtime_error("power shell has no radial nodes");
    TracerPowerTemplates result;
    result.k=0.0;
    result.shell_nodes=nodes.size();
    result.shell_mode_count=
        radial_rule.lattice_mode_count;
    result.shell_unique_radius_count=
        radial_rule.lattice_unique_radius_count;
    result.shell_boundary_mode_adjustment=
        radial_rule.lattice_boundary_mode_adjustment;
    result.stochastic_pshot=0.0;
    for (const auto& node:nodes) {
        const auto value=tracer_power_templates(
            loop_power,tree_power,node.k,provider,integration,k_nl);
        result.k+=node.weight*node.k;
        accumulate(result,value,node.weight);
    }
    result.one_loop=result.P22+result.P13;
    result.spt=result.tree+result.one_loop;
    return result;
}

RenormalizedTracerP13Template renormalized_tracer_p13_template(
    const PowerSpectrum& linear_power,
    double k,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration) {
    if (!(k>0.0 && integration.qmin>0.0
          && integration.qmax>k
          && integration.n_radial>0
          && integration.n_mu>0)) {
        throw std::invalid_argument(
            "invalid renormalized tracer P13 configuration");
    }
    const Vec3 external{0.0,0.0,k};
    const SparsePolynomial K1=provider.deterministic({external});
    const SoftRegionMapper p22_mapper(
        {Vec3{},external},integration.qmax);
    const SoftRegionMapper p13_mapper(
        {Vec3{}},integration.qmax);
    const QuadratureRule mu=gauss_legendre(
        integration.n_mu,-1.0,1.0);
    const double measure=1.0/(4.0*kPi*kPi);
    const double reference=integration.uv.asymptotic_factor
                           *std::max(k,integration.uv.mu_ren);
    RenormalizedTracerP13Template result;
    result.k=k;
    for (int imu=0;imu<integration.n_mu;++imu) {
        const double cosine=mu.nodes[imu];
        const Vec3 direction{
            std::sqrt(std::max(0.0,1.0-cosine*cosine)),
            0.0,cosine};
        const Vec3 q1=scale(direction,reference);
        const Vec3 q2=scale(direction,2.0*reference);
        const SparsePolynomial K3q1=provider.deterministic(
            {external,q1,scale(q1,-1.0)});
        const SparsePolynomial K3q2=provider.deterministic(
            {external,q2,scale(q2,-1.0)});
        const SparsePolynomial K3inf=
            (1.0/3.0)*(4.0*K3q2-K3q1);
        std::vector<double> boundaries={integration.qmin};
        append_boundaries(
            boundaries,p22_mapper,direction,integration.qmin);
        append_boundaries(
            boundaries,p13_mapper,direction,integration.qmin);
        std::sort(boundaries.begin(),boundaries.end());
        boundaries.erase(
            std::unique(
                boundaries.begin(),boundaries.end(),
                [](double left,double right) {
                    return std::fabs(left-right)
                        <=1.0e-12*std::max({left,right,1.0});
                }),
            boundaries.end());
        for (std::size_t segment=0;
             segment+1<boundaries.size();++segment) {
            const QuadratureRule radial=gauss_legendre(
                integration.n_radial,
                std::log(boundaries[segment]),
                std::log(boundaries[segment+1]));
            for (std::size_t iq=0;
                 iq<radial.nodes.size();++iq) {
                const double radius=std::exp(radial.nodes[iq]);
                const Vec3 mapped=scale(direction,radius);
                const SparsePolynomial bare=p13_mapper.map(
                    mapped,[&](const Vec3& q) {
                        const SparsePolynomial K3=
                            provider.deterministic(
                                {external,q,scale(q,-1.0)});
                        return 6.0*linear_power(k)
                            *linear_power(
                                marisa_b_halo_v1::norm(q))
                            *K1*K3;
                    });
                SparsePolynomial subtraction;
                if (radius<=integration.qmax) {
                    subtraction=
                        6.0*linear_power(k)
                        *linear_power(radius)*K1*K3inf;
                }
                const double weight=
                    radial.weights[iq]*radius*radius*radius
                    *mu.weights[imu]*measure;
                result.bare+=weight*bare;
                result.bias_subtraction+=weight*subtraction;
                result.value+=weight*(bare-subtraction);
                ++result.integration_nodes;
            }
        }
    }
    result.uv_tail_restoration=p13_uv_tail_restoration(
        linear_power,k,provider,integration);
    result.value+=result.uv_tail_restoration;
    return result;
}

FactorizedB321IITemplates assemble_factorized_b321ii_templates(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const std::array<RenormalizedTracerP13Template,3>& p13_legs) {
    const marisa_b_halo_v1::CanonicalTriangle canonical=
        marisa_b_halo_v1::canonicalize_closed_vectors(closed_triangle);
    const std::array<Vec3,3> external={{
        canonical.k1,canonical.k2,canonical.k3}};
    std::array<double,3> power={};
    FactorizedB321IITemplates result;
    for (std::size_t index=0;index<external.size();++index) {
        const double k=marisa_b_halo_v1::norm(external[index]);
        power[index]=linear_power(k);
        const double scale=std::max({k,p13_legs[index].k,1.0});
        if (std::fabs(k-p13_legs[index].k)>1.0e-13*scale) {
            throw std::invalid_argument(
                "factorized B321II P13 leg has the wrong wavenumber");
        }
        result.P13_legs[index]=p13_legs[index].value;
        result.P13_bare_legs[index]=p13_legs[index].bare;
        result.P13_bias_subtraction_legs[index]=
            p13_legs[index].bias_subtraction;
        result.P13_uv_tail_restoration_legs[index]=
            p13_legs[index].uv_tail_restoration;
        result.integration_nodes+=p13_legs[index].integration_nodes;
    }
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            const SparsePolynomial K2=provider.deterministic({
                external[static_cast<std::size_t>(left)],
                external[static_cast<std::size_t>(right)]});
            const auto assemble_pair=[
                &K2,&power,left,right](
                    const std::array<SparsePolynomial,3>& legs) {
                return K2*(
                    power[static_cast<std::size_t>(left)]
                        *legs[static_cast<std::size_t>(right)]
                    +power[static_cast<std::size_t>(right)]
                        *legs[static_cast<std::size_t>(left)]);
            };
            result.value+=assemble_pair(result.P13_legs);
            result.bare+=assemble_pair(result.P13_bare_legs);
            result.bias_subtraction+=
                assemble_pair(result.P13_bias_subtraction_legs);
            result.uv_tail_restoration+=
                assemble_pair(result.P13_uv_tail_restoration_legs);
        }
    }
    return result;
}

FactorizedB321IITemplates factorized_b321ii_templates(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const TracerPowerIntegrationConfig& integration) {
    const marisa_b_halo_v1::CanonicalTriangle canonical=
        marisa_b_halo_v1::canonicalize_closed_vectors(
            closed_triangle);
    const std::array<Vec3,3> external={{
        canonical.k1,canonical.k2,canonical.k3}};
    std::array<RenormalizedTracerP13Template,3> legs;
    for (std::size_t index=0;index<external.size();++index) {
        legs[index]=renormalized_tracer_p13_template(
            linear_power,
            marisa_b_halo_v1::norm(external[index]),
            provider,integration);
    }
    return assemble_factorized_b321ii_templates(
        linear_power,closed_triangle,provider,legs);
}

}  // namespace marisa_b_eft_v2
