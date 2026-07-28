#include "direct_evaluator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "PowerSpectrum.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;
constexpr double kTwoPi=2.0*kPi;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(int count,double lower,double upper) {
    if (count<1 || !(upper>lower)) throw std::invalid_argument("invalid Gauss-Legendre rule");
    QuadratureRule result;
    result.nodes.resize(count);
    result.weights.resize(count);
    const int half=(count+1)/2;
    const double midpoint=0.5*(lower+upper);
    const double half_width=0.5*(upper-lower);
    for (int index=0;index<half;++index) {
        double root=std::cos(kPi*(static_cast<double>(index)+0.75)/(static_cast<double>(count)+0.5));
        double derivative=0.0;
        for (int iteration=0;iteration<100;++iteration) {
            double previous=1.0;
            double current=root;
            for (int order=2;order<=count;++order) {
                const double next=((2.0*order-1.0)*root*current-(order-1.0)*previous)/order;
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

QuadratureRule logarithmic_radial_rule(
    int count,double lower,double upper,const std::array<Vec3,3>& external,bool add_breaks) {
    std::vector<double> breaks={lower,upper};
    if (add_breaks) {
        for (const Vec3& value:external) {
            const double radius=marisa_b_halo_v1::norm(value);
            if (radius>lower && radius<upper) breaks.push_back(radius);
        }
    }
    std::sort(breaks.begin(),breaks.end());
    breaks.erase(std::unique(breaks.begin(),breaks.end(),[](double left,double right) {
        return std::fabs(left-right)<=1.0e-13*std::max({left,right,1.0});
    }),breaks.end());
    QuadratureRule result;
    for (std::size_t segment=0;segment+1<breaks.size();++segment) {
        QuadratureRule piece=gauss_legendre(count,std::log(breaks[segment]),std::log(breaks[segment+1]));
        for (std::size_t index=0;index<piece.nodes.size();++index) {
            const double radius=std::exp(piece.nodes[index]);
            piece.nodes[index]=radius;
            piece.weights[index]*=radius*radius*radius;
        }
        result.nodes.insert(result.nodes.end(),piece.nodes.begin(),piece.nodes.end());
        result.weights.insert(result.weights.end(),piece.weights.begin(),piece.weights.end());
    }
    return result;
}

Vec3 scale(const Vec3& value,double factor) {
    return Vec3{factor*value.x,factor*value.y,factor*value.z};
}

Vec3 cross(const Vec3& left,const Vec3& right) {
    return Vec3{
        left.y*right.z-left.z*right.y,
        left.z*right.x-left.x*right.z,
        left.x*right.y-left.y*right.x};
}

Vec3 unit(const Vec3& value) {
    const double length=marisa_b_halo_v1::norm(value);
    if (!(length>0.0)) throw std::invalid_argument("cannot normalize zero vector");
    return scale(value,1.0/length);
}

struct Frame { Vec3 x; Vec3 y; Vec3 z; };

Frame make_frame(const std::array<Vec3,3>& external) {
    const Vec3 z=unit(external[0]);
    Vec3 transverse=marisa_b_halo_v1::subtract(
        external[1],scale(z,marisa_b_halo_v1::dot(external[1],z)));
    if (marisa_b_halo_v1::norm(transverse)<1.0e-12) {
        const Vec3 axis=std::fabs(z.x)<0.8?Vec3{1,0,0}:Vec3{0,1,0};
        transverse=marisa_b_halo_v1::subtract(axis,scale(z,marisa_b_halo_v1::dot(axis,z)));
    }
    const Vec3 x=unit(transverse);
    return Frame{x,unit(cross(z,x)),z};
}

Vec3 spherical_vector(double radius,double mu,double phi,const Frame& frame) {
    const double transverse=radius*std::sqrt(std::max(0.0,1.0-mu*mu));
    return marisa_b_halo_v1::add(
        marisa_b_halo_v1::add(
            scale(frame.x,transverse*std::cos(phi)),
            scale(frame.y,transverse*std::sin(phi))),
        scale(frame.z,radius*mu));
}

std::vector<Vec3> antipodal_fibonacci_sphere(int requested_count) {
    int count=std::max(32,requested_count);
    if (count%2!=0) ++count;
    const int half=count/2;
    const double golden_angle=0.5*kTwoPi*(3.0-std::sqrt(5.0));
    std::vector<Vec3> directions;
    directions.reserve(static_cast<std::size_t>(count));
    for (int index=0;index<half;++index) {
        const double z=(index+0.5)/static_cast<double>(half);
        const double transverse=std::sqrt(std::max(0.0,1.0-z*z));
        const double phi=golden_angle*index;
        const Vec3 value{
            transverse*std::cos(phi),transverse*std::sin(phi),z};
        directions.push_back(value);
        directions.push_back(scale(value,-1.0));
    }
    return directions;
}

struct AngularNode {
    Vec3 direction;
    double weight=0.0;
};

std::vector<AngularNode> angular_nodes(
    const DirectIntegrationConfig& config,
    const std::array<Vec3,3>& external) {
    std::vector<AngularNode> result;
    if (config.angular_rule==DirectAngularRule::AntipodalFibonacci) {
        const std::vector<Vec3> directions=antipodal_fibonacci_sphere(
            2*config.n_mu*config.n_phi);
        const double weight=4.0*kPi/static_cast<double>(directions.size());
        result.reserve(directions.size());
        for (const Vec3& direction:directions) {
            result.push_back(AngularNode{direction,weight});
        }
        return result;
    }
    const QuadratureRule mu=gauss_legendre(config.n_mu,-1.0,1.0);
    const double phi_upper=config.exploit_phi_reflection?kPi:kTwoPi;
    const double phi_multiplicity=config.exploit_phi_reflection?2.0:1.0;
    const QuadratureRule phi=gauss_legendre(
        config.n_phi,0.0,phi_upper);
    const Frame frame=make_frame(external);
    result.reserve(
        static_cast<std::size_t>(config.n_mu)
        *static_cast<std::size_t>(config.n_phi));
    for (int imu=0;imu<config.n_mu;++imu) {
        for (int iphi=0;iphi<config.n_phi;++iphi) {
            result.push_back(AngularNode{
                spherical_vector(
                    1.0,mu.nodes[static_cast<std::size_t>(imu)],
                    phi.nodes[static_cast<std::size_t>(iphi)],frame),
                mu.weights[static_cast<std::size_t>(imu)]
                    *phi.weights[static_cast<std::size_t>(iphi)]
                    *phi_multiplicity});
        }
    }
    return result;
}

void accumulate(DiagramTemplates& output,const DiagramIntegrands& input,double weight) {
    output.B222+=weight*input.B222;
    output.B321I+=weight*input.B321I;
    output.B321II+=weight*input.B321II;
    output.B411+=weight*input.B411;
}

void accumulate_uv(
    DiagramTemplates& output,
    const DiagramIntegrands& bare,
    const UvSubtractionComponents& subtraction,
    double weight) {
    output.B222+=weight*bare.B222;
    output.B321I+=weight*bare.B321I;
    output.B321II_bare+=weight*bare.B321II;
    output.B411_bare+=weight*bare.B411;
    output.B321II_uv_subtraction+=weight*subtraction.B321II;
    output.B411_uv_subtraction+=weight*subtraction.B411;
    output.B321II+=weight*(bare.B321II-subtraction.B321II);
    output.B411+=weight*(bare.B411-subtraction.B411);
}

DiagramIntegrands selected_integrands(
    const DiagramAssembler& assembler,
    const IrSafeIntegrands* mapped,
    const Vec3& loop,
    DirectDiagramMask mask) {
    if (mask==kDirectAllDiagrams) {
        return mapped?mapped->all(loop):assembler.integrands(loop);
    }
    const auto evaluate=[&](Diagram diagram) {
        return mapped
            ?mapped->component(diagram,loop)
            :assembler.integrand(diagram,loop);
    };
    DiagramIntegrands result;
    if ((mask&kDirectB222)!=0) {
        result.B222=evaluate(Diagram::B222);
    }
    if ((mask&kDirectB321I)!=0) {
        result.B321I=evaluate(Diagram::B321I);
    }
    if ((mask&kDirectB321II)!=0) {
        result.B321II=evaluate(Diagram::B321II);
    }
    if ((mask&kDirectB411)!=0) {
        result.B411=evaluate(Diagram::B411);
    }
    result.total=
        result.B222+result.B321I+result.B321II+result.B411;
    return result;
}

UvSubtractionComponents selected_uv_subtraction(
    const UvSubtraction& uv,const Vec3& loop,
    DirectDiagramMask mask) {
    if ((mask&(kDirectB321II|kDirectB411))==0) {
        return {};
    }
    UvSubtractionComponents result=uv.integrands(loop);
    if ((mask&kDirectB321II)==0) {
        result.B321II=SparsePolynomial();
    }
    if ((mask&kDirectB411)==0) {
        result.B411=SparsePolynomial();
    }
    return result;
}

}  // namespace

const char* direct_angular_rule_name(DirectAngularRule rule) noexcept {
    switch (rule) {
        case DirectAngularRule::TensorGaussLegendre:
            return "tensor-gauss-legendre";
        case DirectAngularRule::AntipodalFibonacci:
            return "antipodal-fibonacci";
    }
    return "unknown";
}

DiagramTemplates evaluate_direct(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& config) {
    if (!(config.qmin>0.0 && config.qmax>config.qmin)) {
        throw std::invalid_argument("direct evaluator requires 0<qmin<qmax");
    }
    if (config.n_radial<1 || config.n_mu<1 || config.n_phi<1) {
        throw std::invalid_argument("direct evaluator quadrature orders must be positive");
    }
    if (config.exploit_phi_reflection
        &&!provider.supports_triangle_plane_reflection()) {
        throw std::invalid_argument(
            "phi-reflection reduction is not a symmetry of "
            "the selected field-kernel provider");
    }
    if (config.diagram_mask==0
        ||(config.diagram_mask&~kDirectAllDiagrams)!=0) {
        throw std::invalid_argument(
            "direct evaluator diagram mask is empty or unsupported");
    }
    if (config.restore_uv_tail
        &&(!config.uv_subtract
           ||!(config.uv_tail_kmax>config.qmax)
           ||config.uv_tail_quadrature_order<16
           ||config.uv_tail_n_mu<1
           ||config.uv_tail_n_phi<1)) {
        throw std::invalid_argument(
            "direct UV-tail restoration requires subtraction, "
            "sampled kmax>qmax, and positive quadrature orders");
    }
    DiagramAssembler assembler(linear_power,closed_triangle,provider);
    DiagramTemplates output;
    output.external=assembler.external();
    output.tree=assembler.tree();
    output.ir_safe=config.ir_safe;
    IrSafeIntegrands mapped(assembler,config.qmax);
    UvSubtraction uv(assembler,config.uv);
    const double radial_upper=config.ir_safe?mapped.mapped_radius_max():config.qmax;
    const double measure=1.0/(kTwoPi*kTwoPi*kTwoPi);
    if (config.angular_rule==DirectAngularRule::TensorGaussLegendre) {
        // Preserve the historical loop order and multiplication grouping.
        // Tadpole diagrams contain large cancelling pieces, so even a
        // mathematically equivalent reordering is not a regression-neutral
        // implementation change.
        const QuadratureRule mu=gauss_legendre(config.n_mu,-1.0,1.0);
        const double phi_upper=config.exploit_phi_reflection?kPi:kTwoPi;
        const double phi_multiplicity=config.exploit_phi_reflection?2.0:1.0;
        const QuadratureRule phi=gauss_legendre(
            config.n_phi,0.0,phi_upper);
        const Frame frame=make_frame(output.external);
        if (!config.ir_safe) {
            const QuadratureRule radial=logarithmic_radial_rule(
                config.n_radial,config.qmin,radial_upper,
                output.external,true);
            for (std::size_t iq=0;iq<radial.nodes.size();++iq) {
                for (int imu=0;imu<config.n_mu;++imu) {
                    for (int iphi=0;iphi<config.n_phi;++iphi) {
                        const Vec3 loop=spherical_vector(
                            radial.nodes[iq],mu.nodes[imu],
                            phi.nodes[iphi],frame);
                        const double weight=
                            radial.weights[iq]*mu.weights[imu]
                            *phi.weights[iphi]*phi_multiplicity*measure;
                        const DiagramIntegrands values=
                            selected_integrands(
                                assembler,nullptr,loop,
                                config.diagram_mask);
                        if (config.uv_subtract) {
                            accumulate_uv(
                                output,values,
                                selected_uv_subtraction(
                                    uv,loop,config.diagram_mask),
                                weight);
                        } else {
                            accumulate(output,values,weight);
                        }
                    }
                }
            }
            output.integration_nodes=
                radial.nodes.size()
                *static_cast<std::size_t>(config.n_mu)
                *static_cast<std::size_t>(config.n_phi);
        } else {
            for (int imu=0;imu<config.n_mu;++imu) {
                for (int iphi=0;iphi<config.n_phi;++iphi) {
                    const Vec3 direction=spherical_vector(
                        1.0,mu.nodes[imu],phi.nodes[iphi],frame);
                    const std::vector<double> boundaries=
                        mapped.radial_boundaries(
                            direction,config.qmin);
                    for (std::size_t segment=0;
                         segment+1<boundaries.size();++segment) {
                        const QuadratureRule radial=gauss_legendre(
                            config.n_radial,
                            std::log(boundaries[segment]),
                            std::log(boundaries[segment+1]));
                        for (std::size_t iq=0;
                             iq<radial.nodes.size();++iq) {
                            const double radius=
                                std::exp(radial.nodes[iq]);
                            const Vec3 loop=scale(direction,radius);
                            const double radial_weight=
                                radial.weights[iq]
                                *radius*radius*radius;
                            const double weight=
                                radial_weight*mu.weights[imu]
                                *phi.weights[iphi]
                                *phi_multiplicity*measure;
                            const DiagramIntegrands values=
                                selected_integrands(
                                    assembler,&mapped,loop,
                                    config.diagram_mask);
                            if (config.uv_subtract) {
                                UvSubtractionComponents subtraction;
                                if (radius<=config.qmax) {
                                    subtraction=
                                        selected_uv_subtraction(
                                            uv,loop,
                                            config.diagram_mask);
                                }
                                accumulate_uv(
                                    output,values,subtraction,weight);
                            } else {
                                accumulate(output,values,weight);
                            }
                            ++output.integration_nodes;
                        }
                    }
                }
            }
        }
    } else {
        const std::vector<AngularNode> angular=
            angular_nodes(config,output.external);
        QuadratureRule bare_radial;
        if (!config.ir_safe) {
            bare_radial=logarithmic_radial_rule(
                config.n_radial,config.qmin,radial_upper,
                output.external,true);
        }
        for (const AngularNode& angle:angular) {
            const std::vector<double> boundaries=config.ir_safe
                ?mapped.radial_boundaries(
                    angle.direction,config.qmin)
                :std::vector<double>{};
            const std::size_t segment_count=config.ir_safe
                ?boundaries.size()-1:1;
            for (std::size_t segment=0;
                 segment<segment_count;++segment) {
                const QuadratureRule radial=config.ir_safe
                    ?gauss_legendre(
                        config.n_radial,
                        std::log(boundaries[segment]),
                        std::log(boundaries[segment+1]))
                    :bare_radial;
                for (std::size_t iq=0;
                     iq<radial.nodes.size();++iq) {
                    const double radius=config.ir_safe
                        ?std::exp(radial.nodes[iq])
                        :radial.nodes[iq];
                    const Vec3 loop=scale(
                        angle.direction,radius);
                    const double radial_weight=config.ir_safe
                        ?radial.weights[iq]*radius*radius*radius
                        :radial.weights[iq];
                    const double weight=
                        radial_weight*angle.weight*measure;
                    const DiagramIntegrands values=
                        selected_integrands(
                            assembler,
                            config.ir_safe?&mapped:nullptr,
                            loop,config.diagram_mask);
                    if (config.uv_subtract) {
                        UvSubtractionComponents subtraction;
                        if (radius<=config.qmax) {
                            subtraction=selected_uv_subtraction(
                                uv,loop,config.diagram_mask);
                        }
                        accumulate_uv(
                            output,values,subtraction,weight);
                    } else {
                        accumulate(output,values,weight);
                    }
                    ++output.integration_nodes;
                }
            }
        }
    }
    if (!config.uv_subtract) {
        output.B321II_bare=output.B321II;
        output.B411_bare=output.B411;
    }
    if (config.restore_uv_tail) {
        const UvSubtractionComponents subleading=
            uv.angular_averaged_subleading_coefficients(
                config.uv_tail_n_mu,
                config.uv_tail_n_phi);
        const double tail_moment=
            normalized_linear_power_uv_tail_moment(
                linear_power,config.qmax,
                config.uv_tail_kmax,
                config.uv_tail_quadrature_order);
        if ((config.diagram_mask&kDirectB321II)!=0) {
            output.B321II_uv_restoration=
                tail_moment*subleading.B321II;
            output.B321II+=
                output.B321II_uv_restoration;
        }
        if ((config.diagram_mask&kDirectB411)!=0) {
            output.B411_uv_restoration=
                tail_moment*subleading.B411;
            output.B411+=
                output.B411_uv_restoration;
        }
    }
    output.uv_subtraction_total=
        output.B321II_uv_subtraction+output.B411_uv_subtraction;
    output.uv_restoration_total=
        output.B321II_uv_restoration
        +output.B411_uv_restoration;
    output.one_loop=output.B222+output.B321I+output.B321II+output.B411;
    output.total=output.tree+output.one_loop;
    return output;
}

}  // namespace marisa_b_eft_v2
