#include "counterterms.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace marisa_b_eft_v2 {
namespace {

double square(const Vec3& value) {
    return marisa_b_halo_v1::dot(value,value);
}

void validate_scale(double k_nl) {
    if (!(k_nl>0.0 && std::isfinite(k_nl))) {
        throw std::invalid_argument("counterterm k_NL must be positive and finite");
    }
}

void validate_triangle(
    const std::array<Vec3,3>& closed_triangle) {
    const Vec3 closure=marisa_b_halo_v1::add(
        marisa_b_halo_v1::add(
            closed_triangle[0],closed_triangle[1]),
        closed_triangle[2]);
    const double scale=std::max({
        marisa_b_halo_v1::norm(closed_triangle[0]),
        marisa_b_halo_v1::norm(closed_triangle[1]),
        marisa_b_halo_v1::norm(closed_triangle[2]),1.0});
    if (marisa_b_halo_v1::norm(closure)>1.0e-12*scale) {
        throw std::invalid_argument(
            "counterterm bispectrum triangle does not close");
    }
}

class CountertermDirectionProvider final:
    public FieldKernelProvider {
public:
    CountertermDirectionProvider(
        std::size_t index,double k_nl)
        :index_(index),k_nl_(k_nl) {
        if (index_>=kCountertermCount) {
            throw std::out_of_range(
                "counterterm direction index out of range");
        }
        validate_scale(k_nl_);
    }

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override {
        double coefficient=0.0;
        if (momenta.size()==1) {
            coefficient=counterterm_K1(
                momenta[0],k_nl_)[index_];
        } else if (momenta.size()==2) {
            coefficient=counterterm_K2(
                momenta[0],momenta[1],k_nl_)[index_];
        }
        return SparsePolynomial::constant(coefficient);
    }

    std::string_view name() const noexcept override {
        return "eft_v2_counterterm_direction";
    }

private:
    std::size_t index_;
    double k_nl_;
};

}  // namespace

const std::array<ParameterId,kCountertermCount>& counterterm_parameter_ids() {
    static const std::array<ParameterId,kCountertermCount> ids={{
        ParameterId::BNabla2Delta,
        ParameterId::BNabla2Delta2,
        ParameterId::BNabla2G2,
        ParameterId::BGradDelta2,
        ParameterId::BGradT2}};
    return ids;
}

std::array<double,kCountertermCount> CountertermTemplates::linear_shapes(
    const std::array<double,kParameterCount>& bias_values) const {
    std::array<double,kCountertermCount> result={};
    for (std::size_t index=0;index<result.size();++index) {
        result[index]=total[index].evaluate(bias_values);
    }
    return result;
}

double CountertermTemplates::evaluate(
    const std::array<double,kParameterCount>& parameter_values) const {
    const auto shapes=linear_shapes(parameter_values);
    double result=0.0;
    for (std::size_t index=0;index<shapes.size();++index) {
        result+=parameter_values[static_cast<std::size_t>(
                    counterterm_parameter_ids()[index])]*shapes[index];
    }
    return result;
}

CountertermKernel counterterm_K1(const Vec3& momentum,double k_nl) {
    validate_scale(k_nl);
    CountertermKernel result={};
    result[0]=-square(momentum)/(k_nl*k_nl);
    return result;
}

CountertermKernel counterterm_K2(
    const Vec3& first,const Vec3& second,double k_nl) {
    validate_scale(k_nl);
    const Vec3 total=marisa_b_halo_v1::add(first,second);
    const double output_square=square(total);
    const double dot=marisa_b_halo_v1::dot(first,second);
    const double tidal=kappa(first,second);
    const double inverse_scale=1.0/(k_nl*k_nl);
    CountertermKernel result={};
    result[0]=-inverse_scale*output_square*spt_F({first,second});
    result[1]=-inverse_scale*output_square;
    result[2]=-inverse_scale*output_square*tidal;
    result[3]=-inverse_scale*dot;
    result[4]=-inverse_scale*dot*(1.0+tidal);
    return result;
}

CountertermTemplates counterterm_bispectrum(
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    double k_nl) {
    validate_scale(k_nl);
    validate_triangle(closed_triangle);
    std::array<SparsePolynomial,3> K1;
    std::array<CountertermKernel,3> K1ctr;
    for (int index=0;index<3;++index) {
        K1[index]=deterministic_kernel({closed_triangle[index]});
        K1ctr[index]=counterterm_K1(closed_triangle[index],k_nl);
    }
    CountertermTemplates result;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            const SparsePolynomial K2=deterministic_kernel(
                {closed_triangle[left],closed_triangle[right]});
            const CountertermKernel K2ctr=counterterm_K2(
                closed_triangle[left],closed_triangle[right],k_nl);
            const double powers=linear_power[left]*linear_power[right];
            for (std::size_t index=0;index<kCountertermCount;++index) {
                result.BctrI[index]+=2.0*powers*K2ctr[index]*K1[left]*K1[right];
                result.BctrII[index]+=powers*(
                    K1ctr[left][index]*K1[right]*K2
                    +K1ctr[right][index]*K1[left]*K2);
            }
        }
    }
    for (std::size_t index=0;index<kCountertermCount;++index) {
        result.total[index]=result.BctrI[index]+result.BctrII[index];
    }
    return result;
}

CountertermTemplates reconstructed_counterterm_bispectrum(
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    double k_nl) {
    validate_scale(k_nl);
    validate_triangle(closed_triangle);

    const ReconstructedFieldKernelProvider reconstructed(
        base,reconstruction);
    std::array<SparsePolynomial,3> K1;
    for (int leg=0;leg<3;++leg) {
        K1[leg]=reconstructed.deterministic(
            {closed_triangle[leg]});
    }

    CountertermTemplates result;
    for (std::size_t counterterm=0;
         counterterm<kCountertermCount;++counterterm) {
        const CountertermDirectionProvider direction(
            counterterm,k_nl);
        std::array<SparsePolynomial,3> K1ctr;
        for (int leg=0;leg<3;++leg) {
            K1ctr[leg]=reconstructed_field_kernel_variation(
                base,direction,reconstruction,
                {closed_triangle[leg]}).direction;
        }
        for (int left=0;left<3;++left) {
            for (int right=left+1;right<3;++right) {
                const FieldKernelVariation K2=
                    reconstructed_field_kernel_variation(
                        base,direction,reconstruction,
                        {closed_triangle[left],
                         closed_triangle[right]});
                const double powers=
                    linear_power[left]*linear_power[right];
                /*
                 * Bakx et al. Eq. (36): the quadratic counterterm
                 * insertion carries the tree Wick factor two, while the
                 * two ordered linear-counterterm insertions are written
                 * with unit coefficient ("+5 perms").
                 */
                result.BctrI[counterterm]+=
                    2.0*powers*K2.direction
                    *K1[left]*K1[right];
                result.BctrII[counterterm]+=
                    powers*(
                        K1ctr[left]*K1[right]*K2.value
                        +K1ctr[right]*K1[left]*K2.value);
            }
        }
        result.total[counterterm]=
            result.BctrI[counterterm]
            +result.BctrII[counterterm];
    }
    return result;
}

}  // namespace marisa_b_eft_v2
