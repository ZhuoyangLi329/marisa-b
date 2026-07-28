#ifndef MARISA_B_EFT_V2_COUNTERTERMS_H
#define MARISA_B_EFT_V2_COUNTERTERMS_H

#include <array>

#include "bias_operators.h"
#include "field_kernel_provider.h"

namespace marisa_b_eft_v2 {

constexpr std::size_t kCountertermCount=5;

using CountertermKernel=std::array<double,kCountertermCount>;

struct CountertermTemplates {
    std::array<SparsePolynomial,kCountertermCount> BctrI;
    std::array<SparsePolynomial,kCountertermCount> BctrII;
    std::array<SparsePolynomial,kCountertermCount> total;

    std::array<double,kCountertermCount> linear_shapes(
        const std::array<double,kParameterCount>& bias_values) const;
    double evaluate(const std::array<double,kParameterCount>& parameter_values) const;
};

const std::array<ParameterId,kCountertermCount>& counterterm_parameter_ids();
CountertermKernel counterterm_K1(const Vec3& momentum,double k_nl=0.30);
CountertermKernel counterterm_K2(
    const Vec3& first,const Vec3& second,double k_nl=0.30);

CountertermTemplates counterterm_bispectrum(
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    double k_nl=0.30);

CountertermTemplates reconstructed_counterterm_bispectrum(
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    double k_nl=0.30);

}  // namespace marisa_b_eft_v2

#endif
