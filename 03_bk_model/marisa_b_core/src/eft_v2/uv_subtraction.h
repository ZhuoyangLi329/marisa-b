#ifndef MARISA_B_EFT_V2_UV_SUBTRACTION_H
#define MARISA_B_EFT_V2_UV_SUBTRACTION_H

#include <map>
#include <string>

#include "diagram_assembler.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

struct UvSubtractionConfig {
    double mu_ren=0.30;
    double asymptotic_factor=64.0;
    int richardson_order=2;
};

struct UvSubtractionComponents {
    SparsePolynomial B321II;
    SparsePolynomial B411;
    SparsePolynomial total;
};

SparsePolynomial p13_uv_coefficient(
    const FieldKernelProvider& provider);

double normalized_linear_power_uv_tail_moment(
    const PowerSpectrum& power,
    double lower,
    double sampled_upper=80.0,
    int quadrature_order=128);

class UvSubtraction {
public:
    UvSubtraction(const DiagramAssembler& assembler,UvSubtractionConfig config);
    UvSubtractionComponents asymptotic_coefficients(const Vec3& unit_direction) const;
    UvSubtractionComponents subleading_coefficients(
        const Vec3& unit_direction) const;
    UvSubtractionComponents angular_averaged_subleading_coefficients(
        int n_mu,int n_phi) const;
    UvSubtractionComponents integrands(const Vec3& loop) const;
    double reference_q() const noexcept { return reference_q_; }
    const UvSubtractionConfig& config() const noexcept { return config_; }

private:
    std::string direction_key(const Vec3& direction) const;
    UvSubtractionComponents compute_asymptotic(const Vec3& direction) const;
    const DiagramAssembler& assembler_;
    UvSubtractionConfig config_;
    double reference_q_=0.0;
    mutable std::map<std::string,UvSubtractionComponents> cache_;
};

}  // namespace marisa_b_eft_v2

#endif
