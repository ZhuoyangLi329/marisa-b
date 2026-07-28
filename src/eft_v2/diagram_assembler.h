#ifndef MARISA_B_EFT_V2_DIAGRAM_ASSEMBLER_H
#define MARISA_B_EFT_V2_DIAGRAM_ASSEMBLER_H

#include <array>
#include <vector>

#include "field_kernel_provider.h"

class PowerSpectrum;

namespace marisa_b_eft_v2 {

enum class Diagram {
    B222,
    B321I,
    B321II,
    B411,
};

struct DiagramIntegrands {
    SparsePolynomial B222;
    SparsePolynomial B321I;
    SparsePolynomial B321II;
    SparsePolynomial B411;
    SparsePolynomial total;
};

class DiagramAssembler {
public:
    DiagramAssembler(
        const PowerSpectrum& linear_power,
        const std::array<Vec3,3>& closed_triangle,
        const FieldKernelProvider& provider);

    const std::array<Vec3,3>& external() const noexcept { return external_; }
    SparsePolynomial tree() const;
    SparsePolynomial integrand(Diagram diagram,const Vec3& loop) const;
    SparsePolynomial b321i_route(
        int external_power_index,
        int shifted_power_index,
        const Vec3& loop) const;
    SparsePolynomial b321ii_route(
        int left_external_index,
        int right_external_index,
        int tadpole_external_index,
        const Vec3& loop) const;
    SparsePolynomial b411_route(
        int left_external_index,
        int right_external_index,
        const Vec3& loop) const;
    SparsePolynomial tadpole_coefficient(Diagram diagram,const Vec3& loop) const;
    double loop_power(const Vec3& loop) const;
    DiagramIntegrands integrands(const Vec3& loop) const;
    std::vector<Vec3> soft_centers(Diagram diagram) const;

private:
    double power(double k) const;
    const PowerSpectrum& linear_power_;
    std::array<Vec3,3> external_;
    const FieldKernelProvider& provider_;
    std::array<double,3> external_power_={};
    std::array<SparsePolynomial,3> K1_;
    std::array<SparsePolynomial,3> K2_pairs_;
};

}  // namespace marisa_b_eft_v2

#endif
