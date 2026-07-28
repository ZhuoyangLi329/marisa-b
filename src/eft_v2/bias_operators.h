#ifndef MARISA_B_EFT_V2_BIAS_OPERATORS_H
#define MARISA_B_EFT_V2_BIAS_OPERATORS_H

#include <array>
#include <vector>

#include "kernel_primitives.h"
#include "template_algebra.h"

namespace marisa_b_eft_v2 {

using BiasValues = std::array<double, kBiasParameterCount>;

SparsePolynomial deterministic_kernel_slow(const std::vector<Vec3>& momenta);
SparsePolynomial deterministic_kernel(const std::vector<Vec3>& momenta);
double evaluate_bias_kernel(const SparsePolynomial& kernel, const BiasValues& values);

double tree_bispectrum(
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    const BiasValues& values);

}  // namespace marisa_b_eft_v2

#endif
