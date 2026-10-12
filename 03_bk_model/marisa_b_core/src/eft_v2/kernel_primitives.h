#ifndef MARISA_B_EFT_V2_KERNEL_PRIMITIVES_H
#define MARISA_B_EFT_V2_KERNEL_PRIMITIVES_H

#include <vector>

#include "halo_v1.h"

namespace marisa_b_eft_v2 {

using Vec3 = marisa_b_halo_v1::Vec3;

struct SptKernelPair {
    double F = 0.0;
    double G = 0.0;
};

double cosine(const Vec3& left, const Vec3& right);
double kappa(const Vec3& left, const Vec3& right);
double galileon_L(const Vec3& first, const Vec3& second, const Vec3& third);
double galileon_M(
    const Vec3& first,
    const Vec3& second,
    const Vec3& third,
    const Vec3& fourth);
Vec3 sum(const std::vector<Vec3>& vectors);
SptKernelPair spt_kernels(const std::vector<Vec3>& momenta);
double spt_F(const std::vector<Vec3>& momenta);
double spt_G(const std::vector<Vec3>& momenta);

}  // namespace marisa_b_eft_v2

#endif
