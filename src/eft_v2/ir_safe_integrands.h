#ifndef MARISA_B_EFT_V2_IR_SAFE_INTEGRANDS_H
#define MARISA_B_EFT_V2_IR_SAFE_INTEGRANDS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "diagram_assembler.h"

namespace marisa_b_eft_v2 {

class SoftRegionMapper {
public:
    SoftRegionMapper(std::vector<Vec3> centers,double qmax);
    const std::vector<Vec3>& centers() const noexcept { return centers_; }
    double mapped_radius_max() const noexcept { return mapped_radius_max_; }
    std::vector<double> radial_boundaries(const Vec3& unit_direction) const;

    template<typename Evaluator>
    auto map(const Vec3& mapped_loop,Evaluator evaluator) const
        -> decltype(evaluator(mapped_loop)) {
        using Value=decltype(evaluator(mapped_loop));
        Value result{};
        for (std::size_t center_index=0;center_index<centers_.size();++center_index) {
            const Vec3 original=marisa_b_halo_v1::add(mapped_loop,centers_[center_index]);
            const double original_radius=marisa_b_halo_v1::norm(original);
            if (!(original_radius<=qmax_)) continue;
            std::size_t owner=0;
            double best=std::numeric_limits<double>::infinity();
            for (std::size_t candidate=0;candidate<centers_.size();++candidate) {
                const double distance=marisa_b_halo_v1::norm(
                    marisa_b_halo_v1::subtract(original,centers_[candidate]));
                const double tolerance=1.0e-14*std::max(distance,1.0);
                if (!std::isfinite(best) || distance<best-tolerance) {
                    best=distance;
                    owner=candidate;
                }
            }
            if (owner==center_index) result+=evaluator(original);
        }
        return result;
    }

private:
    std::vector<Vec3> centers_;
    double qmax_=0.0;
    double mapped_radius_max_=0.0;
};

class IrSafeIntegrands {
public:
    IrSafeIntegrands(const DiagramAssembler& assembler,double qmax);
    SparsePolynomial component(Diagram diagram,const Vec3& mapped_loop) const;
    DiagramIntegrands all(const Vec3& mapped_loop) const;
    double mapped_radius_max() const noexcept { return mapped_radius_max_; }
    std::vector<double> radial_boundaries(const Vec3& unit_direction,double qmin) const;

private:
    const DiagramAssembler& assembler_;
    double mapped_radius_max_=0.0;
    SoftRegionMapper B222_mapper_;
    SoftRegionMapper B321I_mapper_;
    SoftRegionMapper B321II_mapper_;
    SoftRegionMapper B411_mapper_;
};

}  // namespace marisa_b_eft_v2

#endif
