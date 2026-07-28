#include "ir_safe_integrands.h"

#include <algorithm>
#include <set>

namespace marisa_b_eft_v2 {

SoftRegionMapper::SoftRegionMapper(std::vector<Vec3> centers,double qmax)
    : centers_(std::move(centers)),qmax_(qmax) {
    if (!(qmax_>0.0)) throw std::invalid_argument("soft-region qmax must be positive");
    if (centers_.empty()) throw std::invalid_argument("soft-region center list is empty");
    for (const Vec3& center:centers_) {
        mapped_radius_max_=std::max(
            mapped_radius_max_,qmax_+marisa_b_halo_v1::norm(center));
    }
}

std::vector<double> SoftRegionMapper::radial_boundaries(const Vec3& direction) const {
    const double direction_norm=marisa_b_halo_v1::norm(direction);
    if (std::fabs(direction_norm-1.0)>1.0e-12) {
        throw std::invalid_argument("soft-region radial direction must be a unit vector");
    }
    std::vector<double> result;
    result.reserve(centers_.size());
    for (std::size_t owner=0;owner<centers_.size();++owner) {
        const Vec3& center=centers_[owner];
        const double center_square=marisa_b_halo_v1::dot(center,center);
        if (center_square>=qmax_*qmax_) {
            throw std::invalid_argument("soft center must lie strictly inside qmax sphere");
        }
        const double projection=marisa_b_halo_v1::dot(direction,center);
        const double discriminant=projection*projection+qmax_*qmax_-center_square;
        double upper=-projection+std::sqrt(std::max(0.0,discriminant));
        for (std::size_t candidate=0;candidate<centers_.size();++candidate) {
            if (candidate==owner) continue;
            const Vec3 difference=marisa_b_halo_v1::subtract(center,centers_[candidate]);
            const double directional=marisa_b_halo_v1::dot(direction,difference);
            if (directional<0.0) {
                const double boundary=-marisa_b_halo_v1::dot(difference,difference)/(2.0*directional);
                upper=std::min(upper,boundary);
            }
        }
        if (upper>0.0 && std::isfinite(upper)) result.push_back(upper);
    }
    return result;
}

IrSafeIntegrands::IrSafeIntegrands(const DiagramAssembler& assembler,double qmax)
    : assembler_(assembler),
      B222_mapper_(assembler.soft_centers(Diagram::B222),qmax),
      B321I_mapper_(assembler.soft_centers(Diagram::B321I),qmax),
      B321II_mapper_(assembler.soft_centers(Diagram::B321II),qmax),
      B411_mapper_(assembler.soft_centers(Diagram::B411),qmax) {
    mapped_radius_max_=std::max({
        B222_mapper_.mapped_radius_max(),B321I_mapper_.mapped_radius_max(),
        B321II_mapper_.mapped_radius_max(),B411_mapper_.mapped_radius_max()});
}

SparsePolynomial IrSafeIntegrands::component(Diagram diagram,const Vec3& mapped_loop) const {
    const auto evaluator=[this,diagram](const Vec3& original) {
        return assembler_.integrand(diagram,original);
    };
    if (diagram==Diagram::B222) return B222_mapper_.map(mapped_loop,evaluator);
    if (diagram==Diagram::B321I) return B321I_mapper_.map(mapped_loop,evaluator);
    if (diagram==Diagram::B321II) return B321II_mapper_.map(mapped_loop,evaluator);
    return B411_mapper_.map(mapped_loop,evaluator);
}

DiagramIntegrands IrSafeIntegrands::all(const Vec3& mapped_loop) const {
    DiagramIntegrands result;
    result.B222=component(Diagram::B222,mapped_loop);
    result.B321I=component(Diagram::B321I,mapped_loop);
    result.B321II=component(Diagram::B321II,mapped_loop);
    result.B411=component(Diagram::B411,mapped_loop);
    result.total=result.B222+result.B321I+result.B321II+result.B411;
    return result;
}

std::vector<double> IrSafeIntegrands::radial_boundaries(
    const Vec3& direction,double qmin) const {
    std::vector<double> result={qmin};
    const auto append=[&result,&direction,qmin](const SoftRegionMapper& mapper) {
        for (double boundary:mapper.radial_boundaries(direction)) {
            if (boundary>qmin) result.push_back(boundary);
        }
    };
    append(B222_mapper_);
    append(B321I_mapper_);
    append(B321II_mapper_);
    append(B411_mapper_);
    std::sort(result.begin(),result.end());
    result.erase(std::unique(result.begin(),result.end(),[](double left,double right) {
        return std::fabs(left-right)<=1.0e-12*std::max({left,right,1.0});
    }),result.end());
    return result;
}

}  // namespace marisa_b_eft_v2
