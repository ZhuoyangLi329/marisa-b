#include "diagram_assembler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "PowerSpectrum.h"

namespace marisa_b_eft_v2 {
namespace {

Vec3 minus(const Vec3& value) { return marisa_b_halo_v1::negate(value); }
Vec3 vec_add(const Vec3& left,const Vec3& right) { return marisa_b_halo_v1::add(left,right); }
Vec3 vec_subtract(const Vec3& left,const Vec3& right) { return marisa_b_halo_v1::subtract(left,right); }

int pair_index(int left,int right) {
    if (left>right) std::swap(left,right);
    if (left==0 && right==1) return 0;
    if (left==0 && right==2) return 1;
    if (left==1 && right==2) return 2;
    throw std::invalid_argument("invalid external pair");
}

}  // namespace

DiagramAssembler::DiagramAssembler(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider)
    : linear_power_(linear_power),provider_(provider) {
    const marisa_b_halo_v1::CanonicalTriangle canonical=
        marisa_b_halo_v1::canonicalize_closed_vectors(closed_triangle);
    external_={{canonical.k1,canonical.k2,canonical.k3}};
    for (int index=0;index<3;++index) {
        external_power_[index]=power(marisa_b_halo_v1::norm(external_[index]));
        K1_[index]=provider_.deterministic({external_[index]});
    }
    K2_pairs_[0]=provider_.deterministic({external_[0],external_[1]});
    K2_pairs_[1]=provider_.deterministic({external_[0],external_[2]});
    K2_pairs_[2]=provider_.deterministic({external_[1],external_[2]});
}

double DiagramAssembler::power(double k) const {
    return linear_power_(std::max(k,1.0e-12));
}

SparsePolynomial DiagramAssembler::tree() const {
    SparsePolynomial result;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            result+=2.0*external_power_[left]*external_power_[right]
                    *K1_[left]*K1_[right]*K2_pairs_[pair_index(left,right)];
        }
    }
    return result;
}

SparsePolynomial DiagramAssembler::integrand(Diagram diagram,const Vec3& loop) const {
    const Vec3 minus_loop=minus(loop);
    const double Pq=power(marisa_b_halo_v1::norm(loop));
    if (diagram==Diagram::B222) {
        const Vec3 k1_minus_q=vec_subtract(external_[0],loop);
        const Vec3 k2_plus_q=vec_add(external_[1],loop);
        return 8.0*Pq*power(marisa_b_halo_v1::norm(k1_minus_q))
               *power(marisa_b_halo_v1::norm(k2_plus_q))
               *provider_.deterministic({loop,k1_minus_q})
               *provider_.deterministic({minus_loop,k2_plus_q})
               *provider_.deterministic({minus(k1_minus_q),minus(k2_plus_q)});
    }
    if (diagram==Diagram::B321I) {
        SparsePolynomial result;
        for (int j=0;j<3;++j) {
            for (int i=0;i<3;++i) {
                if (i==j) continue;
                result+=b321i_route(i,j,loop);
            }
        }
        return result;
    }
    if (diagram==Diagram::B321II) {
        return Pq*tadpole_coefficient(diagram,loop);
    }
    if (diagram==Diagram::B411) {
        return Pq*tadpole_coefficient(diagram,loop);
    }
    throw std::invalid_argument("unknown one-loop diagram");
}

SparsePolynomial DiagramAssembler::b321i_route(
    int external_power_index,int shifted_power_index,
    const Vec3& loop) const {
    if (external_power_index<0 || external_power_index>=3
        || shifted_power_index<0 || shifted_power_index>=3
        || external_power_index==shifted_power_index) {
        throw std::invalid_argument("invalid B321I route indices");
    }
    const Vec3 shifted=vec_subtract(
        external_[shifted_power_index],loop);
    const double Pq=power(marisa_b_halo_v1::norm(loop));
    const double shifted_power=power(marisa_b_halo_v1::norm(shifted));
    return 6.0*external_power_[external_power_index]*Pq*shifted_power
           *K1_[external_power_index]
           *provider_.deterministic({loop,shifted})
           *provider_.deterministic(
               {external_[external_power_index],loop,shifted});
}

SparsePolynomial DiagramAssembler::tadpole_coefficient(
    Diagram diagram,const Vec3& loop) const {
    if (diagram==Diagram::B321II) {
        SparsePolynomial result;
        for (int left=0;left<3;++left) {
            for (int right=left+1;right<3;++right) {
                result+=b321ii_route(left,right,right,loop);
                result+=b321ii_route(left,right,left,loop);
            }
        }
        return result;
    }
    if (diagram!=Diagram::B411) {
        throw std::invalid_argument("only B321II and B411 have tadpole coefficients");
    }
    SparsePolynomial result;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            result+=b411_route(left,right,loop);
        }
    }
    return result;
}

SparsePolynomial DiagramAssembler::b321ii_route(
    int left,int right,int tadpole_external,const Vec3& loop) const {
    if (left<0 || left>=3 || right<0 || right>=3 || left>=right
        || (tadpole_external!=left && tadpole_external!=right)) {
        throw std::invalid_argument("invalid B321II route indices");
    }
    const int linear_external=tadpole_external==left?right:left;
    return 6.0*external_power_[left]*external_power_[right]
           *K2_pairs_[pair_index(left,right)]*K1_[linear_external]
           *provider_.deterministic(
               {external_[tadpole_external],loop,minus(loop)});
}

SparsePolynomial DiagramAssembler::b411_route(
    int left,int right,const Vec3& loop) const {
    if (left<0 || left>=3 || right<0 || right>=3 || left>=right) {
        throw std::invalid_argument("invalid B411 route indices");
    }
    return 12.0*external_power_[left]*external_power_[right]
           *K1_[left]*K1_[right]
           *provider_.deterministic(
               {external_[left],external_[right],loop,minus(loop)});
}

double DiagramAssembler::loop_power(const Vec3& loop) const {
    return power(marisa_b_halo_v1::norm(loop));
}

DiagramIntegrands DiagramAssembler::integrands(const Vec3& loop) const {
    DiagramIntegrands result;
    result.B222=integrand(Diagram::B222,loop);
    result.B321I=integrand(Diagram::B321I,loop);
    result.B321II=integrand(Diagram::B321II,loop);
    result.B411=integrand(Diagram::B411,loop);
    result.total=result.B222+result.B321I+result.B321II+result.B411;
    return result;
}

std::vector<Vec3> DiagramAssembler::soft_centers(Diagram diagram) const {
    const Vec3 zero{};
    if (diagram==Diagram::B222) return {zero,external_[0],minus(external_[1])};
    if (diagram==Diagram::B321I) return {zero,external_[0],external_[1],external_[2]};
    return {zero};
}

}  // namespace marisa_b_eft_v2
