#include "kernel_primitives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace marisa_b_eft_v2 {
namespace {

constexpr double kZeroTolerance = 1.0e-14;

int popcount(unsigned int mask) {
    int result = 0;
    while (mask != 0U) {
        result += static_cast<int>(mask & 1U);
        mask >>= 1U;
    }
    return result;
}

double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n-k) k = n-k;
    double result = 1.0;
    for (int index = 1; index <= k; ++index) {
        result *= static_cast<double>(n-k+index)/static_cast<double>(index);
    }
    return result;
}

std::vector<Vec3> select(const std::vector<Vec3>& input, unsigned int mask) {
    std::vector<Vec3> result;
    for (std::size_t index = 0; index < input.size(); ++index) {
        if ((mask & (1U << index)) != 0U) result.push_back(input[index]);
    }
    return result;
}

std::pair<int,int> exact_opposite_pair(const std::vector<Vec3>& momenta) {
    for (int left=0;left<static_cast<int>(momenta.size());++left) {
        for (int right=left+1;right<static_cast<int>(momenta.size());++right) {
            const double scale=std::max({
                marisa_b_halo_v1::norm(momenta[left]),
                marisa_b_halo_v1::norm(momenta[right]),1.0});
            if (marisa_b_halo_v1::norm(
                    marisa_b_halo_v1::add(momenta[left],momenta[right]))
                <=kZeroTolerance*scale) return {left,right};
        }
    }
    return {-1,-1};
}

struct LongVec3 {
    long double x=0.0L;
    long double y=0.0L;
    long double z=0.0L;
};

struct LongKernelPair {
    long double F=0.0L;
    long double G=0.0L;
};

LongVec3 long_vector(const Vec3& value) {
    return LongVec3{value.x,value.y,value.z};
}

LongVec3 long_add(const LongVec3& left,const LongVec3& right) {
    return LongVec3{
        left.x+right.x,left.y+right.y,left.z+right.z};
}

LongVec3 long_scale(const LongVec3& value,long double factor) {
    return LongVec3{
        factor*value.x,factor*value.y,factor*value.z};
}

long double long_dot(const LongVec3& left,const LongVec3& right) {
    return left.x*right.x+left.y*right.y+left.z*right.z;
}

long double long_norm(const LongVec3& value) {
    return std::sqrt(std::max(0.0L,long_dot(value,value)));
}

LongVec3 long_cross(const LongVec3& left,const LongVec3& right) {
    return LongVec3{
        left.y*right.z-left.z*right.y,
        left.z*right.x-left.x*right.z,
        left.x*right.y-left.y*right.x};
}

LongKernelPair regulated_fourth_order_kernel(
    const std::array<LongVec3,4>& vectors) {
    std::array<LongVec3,16> sums={};
    std::array<LongKernelPair,16> kernels={};
    for (unsigned int mask=1;mask<16;++mask) {
        for (unsigned int index=0;index<4;++index) {
            if ((mask&(1U<<index))!=0U) {
                sums[mask]=long_add(sums[mask],vectors[index]);
            }
        }
        if ((mask&(mask-1U))==0U) {
            kernels[mask]=LongKernelPair{1.0L,1.0L};
            continue;
        }
        const int order=popcount(mask);
        const long double denominator=
            static_cast<long double>((2*order+3)*(order-1));
        for (unsigned int left=(mask-1U)&mask;
             left!=0U;left=(left-1U)&mask) {
            const unsigned int right=mask^left;
            if (right==0U) continue;
            const int left_order=popcount(left);
            const long double symmetry=
                1.0L/static_cast<long double>(
                    binomial(order,left_order));
            const long double left_square=
                long_dot(sums[left],sums[left]);
            const long double right_square=
                long_dot(sums[right],sums[right]);
            if (!(left_square>0.0L && right_square>0.0L)) {
                throw std::domain_error(
                    "regulated K4 recurrence has a zero partial sum");
            }
            const LongVec3 total=long_add(sums[left],sums[right]);
            const long double alpha_value=
                long_dot(total,sums[left])/left_square;
            const long double beta_value=
                long_dot(total,total)*long_dot(sums[left],sums[right])
                /(2.0L*left_square*right_square);
            const LongKernelPair& left_kernel=kernels[left];
            const LongKernelPair& right_kernel=kernels[right];
            kernels[mask].F+=
                symmetry*left_kernel.G
                *((2*order+1)*alpha_value*right_kernel.F
                  +2.0L*beta_value*right_kernel.G)
                /denominator;
            kernels[mask].G+=
                symmetry*left_kernel.G
                *(3.0L*alpha_value*right_kernel.F
                  +2.0L*order*beta_value*right_kernel.G)
                /denominator;
        }
    }
    return kernels[15];
}

LongKernelPair fourth_order_opposite_limit(
    const std::vector<Vec3>& momenta,
    const std::pair<int,int>& opposite) {
    std::array<Vec3,2> external={};
    int external_count=0;
    for (int index=0;index<4;++index) {
        if (index!=opposite.first && index!=opposite.second) {
            external[static_cast<std::size_t>(external_count++)]=
                momenta[static_cast<std::size_t>(index)];
        }
    }
    if (external_count!=2) {
        throw std::logic_error(
            "fourth-order opposite limit did not find two external legs");
    }
    Vec3 canonical_loop=momenta[static_cast<std::size_t>(opposite.first)];
    const double loop_scale=marisa_b_halo_v1::norm(canonical_loop);
    const double external_scale=std::min(
        marisa_b_halo_v1::norm(external[0]),
        marisa_b_halo_v1::norm(external[1]));
    if (!(loop_scale>0.0 && external_scale>0.0)) {
        throw std::invalid_argument(
            "fourth-order opposite limit has a zero momentum");
    }
    const std::array<double,3> components={{
        canonical_loop.x,canonical_loop.y,canonical_loop.z}};
    for (double component:components) {
        if (std::fabs(component)<=kZeroTolerance*loop_scale) continue;
        if (component<0.0) {
            canonical_loop=marisa_b_halo_v1::negate(canonical_loop);
        }
        break;
    }

    const LongVec3 loop=long_vector(canonical_loop);
    const long double loop_norm=long_norm(loop);
    const LongVec3 loop_hat=long_scale(loop,1.0L/loop_norm);
    const std::array<long double,3> absolute_components={{
        std::fabs(loop_hat.x),std::fabs(loop_hat.y),
        std::fabs(loop_hat.z)}};
    const std::size_t axis_index=static_cast<std::size_t>(
        std::min_element(
            absolute_components.begin(),absolute_components.end())
        -absolute_components.begin());
    LongVec3 axis;
    if (axis_index==0) axis.x=1.0L;
    else if (axis_index==1) axis.y=1.0L;
    else axis.z=1.0L;
    LongVec3 transverse=long_cross(loop_hat,axis);
    transverse=long_scale(transverse,1.0L/long_norm(transverse));

    const long double theta=std::min(
        0.02L,
        0.02L*static_cast<long double>(external_scale)/loop_norm);
    if (!(theta>64.0L*std::numeric_limits<long double>::epsilon())) {
        throw std::domain_error(
            "fourth-order opposite regulator is below long-double resolution");
    }
    const auto averaged=[
        &external,&loop,&loop_hat,&transverse,loop_norm
    ](long double angle) {
        LongKernelPair result;
        for (const long double sign:{-1.0L,1.0L}) {
            const LongVec3 near_minus=long_scale(
                long_add(
                    long_scale(loop_hat,-std::cos(angle)),
                    long_scale(transverse,sign*std::sin(angle))),
                loop_norm);
            const std::array<LongVec3,4> regulated={{
                long_vector(external[0]),long_vector(external[1]),
                loop,near_minus}};
            const LongKernelPair value=
                regulated_fourth_order_kernel(regulated);
            result.F+=0.5L*value.F;
            result.G+=0.5L*value.G;
        }
        return result;
    };
    const std::array<LongKernelPair,3> values={{
        averaged(theta),averaged(0.5L*theta),averaged(0.25L*theta)}};
    const LongKernelPair first_coarse{
        (4.0L*values[1].F-values[0].F)/3.0L,
        (4.0L*values[1].G-values[0].G)/3.0L};
    const LongKernelPair first_fine{
        (4.0L*values[2].F-values[1].F)/3.0L,
        (4.0L*values[2].G-values[1].G)/3.0L};
    const LongKernelPair result{
        (16.0L*first_fine.F-first_coarse.F)/15.0L,
        (16.0L*first_fine.G-first_coarse.G)/15.0L};
    const double F=static_cast<double>(result.F);
    const double G=static_cast<double>(result.G);
    if (!(std::isfinite(F) && std::isfinite(G))) {
        throw std::runtime_error(
            "fourth-order opposite limit is non-finite");
    }
    return LongKernelPair{result.F,result.G};
}

SptKernelPair third_order_opposite_limit(
    const Vec3& external,
    const Vec3& loop) {
    const double K=marisa_b_halo_v1::norm(external);
    const double Q=marisa_b_halo_v1::norm(loop);
    if (!(K>0.0 && Q>0.0)) throw std::invalid_argument("invalid K3 opposite-pair momentum");
    const double u=cosine(external,loop);
    const long double K2=static_cast<long double>(K)*K;
    const long double Q2=static_cast<long double>(Q)*Q;
    const long double u2=static_cast<long double>(u)*u;
    const long double u4=u2*u2;
    const long double K4=K2*K2;
    const long double Q4=Q2*Q2;
    const long double dminus=-K2+2.0L*K*Q*u-Q2;
    const long double dplus=K2+2.0L*K*Q*u+Q2;
    const long double denominator_F=126.0L*Q2*dminus*dplus;
    const long double denominator_G=42.0L*Q2*dminus*dplus;
    if (std::fabs(denominator_F)<=std::numeric_limits<long double>::epsilon()*K4*Q4) {
        // A finite periodic momentum lattice can hit K=Q and |u|=1
        // exactly.  At that point the closed form below is 0/0, although
        // both the radial and angular continuations have the same removable
        // limit.  Factoring either at fixed K=Q or fixed |u|=1 gives
        // F3(k,k,-k)=G3(k,k,-k)=-1/6.  Continuous quadrature never samples
        // this measure-zero surface, but a discrete loop sum requires the
        // explicit value.
        return SptKernelPair{-1.0/6.0,-1.0/6.0};
    }
    const long double numerator_F=
        -21.0L*K4*u2+76.0L*K2*Q2*u4-44.0L*K2*Q2*u2
        +10.0L*K2*Q2+28.0L*Q4*u4-59.0L*Q4*u2+10.0L*Q4;
    const long double numerator_G=
        -7.0L*K4*u2+20.0L*K2*Q2*u4-4.0L*K2*Q2*u2
        -2.0L*K2*Q2+4.0L*Q4*u4-9.0L*Q4*u2-2.0L*Q4;
    return SptKernelPair{
        static_cast<double>(-K2*numerator_F/denominator_F),
        static_cast<double>(-K2*numerator_G/denominator_G)};
}

double alpha(const Vec3& left, const Vec3& right) {
    const double denominator = marisa_b_halo_v1::dot(left,left);
    if (denominator <= kZeroTolerance*kZeroTolerance) {
        throw std::domain_error("SPT alpha has a zero left partial sum");
    }
    return marisa_b_halo_v1::dot(marisa_b_halo_v1::add(left,right),left)/denominator;
}

double beta(const Vec3& left, const Vec3& right) {
    const double left_square = marisa_b_halo_v1::dot(left,left);
    const double right_square = marisa_b_halo_v1::dot(right,right);
    if (left_square <= kZeroTolerance*kZeroTolerance
        || right_square <= kZeroTolerance*kZeroTolerance) {
        throw std::domain_error("SPT beta has a zero partial sum");
    }
    const Vec3 total = marisa_b_halo_v1::add(left,right);
    return marisa_b_halo_v1::dot(total,total)
           * marisa_b_halo_v1::dot(left,right)/(2.0*left_square*right_square);
}

SptKernelPair recurse(const std::vector<Vec3>& momenta) {
    const int order = static_cast<int>(momenta.size());
    if (order == 1) {
        if (!(marisa_b_halo_v1::norm(momenta.front()) > 0.0)) {
            throw std::invalid_argument("linear SPT kernel received a zero momentum");
        }
        return SptKernelPair{1.0,1.0};
    }
    const double denominator = static_cast<double>((2*order+3)*(order-1));
    const unsigned int full = (1U << order)-1U;
    long double F = 0.0L;
    long double G = 0.0L;
    for (int left_order = 1; left_order < order; ++left_order) {
        const double symmetry = 1.0/binomial(order,left_order);
        for (unsigned int mask = 1U; mask < full; ++mask) {
            if (popcount(mask) != left_order) continue;
            const std::vector<Vec3> left_vectors = select(momenta,mask);
            const std::vector<Vec3> right_vectors = select(momenta,full^mask);
            const SptKernelPair left = recurse(left_vectors);
            const SptKernelPair right = recurse(right_vectors);
            // If a subkernel vanishes at an exact opposite pair, the full
            // recurrence contribution has a continuous zero limit.  Skipping
            // it avoids evaluating the individually undefined alpha/beta.
            if (std::fabs(left.G) <= kZeroTolerance) continue;
            if (std::fabs(right.F) <= kZeroTolerance
                && std::fabs(right.G) <= kZeroTolerance) continue;
            const Vec3 left_sum = sum(left_vectors);
            const Vec3 right_sum = sum(right_vectors);
            const double left_norm = marisa_b_halo_v1::norm(left_sum);
            const double right_norm = marisa_b_halo_v1::norm(right_sum);
            if (left_norm <= kZeroTolerance || right_norm <= kZeroTolerance) continue;
            const double a = alpha(left_sum,right_sum);
            const double b = beta(left_sum,right_sum);
            F += symmetry*left.G*((2*order+1)*a*right.F+2*b*right.G)/denominator;
            G += symmetry*left.G*(3*a*right.F+2*order*b*right.G)/denominator;
        }
    }
    const SptKernelPair result{static_cast<double>(F),static_cast<double>(G)};
    if (!std::isfinite(result.F) || !std::isfinite(result.G)) {
        throw std::runtime_error("non-finite symmetrized SPT kernel");
    }
    return result;
}

}  // namespace

double cosine(const Vec3& left, const Vec3& right) {
    const double denominator = marisa_b_halo_v1::norm(left)*marisa_b_halo_v1::norm(right);
    if (denominator <= kZeroTolerance) return 0.0;
    return std::max(-1.0,std::min(1.0,marisa_b_halo_v1::dot(left,right)/denominator));
}

double kappa(const Vec3& left, const Vec3& right) {
    if (marisa_b_halo_v1::norm(left) <= kZeroTolerance
        || marisa_b_halo_v1::norm(right) <= kZeroTolerance) return 0.0;
    const double mu = cosine(left,right);
    return mu*mu-1.0;
}

double galileon_L(const Vec3& first, const Vec3& second, const Vec3& third) {
    if (marisa_b_halo_v1::norm(first) <= kZeroTolerance
        || marisa_b_halo_v1::norm(second) <= kZeroTolerance
        || marisa_b_halo_v1::norm(third) <= kZeroTolerance) return 0.0;
    const double ab=cosine(first,second),bc=cosine(second,third),ca=cosine(third,first);
    return 2.0*ab*bc*ca-ab*ab-bc*bc-ca*ca+1.0;
}

double galileon_M(
    const Vec3& first,
    const Vec3& second,
    const Vec3& third,
    const Vec3& fourth) {
    if (marisa_b_halo_v1::norm(first) <= kZeroTolerance
        || marisa_b_halo_v1::norm(second) <= kZeroTolerance
        || marisa_b_halo_v1::norm(third) <= kZeroTolerance
        || marisa_b_halo_v1::norm(fourth) <= kZeroTolerance) return 0.0;
    return cosine(first,second)*cosine(second,third)
           *cosine(third,fourth)*cosine(fourth,first);
}

Vec3 sum(const std::vector<Vec3>& vectors) {
    Vec3 result;
    for (const Vec3& value : vectors) result=marisa_b_halo_v1::add(result,value);
    return result;
}

SptKernelPair spt_kernels(const std::vector<Vec3>& momenta) {
    if (momenta.empty() || momenta.size()>4) {
        throw std::invalid_argument("SPT kernel order must be one through four");
    }
    const std::pair<int,int> opposite=exact_opposite_pair(momenta);
    if (momenta.size()==3 && opposite.first>=0) {
        int external=0;
        while (external==opposite.first || external==opposite.second) ++external;
        return third_order_opposite_limit(momenta[external],momenta[opposite.first]);
    }
    if (momenta.size()==4 && opposite.first>=0) {
        // This is the independent EFT-v2 continuation.  The frozen v1/DM
        // compatibility provider still calls halo-v1 directly and therefore
        // retains its historical ReACT prescription.
        const LongKernelPair exact=
            fourth_order_opposite_limit(momenta,opposite);
        return SptKernelPair{
            static_cast<double>(exact.F),
            static_cast<double>(exact.G)};
    }
    return recurse(momenta);
}

double spt_F(const std::vector<Vec3>& momenta) { return spt_kernels(momenta).F; }
double spt_G(const std::vector<Vec3>& momenta) { return spt_kernels(momenta).G; }

}  // namespace marisa_b_eft_v2
