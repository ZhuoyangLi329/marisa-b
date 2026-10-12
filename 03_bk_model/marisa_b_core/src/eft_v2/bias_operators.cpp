#include "bias_operators.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace marisa_b_eft_v2 {
namespace {

SparsePolynomial variable(ParameterId id) { return SparsePolynomial::variable(id); }

Vec3 vec_add(const Vec3& left,const Vec3& right) { return marisa_b_halo_v1::add(left,right); }

SparsePolynomial unsymmetrized(
    const std::vector<Vec3>& q,bool include_matter_kernel=true) {
    const int order=static_cast<int>(q.size());
    const SparsePolynomial b1=variable(ParameterId::B1);
    if (order==1) return b1;
    const SparsePolynomial b2=variable(ParameterId::B2);
    const SparsePolynomial gamma2=variable(ParameterId::Gamma2);
    if (order==2) {
        return (include_matter_kernel?spt_F(q)*b1:SparsePolynomial{})
               +0.5*b2+kappa(q[0],q[1])*gamma2;
    }
    const SparsePolynomial b3=variable(ParameterId::B3);
    const SparsePolynomial gamma2x=variable(ParameterId::Gamma2x);
    const SparsePolynomial gamma3=variable(ParameterId::Gamma3);
    const SparsePolynomial gamma21=variable(ParameterId::Gamma21);
    if (order==3) {
        const Vec3 q23=vec_add(q[1],q[2]);
        return (include_matter_kernel?spt_F(q)*b1:SparsePolynomial{})
               +spt_F({q[0],q[1]})*b2
               +2.0*kappa(q[0],q23)*spt_G({q[1],q[2]})*gamma2
               +(1.0/6.0)*b3
               +kappa(q[0],q[1])*gamma2x
               +galileon_L(q[0],q[1],q[2])*gamma3
               +kappa(q[0],q23)*kappa(q[1],q[2])*gamma21;
    }
    if (order!=4) throw std::invalid_argument("deterministic kernel order must be one through four");

    const SparsePolynomial gamma21x=variable(ParameterId::Gamma21x);
    const SparsePolynomial gamma211=variable(ParameterId::Gamma211);
    const SparsePolynomial gamma22=variable(ParameterId::Gamma22);
    const SparsePolynomial gamma31=variable(ParameterId::Gamma31);
    const Vec3 q12=vec_add(q[0],q[1]);
    const Vec3 q23=vec_add(q[1],q[2]);
    const Vec3 q34=vec_add(q[2],q[3]);
    const Vec3 q123=vec_add(q12,q[2]);
    const Vec3 q234=vec_add(q23,q[3]);
    SparsePolynomial result=include_matter_kernel
        ?spt_F(q)*b1:SparsePolynomial{};
    result += 0.5*(spt_F({q[0],q[1]})*spt_F({q[2],q[3]})
                   +2.0*spt_F({q[0],q[1],q[2]}))*b2;
    result += (kappa(q12,q34)*spt_G({q[0],q[1]})*spt_G({q[2],q[3]})
               +2.0*kappa(q123,q[3])*spt_G({q[0],q[1],q[2]}))*gamma2;
    result += 0.5*spt_F({q[0],q[1]})*b3;
    result += (2.0*kappa(q12,q[2])*spt_G({q[0],q[1]})
               +kappa(q[2],q[3])*spt_F({q[0],q[1]}))*gamma2x;
    result += 3.0*galileon_L(q[0],q[1],q34)*spt_G({q[2],q[3]})*gamma3;
    result += (kappa(q12,q34)*kappa(q[0],q[1])*spt_F({q[2],q[3]})
               +2.0*kappa(q123,q[3])*kappa(q12,q[2])*spt_F({q[0],q[1]}))*gamma21;
    result += kappa(q[0],q23)*kappa(q[1],q[2])*gamma21x;
    result += galileon_L(q[0],q[1],q34)*kappa(q[2],q[3])*gamma211;
    result += kappa(q12,q34)*kappa(q[0],q[1])*kappa(q[2],q[3])*gamma22;
    result += (
        (1.0/18.0)*kappa(q[0],q234)
            *((15.0/7.0)*kappa(q23,q[3])*kappa(q[1],q[2])
              -galileon_L(q[1],q[2],q[3]))
        +(1.0/14.0)*(galileon_M(q[0],q23,q[3],q234)
                     -galileon_M(q[0],q234,q23,q[3]))*kappa(q[1],q[2]))*gamma31;
    return result;
}

SparsePolynomial symmetrized_third_order(
    const std::vector<Vec3>& q) {
    const SparsePolynomial b1=variable(ParameterId::B1);
    const SparsePolynomial b2=variable(ParameterId::B2);
    const SparsePolynomial gamma2=variable(ParameterId::Gamma2);
    const SparsePolynomial b3=variable(ParameterId::B3);
    const SparsePolynomial gamma2x=variable(ParameterId::Gamma2x);
    const SparsePolynomial gamma3=variable(ParameterId::Gamma3);
    const SparsePolynomial gamma21=variable(ParameterId::Gamma21);
    SparsePolynomial result=spt_F(q)*b1+(1.0/6.0)*b3
                            +galileon_L(q[0],q[1],q[2])*gamma3;
    SparsePolynomial b2_sum;
    SparsePolynomial gamma2_sum;
    SparsePolynomial gamma2x_sum;
    SparsePolynomial gamma21_sum;
    for (int singleton=0;singleton<3;++singleton) {
        const int first=(singleton+1)%3;
        const int second=(singleton+2)%3;
        const Vec3 pair=vec_add(q[first],q[second]);
        b2_sum+=spt_F({q[first],q[second]})*b2;
        gamma2_sum+=kappa(q[singleton],pair)
                    *spt_G({q[first],q[second]})*gamma2;
        gamma2x_sum+=kappa(q[first],q[second])*gamma2x;
        gamma21_sum+=kappa(q[singleton],pair)
                     *kappa(q[first],q[second])*gamma21;
    }
    result+=(1.0/3.0)*b2_sum;
    result+=(2.0/3.0)*gamma2_sum;
    result+=(1.0/3.0)*gamma2x_sum;
    result+=(1.0/3.0)*gamma21_sum;
    return result;
}

std::vector<std::array<int,4>> permutation_table(int order) {
    std::array<int,4> indices={{0,1,2,3}};
    std::vector<std::array<int,4>> result;
    do { result.push_back(indices); }
    while (std::next_permutation(indices.begin(),indices.begin()+order));
    return result;
}

SparsePolynomial symmetrized_fourth_order(
    const std::vector<Vec3>& q) {
    const SparsePolynomial b1=variable(ParameterId::B1);
    const SparsePolynomial b2=variable(ParameterId::B2);
    const SparsePolynomial gamma2=variable(ParameterId::Gamma2);
    const SparsePolynomial b3=variable(ParameterId::B3);
    const SparsePolynomial gamma2x=variable(ParameterId::Gamma2x);
    const SparsePolynomial gamma3=variable(ParameterId::Gamma3);
    const SparsePolynomial gamma21=variable(ParameterId::Gamma21);
    const SparsePolynomial gamma21x=variable(ParameterId::Gamma21x);
    const SparsePolynomial gamma211=variable(ParameterId::Gamma211);
    const SparsePolynomial gamma22=variable(ParameterId::Gamma22);
    const SparsePolynomial gamma31=variable(ParameterId::Gamma31);

    std::array<std::array<double,4>,4> F2={};
    std::array<std::array<double,4>,4> G2={};
    for (int left=0;left<4;++left) {
        for (int right=left+1;right<4;++right) {
            F2[left][right]=F2[right][left]=spt_F({q[left],q[right]});
            G2[left][right]=G2[right][left]=spt_G({q[left],q[right]});
        }
    }
    std::array<double,16> F3={};
    std::array<double,16> G3={};
    for (int omitted=0;omitted<4;++omitted) {
        std::vector<Vec3> triple;
        int mask=0;
        for (int index=0;index<4;++index) {
            if (index==omitted) continue;
            triple.push_back(q[index]);
            mask|=1<<index;
        }
        F3[static_cast<std::size_t>(mask)]=spt_F(triple);
        G3[static_cast<std::size_t>(mask)]=spt_G(triple);
    }

    static const std::vector<std::array<int,4>> permutations=permutation_table(4);
    SparsePolynomial remainder;
    for (const auto& p:permutations) {
        const Vec3& q0=q[p[0]];
        const Vec3& q1=q[p[1]];
        const Vec3& q2=q[p[2]];
        const Vec3& q3=q[p[3]];
        const Vec3 q12=vec_add(q0,q1);
        const Vec3 q23=vec_add(q1,q2);
        const Vec3 q34=vec_add(q2,q3);
        const Vec3 q123=vec_add(q12,q2);
        const Vec3 q234=vec_add(q23,q3);
        const int triple012=(1<<p[0])|(1<<p[1])|(1<<p[2]);
        remainder+=0.5*(F2[p[0]][p[1]]*F2[p[2]][p[3]]
                        +2.0*F3[static_cast<std::size_t>(triple012)])*b2;
        remainder+=(kappa(q12,q34)*G2[p[0]][p[1]]*G2[p[2]][p[3]]
                    +2.0*kappa(q123,q3)
                         *G3[static_cast<std::size_t>(triple012)])*gamma2;
        remainder+=0.5*F2[p[0]][p[1]]*b3;
        remainder+=(2.0*kappa(q12,q2)*G2[p[0]][p[1]]
                    +kappa(q2,q3)*F2[p[0]][p[1]])*gamma2x;
        remainder+=3.0*galileon_L(q0,q1,q34)*G2[p[2]][p[3]]*gamma3;
        remainder+=(kappa(q12,q34)*kappa(q0,q1)*F2[p[2]][p[3]]
                    +2.0*kappa(q123,q3)*kappa(q12,q2)
                         *F2[p[0]][p[1]])*gamma21;
        remainder+=kappa(q0,q23)*kappa(q1,q2)*gamma21x;
        remainder+=galileon_L(q0,q1,q34)*kappa(q2,q3)*gamma211;
        remainder+=kappa(q12,q34)*kappa(q0,q1)*kappa(q2,q3)*gamma22;
        remainder+=(
            (1.0/18.0)*kappa(q0,q234)
                *((15.0/7.0)*kappa(q23,q3)*kappa(q1,q2)
                  -galileon_L(q1,q2,q3))
            +(1.0/14.0)*(galileon_M(q0,q23,q3,q234)
                         -galileon_M(q0,q234,q23,q3))*kappa(q1,q2))*gamma31;
    }
    remainder*=1.0/24.0;
    return spt_F(q)*b1+remainder;
}

SparsePolynomial symmetrize(const std::vector<Vec3>& momenta,bool slow) {
    const int order=static_cast<int>(momenta.size());
    if (order<1 || order>4) throw std::invalid_argument("deterministic kernel order must be one through four");
    if (order==1 || order==2) return unsymmetrized(momenta);
    if (!slow && order==3) return symmetrized_third_order(momenta);
    if (!slow && order==4) return symmetrized_fourth_order(momenta);
    SparsePolynomial result;
    if (slow) {
        std::vector<int> indices(order);
        std::iota(indices.begin(),indices.end(),0);
        do {
            std::vector<Vec3> permuted;
            for (int index:indices) permuted.push_back(momenta[index]);
            result+=unsymmetrized(permuted);
        } while (std::next_permutation(indices.begin(),indices.end()));
        result*=1.0/static_cast<double>(order==3?6:24);
    } else {
        // F_n is already fully symmetrized.  Evaluating its recursive kernel
        // in every one of the 3! or 4! bias-operator permutations is exactly
        // redundant and dominates R0 template generation, especially for K4.
        result=spt_F(momenta)*variable(ParameterId::B1);
        static const std::vector<std::array<int,4>> permutations3=permutation_table(3);
        static const std::vector<std::array<int,4>> permutations4=permutation_table(4);
        const auto& table=order==3?permutations3:permutations4;
        SparsePolynomial bias_remainder;
        for (const auto& indices:table) {
            std::vector<Vec3> permuted;
            for (int position=0;position<order;++position) permuted.push_back(momenta[indices[position]]);
            bias_remainder+=unsymmetrized(permuted,false);
        }
        bias_remainder*=1.0/static_cast<double>(order==3?6:24);
        result+=bias_remainder;
    }
    return result;
}

}  // namespace

SparsePolynomial deterministic_kernel_slow(const std::vector<Vec3>& momenta) {
    return symmetrize(momenta,true);
}

SparsePolynomial deterministic_kernel(const std::vector<Vec3>& momenta) {
    return symmetrize(momenta,false);
}

double evaluate_bias_kernel(const SparsePolynomial& kernel,const BiasValues& values) {
    std::array<double,kParameterCount> all={};
    std::copy(values.begin(),values.end(),all.begin());
    return kernel.evaluate(all);
}

double tree_bispectrum(
    const std::array<Vec3,3>& closed_triangle,
    const std::array<double,3>& linear_power,
    const BiasValues& values) {
    const Vec3 closure=vec_add(vec_add(closed_triangle[0],closed_triangle[1]),closed_triangle[2]);
    const double scale=std::max({
        marisa_b_halo_v1::norm(closed_triangle[0]),
        marisa_b_halo_v1::norm(closed_triangle[1]),
        marisa_b_halo_v1::norm(closed_triangle[2]),1.0});
    if (marisa_b_halo_v1::norm(closure)>1.0e-12*scale) {
        throw std::invalid_argument("tree bispectrum triangle does not close");
    }
    double result=0.0;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            const double K1left=evaluate_bias_kernel(deterministic_kernel({closed_triangle[left]}),values);
            const double K1right=evaluate_bias_kernel(deterministic_kernel({closed_triangle[right]}),values);
            const double K2=evaluate_bias_kernel(
                deterministic_kernel({closed_triangle[left],closed_triangle[right]}),values);
            result+=2.0*K1left*K1right*K2*linear_power[left]*linear_power[right];
        }
    }
    return result;
}

}  // namespace marisa_b_eft_v2
