#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "PowerSpectrum.h"
#include "diagram_assembler.h"
#include "direct_evaluator.h"
#include "field_kernel_provider.h"
#include "lattice_shell_rule.h"
#include "parameter_registry.h"
#include "shell_projector.h"
#include "stochastic.h"

namespace eft = marisa_b_eft_v2;
namespace shell = marisa_b_shell_v1;

namespace {

constexpr double kBoxSize = 1000.0;
constexpr int kMeshSize = 256;
constexpr int kDefaultRadialStop = 7;
#ifndef MARISA_B_V0P9_SMOOTHING_RADIUS
#define MARISA_B_V0P9_SMOOTHING_RADIUS 15.0
#endif
#ifndef MARISA_B_V0P9_RECONSTRUCTION_BIAS
#define MARISA_B_V0P9_RECONSTRUCTION_BIAS 2.7340475186190334
#endif
#ifndef MARISA_B_V0P9_RECONSTRUCTION_PHI_BIAS
#define MARISA_B_V0P9_RECONSTRUCTION_PHI_BIAS 5.84720823278338
#endif
// Keep the released z=1 values as compile-time defaults, while allowing an
// independently hashed executable to be built for another frozen
// redshift/reconstruction contract.  The values are echoed in every JSONL
// header, so a cache cannot silently cross configurations.
constexpr double kSmoothingRadius = MARISA_B_V0P9_SMOOTHING_RADIUS;
constexpr double kHaloReconstructionBias =
    MARISA_B_V0P9_RECONSTRUCTION_BIAS;
constexpr double kHaloReconstructionPhiBias =
    MARISA_B_V0P9_RECONSTRUCTION_PHI_BIAS;
constexpr double kCellSize = 8.0;
constexpr double kReconstructionMinimumK = 2.0 * 3.14159265358979323846 / kBoxSize;

class LinearBiasMatterKernelProvider final : public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        return eft::spt_F(momenta)
               * eft::SparsePolynomial::variable(eft::ParameterId::B1);
    }

    std::string_view name() const noexcept override {
        return "linear_bias_times_matter_spt";
    }
};

constexpr eft::ParameterId kOracleMarker=eft::ParameterId::Gamma31;

bool contains_parameter(
    const eft::SparsePolynomial& polynomial,
    eft::ParameterId parameter) {
    for (const auto& term:polynomial.terms()) {
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==parameter) return true;
        }
    }
    return false;
}

eft::SparsePolynomial build_term(
    double coefficient,
    const std::vector<eft::MonomialFactor>& factors) {
    eft::SparsePolynomial result=
        eft::SparsePolynomial::constant(coefficient);
    for (const eft::MonomialFactor& factor:factors) {
        for (int power=0;power<static_cast<int>(factor.power);++power) {
            result=result*eft::SparsePolynomial::variable(factor.id);
        }
    }
    return result;
}

eft::SparsePolynomial factor_out_one_b1(
    const eft::SparsePolynomial& polynomial) {
    eft::SparsePolynomial result;
    for (const auto& term:polynomial.terms()) {
        bool found=false;
        std::vector<eft::MonomialFactor> factors;
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==eft::ParameterId::B1) {
                if (factor.power<1) {
                    throw std::logic_error("invalid zero b1 power");
                }
                found=true;
                if (factor.power>1) {
                    factors.push_back(eft::MonomialFactor{
                        factor.id,
                        static_cast<std::uint8_t>(factor.power-1)});
                }
            } else {
                factors.push_back(factor);
            }
        }
        if (!found) {
            throw std::logic_error(
                "oracle tagged kernel term lacks the exact common b1 factor");
        }
        result+=build_term(term.second,factors);
    }
    return result;
}

class TaggedOracleMatterKernelProvider final : public eft::FieldKernelProvider {
public:
    TaggedOracleMatterKernelProvider(
        const eft::FieldKernelProvider& base,
        const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
        const PowerSpectrum& transfer)
        :base_(base),reconstruction_(reconstruction),transfer_(transfer) {
        if (reconstruction_.local_png_bias_transfer!=nullptr
            ||reconstruction_.local_png_bias_amplitude!=0.0) {
            throw std::invalid_argument(
                "tagged oracle provider needs a fixed reconstruction map");
        }
    }

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        const eft::FieldKernelVariation variation=
            eft::reconstructed_field_kernel_local_png_denominator_variation(
                base_,reconstruction_,transfer_,
                kReconstructionMinimumK,momenta);
        if (contains_parameter(variation.value,kOracleMarker)
            ||contains_parameter(variation.direction,kOracleMarker)) {
            throw std::logic_error("oracle marker collides with physical kernel");
        }
        const eft::SparsePolynomial value=
            factor_out_one_b1(variation.value);
        eft::SparsePolynomial direction;
        if (!variation.direction.empty()) {
            direction=
                kHaloReconstructionPhiBias
                *factor_out_one_b1(variation.direction);
        }
        return value
               +direction*eft::SparsePolynomial::variable(kOracleMarker);
    }

    std::string_view name() const noexcept override {
        return "tagged_oracle_linear_bias_matter_spt";
    }

    bool supports_triangle_plane_reflection() const noexcept override {
        return false;
    }

private:
    const eft::FieldKernelProvider& base_;
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    const PowerSpectrum& transfer_;
};

constexpr std::array<double,4> kGridB1{{1.0,2.0,3.0,4.0}};

struct GridDual {
    std::array<double,kGridB1.size()> value{};
    std::array<double,kGridB1.size()> direction{};

    GridDual& operator+=(const GridDual& other) {
        for (std::size_t index=0;index<kGridB1.size();++index) {
            value[index]+=other.value[index];
            direction[index]+=other.direction[index];
        }
        return *this;
    }

    GridDual& operator-=(const GridDual& other) {
        for (std::size_t index=0;index<kGridB1.size();++index) {
            value[index]-=other.value[index];
            direction[index]-=other.direction[index];
        }
        return *this;
    }

    GridDual& operator*=(double scalar) {
        for (std::size_t index=0;index<kGridB1.size();++index) {
            value[index]*=scalar;
            direction[index]*=scalar;
        }
        return *this;
    }
};

GridDual operator+(GridDual left,const GridDual& right) {
    left+=right;
    return left;
}

GridDual operator-(GridDual left,const GridDual& right) {
    left-=right;
    return left;
}

GridDual operator*(const GridDual& left,const GridDual& right) {
    GridDual result;
    for (std::size_t index=0;index<kGridB1.size();++index) {
        result.value[index]=left.value[index]*right.value[index];
        result.direction[index]=
            left.direction[index]*right.value[index]
            +left.value[index]*right.direction[index];
    }
    return result;
}

GridDual operator*(GridDual value,double scalar) {
    value*=scalar;
    return value;
}

GridDual operator*(double scalar,GridDual value) {
    value*=scalar;
    return value;
}

struct GridPartition {
    std::vector<unsigned int> masks;
    double symmetry=0.0;
};

double grid_factorial(int value) {
    double result=1.0;
    for (int factor=2;factor<=value;++factor) {
        result*=static_cast<double>(factor);
    }
    return result;
}

void generate_grid_partitions(
    int next,
    int count,
    std::vector<unsigned int>& blocks,
    std::vector<GridPartition>& output) {
    if (next==count) {
        double symmetry=1.0/grid_factorial(count);
        for (const unsigned int mask:blocks) {
            symmetry*=grid_factorial(__builtin_popcount(mask));
        }
        output.push_back(GridPartition{blocks,symmetry});
        return;
    }
    const unsigned int bit=1U<<static_cast<unsigned int>(next);
    for (std::size_t index=0;index<blocks.size();++index) {
        blocks[index]|=bit;
        generate_grid_partitions(next+1,count,blocks,output);
        blocks[index]&=~bit;
    }
    blocks.push_back(bit);
    generate_grid_partitions(next+1,count,blocks,output);
    blocks.pop_back();
}

const std::vector<GridPartition>& grid_partitions(int order) {
    if (order<1 ||order>4) {
        throw std::invalid_argument(
            "grid reconstructed kernel order must be one through four");
    }
    static const std::array<std::vector<GridPartition>,5> cache=[] {
        std::array<std::vector<GridPartition>,5> result;
        for (int value=1;value<=4;++value) {
            std::vector<unsigned int> blocks;
            generate_grid_partitions(
                0,value,blocks,result[static_cast<std::size_t>(value)]);
        }
        return result;
    }();
    return cache[static_cast<std::size_t>(order)];
}

GridDual grid_reconstructed_matter_kernel(
    const std::vector<eft::Vec3>& momenta,
    const marisa_b_halo_v1::ReconstructionConfig& fixed_reconstruction,
    const PowerSpectrum& transfer) {
    const int order=static_cast<int>(momenta.size());
    if (order<1 ||order>4) {
        throw std::invalid_argument(
            "grid reconstructed kernel order must be one through four");
    }
    const std::size_t subset_count=std::size_t{1}<<order;
    std::array<std::vector<eft::Vec3>,16> vectors;
    std::array<eft::Vec3,16> sums{};
    std::array<double,16> scales{};
    std::array<double,16> kernels{};
    for (std::size_t mask=1;mask<subset_count;++mask) {
        scales[mask]=1.0;
        for (int index=0;index<order;++index) {
            if ((mask&(std::size_t{1}<<index))==0) continue;
            const eft::Vec3& momentum=momenta[static_cast<std::size_t>(index)];
            vectors[mask].push_back(momentum);
            sums[mask]=marisa_b_halo_v1::add(sums[mask],momentum);
            scales[mask]=std::max(
                scales[mask],marisa_b_halo_v1::norm(momentum));
        }
        kernels[mask]=eft::spt_F(vectors[mask]);
    }

    GridDual result;
    if (!fixed_reconstruction.enabled) {
        result.value.fill(kernels[subset_count-1]);
        return result;
    }
    const eft::Vec3 output=sums[subset_count-1];
    for (std::size_t probe=0;probe<kGridB1.size();++probe) {
        marisa_b_halo_v1::ReconstructionConfig reconstruction=
            fixed_reconstruction;
        reconstruction.bias_recon/=kGridB1[probe];
        double value_sum=0.0;
        double direction_sum=0.0;
        for (const GridPartition& partition:grid_partitions(order)) {
            for (std::size_t density_index=0;
                 density_index<partition.masks.size();++density_index) {
                const unsigned int density_mask=
                    partition.masks[density_index];
                if (marisa_b_halo_v1::norm(sums[density_mask])
                    <=1.0e-14*scales[density_mask]) {
                    continue;
                }
                double term_value=kernels[density_mask];
                double term_direction=0.0;
                bool vanishes=false;
                for (std::size_t block_index=0;
                     block_index<partition.masks.size();++block_index) {
                    if (block_index==density_index) continue;
                    const unsigned int shift_mask=
                        partition.masks[block_index];
                    const auto shift=marisa_b_halo_v1::
                        reconstruction_shift_factor_local_png_variation(
                            output,sums[shift_mask],reconstruction,
                            transfer,kReconstructionMinimumK);
                    if (shift.value==0.0
                        &&shift.local_png_denominator_direction==0.0) {
                        vanishes=true;
                        break;
                    }
                    const double old_value=term_value;
                    term_value*=shift.value*kernels[shift_mask];
                    term_direction=(
                        term_direction*shift.value
                        +old_value*shift.local_png_denominator_direction
                    )*kernels[shift_mask];
                }
                if (!vanishes) {
                    value_sum+=partition.symmetry*term_value;
                    direction_sum+=partition.symmetry*term_direction;
                }
            }
        }
        result.value[probe]=value_sum;
        result.direction[probe]=
            kHaloReconstructionPhiBias*direction_sum/kGridB1[probe];
    }
    return result;
}

class GridMatterKernelProvider {
public:
    GridMatterKernelProvider(
        marisa_b_halo_v1::ReconstructionConfig reconstruction,
        const PowerSpectrum& transfer)
        :reconstruction_(std::move(reconstruction)),transfer_(transfer) {}

    GridDual deterministic(const std::vector<eft::Vec3>& momenta) const {
        return grid_reconstructed_matter_kernel(
            momenta,reconstruction_,transfer_);
    }

private:
    marisa_b_halo_v1::ReconstructionConfig reconstruction_;
    const PowerSpectrum& transfer_;
};

eft::Vec3 grid_minus(const eft::Vec3& value) {
    return marisa_b_halo_v1::negate(value);
}

eft::Vec3 grid_add(const eft::Vec3& left,const eft::Vec3& right) {
    return marisa_b_halo_v1::add(left,right);
}

eft::Vec3 grid_subtract(const eft::Vec3& left,const eft::Vec3& right) {
    return marisa_b_halo_v1::subtract(left,right);
}

int grid_pair_index(int left,int right) {
    if (left>right) std::swap(left,right);
    if (left==0 &&right==1) return 0;
    if (left==0 &&right==2) return 1;
    if (left==1 &&right==2) return 2;
    throw std::invalid_argument("invalid grid external pair");
}

class GridDiagramAssembler {
public:
    GridDiagramAssembler(
        const PowerSpectrum& power,
        const std::array<eft::Vec3,3>& closed_triangle,
        const GridMatterKernelProvider& provider)
        :power_(power),provider_(provider) {
        const auto canonical=
            marisa_b_halo_v1::canonicalize_closed_vectors(closed_triangle);
        external_={{canonical.k1,canonical.k2,canonical.k3}};
        for (int index=0;index<3;++index) {
            external_power_[index]=evaluate_power(
                marisa_b_halo_v1::norm(external_[index]));
            K1_[index]=provider_.deterministic({external_[index]});
        }
        K2_pairs_[0]=provider_.deterministic({external_[0],external_[1]});
        K2_pairs_[1]=provider_.deterministic({external_[0],external_[2]});
        K2_pairs_[2]=provider_.deterministic({external_[1],external_[2]});
    }

    const std::array<eft::Vec3,3>& external() const noexcept {
        return external_;
    }

    GridDual tree() const {
        GridDual result;
        for (int left=0;left<3;++left) {
            for (int right=left+1;right<3;++right) {
                result+=2.0*external_power_[left]*external_power_[right]
                        *K1_[left]*K1_[right]
                        *K2_pairs_[grid_pair_index(left,right)];
            }
        }
        return result;
    }

    GridDual integrand(eft::Diagram diagram,const eft::Vec3& loop) const {
        const eft::Vec3 negative_loop=grid_minus(loop);
        const double Pq=evaluate_power(marisa_b_halo_v1::norm(loop));
        if (diagram==eft::Diagram::B222) {
            const eft::Vec3 k1_minus_q=grid_subtract(external_[0],loop);
            const eft::Vec3 k2_plus_q=grid_add(external_[1],loop);
            return 8.0*Pq
                   *evaluate_power(marisa_b_halo_v1::norm(k1_minus_q))
                   *evaluate_power(marisa_b_halo_v1::norm(k2_plus_q))
                   *provider_.deterministic({loop,k1_minus_q})
                   *provider_.deterministic({negative_loop,k2_plus_q})
                   *provider_.deterministic(
                       {grid_minus(k1_minus_q),grid_minus(k2_plus_q)});
        }
        if (diagram==eft::Diagram::B321I) {
            GridDual result;
            for (int shifted=0;shifted<3;++shifted) {
                for (int external_power=0;external_power<3;++external_power) {
                    if (external_power==shifted) continue;
                    result+=b321i_route(external_power,shifted,loop);
                }
            }
            return result;
        }
        return Pq*tadpole_coefficient(diagram,loop);
    }

    GridDual b321i_route(
        int external_power_index,
        int shifted_power_index,
        const eft::Vec3& loop) const {
        const eft::Vec3 shifted=grid_subtract(
            external_[shifted_power_index],loop);
        const double Pq=evaluate_power(marisa_b_halo_v1::norm(loop));
        const double shifted_power=evaluate_power(
            marisa_b_halo_v1::norm(shifted));
        return 6.0*external_power_[external_power_index]*Pq*shifted_power
               *K1_[external_power_index]
               *provider_.deterministic({loop,shifted})
               *provider_.deterministic(
                   {external_[external_power_index],loop,shifted});
    }

    GridDual tadpole_coefficient(
        eft::Diagram diagram,const eft::Vec3& loop) const {
        if (diagram==eft::Diagram::B321II) {
            GridDual result;
            for (int left=0;left<3;++left) {
                for (int right=left+1;right<3;++right) {
                    result+=b321ii_route(left,right,right,loop);
                    result+=b321ii_route(left,right,left,loop);
                }
            }
            return result;
        }
        if (diagram!=eft::Diagram::B411) {
            throw std::invalid_argument("invalid grid tadpole diagram");
        }
        GridDual result;
        for (int left=0;left<3;++left) {
            for (int right=left+1;right<3;++right) {
                result+=12.0*external_power_[left]*external_power_[right]
                        *K1_[left]*K1_[right]
                        *provider_.deterministic(
                            {external_[left],external_[right],
                             loop,grid_minus(loop)});
            }
        }
        return result;
    }

    double loop_power(const eft::Vec3& loop) const {
        return evaluate_power(marisa_b_halo_v1::norm(loop));
    }

    std::vector<eft::Vec3> soft_centers(eft::Diagram diagram) const {
        const eft::Vec3 zero{};
        if (diagram==eft::Diagram::B222) {
            return {zero,external_[0],grid_minus(external_[1])};
        }
        if (diagram==eft::Diagram::B321I) {
            return {zero,external_[0],external_[1],external_[2]};
        }
        return {zero};
    }

private:
    double evaluate_power(double k) const {
        return power_(std::max(k,1.0e-12));
    }

    GridDual b321ii_route(
        int left,int right,int tadpole_external,
        const eft::Vec3& loop) const {
        const int linear_external=tadpole_external==left?right:left;
        return 6.0*external_power_[left]*external_power_[right]
               *K2_pairs_[grid_pair_index(left,right)]
               *K1_[linear_external]
               *provider_.deterministic(
                   {external_[tadpole_external],loop,grid_minus(loop)});
    }

    const PowerSpectrum& power_;
    const GridMatterKernelProvider& provider_;
    std::array<eft::Vec3,3> external_{};
    std::array<double,3> external_power_{};
    std::array<GridDual,3> K1_{};
    std::array<GridDual,3> K2_pairs_{};
};

struct GridIntegrands {
    GridDual B222;
    GridDual B321I;
    GridDual B321II;
    GridDual B411;
};

class GridIrSafeIntegrands {
public:
    GridIrSafeIntegrands(
        const GridDiagramAssembler& assembler,double qmax)
        :assembler_(assembler),
         B222_(assembler.soft_centers(eft::Diagram::B222),qmax),
         B321I_(assembler.soft_centers(eft::Diagram::B321I),qmax),
         B321II_(assembler.soft_centers(eft::Diagram::B321II),qmax),
         B411_(assembler.soft_centers(eft::Diagram::B411),qmax) {}

    GridDual component(eft::Diagram diagram,const eft::Vec3& loop) const {
        const auto evaluator=[this,diagram](const eft::Vec3& original) {
            return assembler_.integrand(diagram,original);
        };
        if (diagram==eft::Diagram::B222) return B222_.map(loop,evaluator);
        if (diagram==eft::Diagram::B321I) return B321I_.map(loop,evaluator);
        if (diagram==eft::Diagram::B321II) return B321II_.map(loop,evaluator);
        return B411_.map(loop,evaluator);
    }

    GridIntegrands all(const eft::Vec3& loop) const {
        return GridIntegrands{
            component(eft::Diagram::B222,loop),
            component(eft::Diagram::B321I,loop),
            component(eft::Diagram::B321II,loop),
            component(eft::Diagram::B411,loop)};
    }

    std::vector<double> radial_boundaries(
        const eft::Vec3& direction,double qmin) const {
        std::vector<double> result{qmin};
        const auto append=[&](const eft::SoftRegionMapper& mapper) {
            for (const double boundary:mapper.radial_boundaries(direction)) {
                if (boundary>qmin) result.push_back(boundary);
            }
        };
        append(B222_);
        append(B321I_);
        append(B321II_);
        append(B411_);
        std::sort(result.begin(),result.end());
        result.erase(
            std::unique(
                result.begin(),result.end(),
                [](double left,double right) {
                    return std::fabs(left-right)
                           <=1.0e-12*std::max({left,right,1.0});
                }),
            result.end());
        return result;
    }

private:
    const GridDiagramAssembler& assembler_;
    eft::SoftRegionMapper B222_;
    eft::SoftRegionMapper B321I_;
    eft::SoftRegionMapper B321II_;
    eft::SoftRegionMapper B411_;
};

eft::Vec3 grid_scale(const eft::Vec3& value,double factor) {
    return eft::Vec3{factor*value.x,factor*value.y,factor*value.z};
}

eft::Vec3 grid_unit(const eft::Vec3& value) {
    const double length=marisa_b_halo_v1::norm(value);
    if (!(length>0.0)) {
        throw std::invalid_argument("grid direction has zero length");
    }
    return grid_scale(value,1.0/length);
}

struct GridQuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

GridQuadratureRule grid_gauss_legendre(
    int count,double lower,double upper) {
    if (count<1 ||!(upper>lower)) {
        throw std::invalid_argument("invalid grid Gauss-Legendre rule");
    }
    GridQuadratureRule result;
    result.nodes.resize(static_cast<std::size_t>(count));
    result.weights.resize(static_cast<std::size_t>(count));
    const int half=(count+1)/2;
    const double midpoint=0.5*(lower+upper);
    const double half_width=0.5*(upper-lower);
    constexpr double pi=3.141592653589793238462643383279502884;
    for (int index=0;index<half;++index) {
        double root=std::cos(
            pi*(static_cast<double>(index)+0.75)
            /(static_cast<double>(count)+0.5));
        double derivative=0.0;
        for (int iteration=0;iteration<100;++iteration) {
            double previous=1.0;
            double current=root;
            for (int order=2;order<=count;++order) {
                const double next=(
                    (2.0*order-1.0)*root*current
                    -(order-1.0)*previous)/order;
                previous=current;
                current=next;
            }
            const double pn=count==1?root:current;
            const double pnm1=count==1?1.0:previous;
            derivative=count*(root*pn-pnm1)/(root*root-1.0);
            const double update=pn/derivative;
            root-=update;
            if (std::fabs(update)
                <4.0*std::numeric_limits<double>::epsilon()) break;
        }
        const double weight=
            2.0/((1.0-root*root)*derivative*derivative);
        result.nodes[static_cast<std::size_t>(index)]=
            midpoint-half_width*root;
        result.nodes[static_cast<std::size_t>(count-1-index)]=
            midpoint+half_width*root;
        result.weights[static_cast<std::size_t>(index)]=half_width*weight;
        result.weights[static_cast<std::size_t>(count-1-index)]=
            half_width*weight;
    }
    return result;
}

struct GridAngularNode {
    eft::Vec3 direction;
    double weight=0.0;
};

std::vector<GridAngularNode> grid_angular_nodes(int requested_count) {
    int count=std::max(32,requested_count);
    if (count%2!=0) ++count;
    const int half=count/2;
    constexpr double pi=3.141592653589793238462643383279502884;
    const double golden_angle=pi*(3.0-std::sqrt(5.0));
    const double weight=4.0*pi/static_cast<double>(count);
    std::vector<GridAngularNode> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int index=0;index<half;++index) {
        const double z=(index+0.5)/static_cast<double>(half);
        const double transverse=std::sqrt(std::max(0.0,1.0-z*z));
        const double phi=golden_angle*index;
        const eft::Vec3 value{
            transverse*std::cos(phi),transverse*std::sin(phi),z};
        result.push_back(GridAngularNode{value,weight});
        result.push_back(GridAngularNode{grid_scale(value,-1.0),weight});
    }
    return result;
}

struct GridUvComponents {
    GridDual B321II;
    GridDual B411;
};

class GridUvSubtraction {
public:
    GridUvSubtraction(
        const GridDiagramAssembler& assembler,
        eft::UvSubtractionConfig config)
        :assembler_(assembler),config_(config) {
        double external_scale=0.0;
        for (const eft::Vec3& value:assembler_.external()) {
            external_scale=std::max(
                external_scale,marisa_b_halo_v1::norm(value));
        }
        reference_q_=config_.asymptotic_factor
                     *std::max(external_scale,config_.mu_ren);
    }

    GridUvComponents integrands(const eft::Vec3& loop) const {
        const double radius=marisa_b_halo_v1::norm(loop);
        if (!(radius>0.0)) {
            throw std::invalid_argument("grid UV loop has zero length");
        }
        GridUvComponents result=asymptotic_coefficients(
            grid_scale(loop,1.0/radius));
        const double Pq=assembler_.loop_power(loop);
        result.B321II*=Pq;
        result.B411*=Pq;
        return result;
    }

    GridUvComponents angular_averaged_subleading_coefficients(
        int n_mu,int n_phi) const {
        constexpr double pi=3.141592653589793238462643383279502884;
        const GridQuadratureRule mu=grid_gauss_legendre(n_mu,-1.0,1.0);
        const GridQuadratureRule phi=
            grid_gauss_legendre(n_phi,0.0,2.0*pi);
        GridUvComponents result;
        for (int imu=0;imu<n_mu;++imu) {
            const double z=mu.nodes[static_cast<std::size_t>(imu)];
            const double transverse=std::sqrt(std::max(0.0,1.0-z*z));
            for (int iphi=0;iphi<n_phi;++iphi) {
                const double angle=phi.nodes[static_cast<std::size_t>(iphi)];
                const eft::Vec3 direction{
                    transverse*std::cos(angle),
                    transverse*std::sin(angle),z};
                const double weight=
                    mu.weights[static_cast<std::size_t>(imu)]
                    *phi.weights[static_cast<std::size_t>(iphi)]
                    /(4.0*pi);
                const GridUvComponents value=
                    subleading_coefficients(direction);
                result.B321II+=weight*value.B321II;
                result.B411+=weight*value.B411;
            }
        }
        return result;
    }

private:
    std::string direction_key(const eft::Vec3& direction) const {
        const eft::Vec3 normalized=grid_unit(direction);
        std::ostringstream output;
        output<<std::fixed<<std::setprecision(12)
              <<normalized.x<<','<<normalized.y<<','<<normalized.z;
        return output.str();
    }

    GridUvComponents compute_asymptotic(
        const eft::Vec3& direction) const {
        const eft::Vec3 normalized=grid_unit(direction);
        const eft::Vec3 q1=grid_scale(normalized,reference_q_);
        const eft::Vec3 q2=grid_scale(normalized,2.0*reference_q_);
        return GridUvComponents{
            (1.0/3.0)*(
                4.0*assembler_.tadpole_coefficient(
                    eft::Diagram::B321II,q2)
                -assembler_.tadpole_coefficient(
                    eft::Diagram::B321II,q1)),
            (1.0/3.0)*(
                4.0*assembler_.tadpole_coefficient(
                    eft::Diagram::B411,q2)
                -assembler_.tadpole_coefficient(
                    eft::Diagram::B411,q1))};
    }

    GridUvComponents asymptotic_coefficients(
        const eft::Vec3& direction) const {
        const std::string key=direction_key(direction);
        const auto found=cache_.find(key);
        if (found!=cache_.end()) return found->second;
        const GridUvComponents value=compute_asymptotic(direction);
        cache_.emplace(key,value);
        return value;
    }

    GridUvComponents subleading_coefficients(
        const eft::Vec3& input_direction) const {
        const eft::Vec3 direction=grid_unit(input_direction);
        const auto coefficient=[&](eft::Diagram diagram) {
            const auto antipodal=[&](double radius) {
                return 0.5*(
                    assembler_.tadpole_coefficient(
                        diagram,grid_scale(direction,radius))
                    +assembler_.tadpole_coefficient(
                        diagram,grid_scale(direction,-radius)));
            };
            const double q1=reference_q_;
            const GridDual value1=antipodal(q1);
            const GridDual value2=antipodal(2.0*q1);
            const GridDual value4=antipodal(4.0*q1);
            const GridDual estimate_q=(4.0/3.0)*q1*q1*(value1-value2);
            const GridDual estimate_2q=
                (16.0/3.0)*q1*q1*(value2-value4);
            return (1.0/3.0)*(4.0*estimate_2q-estimate_q);
        };
        return GridUvComponents{
            coefficient(eft::Diagram::B321II),
            coefficient(eft::Diagram::B411)};
    }

    const GridDiagramAssembler& assembler_;
    eft::UvSubtractionConfig config_;
    double reference_q_=0.0;
    mutable std::map<std::string,GridUvComponents> cache_;
};

struct GridTemplates {
    GridDual tree;
    GridDual B222;
    GridDual B321I;
    GridDual B321II;
    GridDual B411;
    GridDual B321II_bare;
    GridDual B411_bare;
    GridDual B321II_uv_subtraction;
    GridDual B411_uv_subtraction;
    GridDual B321II_uv_restoration;
    GridDual B411_uv_restoration;
    std::size_t integration_nodes=0;
};

GridTemplates grid_evaluate_direct(
    const PowerSpectrum& power,
    const std::array<eft::Vec3,3>& triangle,
    const GridMatterKernelProvider& provider,
    const eft::DirectIntegrationConfig& config) {
    constexpr double pi=3.141592653589793238462643383279502884;
    const double measure=1.0/(8.0*pi*pi*pi);
    const GridDiagramAssembler assembler(power,triangle,provider);
    const GridIrSafeIntegrands mapped(assembler,config.qmax);
    const GridUvSubtraction uv(assembler,config.uv);
    GridTemplates output;
    output.tree=assembler.tree();
    const std::vector<GridAngularNode> angular=grid_angular_nodes(
        2*config.n_mu*config.n_phi);
    for (const GridAngularNode& angle:angular) {
        const std::vector<double> boundaries=
            mapped.radial_boundaries(angle.direction,config.qmin);
        for (std::size_t segment=0;segment+1<boundaries.size();++segment) {
            const GridQuadratureRule radial=grid_gauss_legendre(
                config.n_radial,
                std::log(boundaries[segment]),
                std::log(boundaries[segment+1]));
            for (std::size_t iq=0;iq<radial.nodes.size();++iq) {
                const double radius=std::exp(radial.nodes[iq]);
                const eft::Vec3 loop=grid_scale(angle.direction,radius);
                const double weight=
                    radial.weights[iq]*radius*radius*radius
                    *angle.weight*measure;
                const GridIntegrands values=mapped.all(loop);
                GridUvComponents subtraction;
                if (radius<=config.qmax) {
                    subtraction=uv.integrands(loop);
                }
                output.B222+=weight*values.B222;
                output.B321I+=weight*values.B321I;
                output.B321II_bare+=weight*values.B321II;
                output.B411_bare+=weight*values.B411;
                output.B321II_uv_subtraction+=
                    weight*subtraction.B321II;
                output.B411_uv_subtraction+=weight*subtraction.B411;
                output.B321II+=weight*(
                    values.B321II-subtraction.B321II);
                output.B411+=weight*(values.B411-subtraction.B411);
                ++output.integration_nodes;
            }
        }
    }
    const GridUvComponents subleading=
        uv.angular_averaged_subleading_coefficients(
            config.uv_tail_n_mu,config.uv_tail_n_phi);
    const double tail_moment=eft::normalized_linear_power_uv_tail_moment(
        power,config.qmax,config.uv_tail_kmax,
        config.uv_tail_quadrature_order);
    output.B321II_uv_restoration=tail_moment*subleading.B321II;
    output.B411_uv_restoration=tail_moment*subleading.B411;
    output.B321II+=output.B321II_uv_restoration;
    output.B411+=output.B411_uv_restoration;
    return output;
}

void accumulate_grid_templates(
    GridTemplates& output,const GridTemplates& input,double weight) {
    output.tree+=weight*input.tree;
    output.B222+=weight*input.B222;
    output.B321I+=weight*input.B321I;
    output.B321II+=weight*input.B321II;
    output.B411+=weight*input.B411;
    output.B321II_bare+=weight*input.B321II_bare;
    output.B411_bare+=weight*input.B411_bare;
    output.B321II_uv_subtraction+=weight*input.B321II_uv_subtraction;
    output.B411_uv_subtraction+=weight*input.B411_uv_subtraction;
    output.B321II_uv_restoration+=weight*input.B321II_uv_restoration;
    output.B411_uv_restoration+=weight*input.B411_uv_restoration;
    output.integration_nodes+=input.integration_nodes;
}

struct GridShellTemplates {
    shell::ShellBin bin;
    GridTemplates diagrams;
    std::size_t shell_nodes=0;
    std::size_t invariant_shell_nodes=0;
    std::size_t orientation_nodes=0;
    std::size_t total_loop_nodes=0;
    std::uint64_t zero_external_leg_pairs=0;
    std::uint64_t closing_zero_pairs=0;
    double valid_pair_fraction=0.0;
};

GridShellTemplates compute_grid_shell_templates(
    const PowerSpectrum& power,
    const shell::ShellBin& bin,
    const GridMatterKernelProvider& provider,
    const eft::ReconstructedShellIntegrationConfig& integration) {
    const eft::HaarOrientedLatticeShellRule rule=
        eft::make_haar_oriented_lattice_shell_rule(bin,integration.shell);
    if (rule.nodes.empty()) {
        throw std::runtime_error("grid oracle shell has no projection nodes");
    }
    const int thread_count=std::max(
        1,std::min({8,omp_get_max_threads(),
                    static_cast<int>(rule.nodes.size())}));
    std::vector<GridTemplates> partial(
        static_cast<std::size_t>(thread_count));
#pragma omp parallel num_threads(thread_count)
    {
        const int thread=omp_get_thread_num();
#pragma omp for schedule(static)
        for (std::size_t index=0;index<rule.nodes.size();++index) {
            const shell::ShellNode& node=rule.nodes[index];
            const GridTemplates value=grid_evaluate_direct(
                power,node.closed_vectors,provider,integration.loop);
            accumulate_grid_templates(
                partial[static_cast<std::size_t>(thread)],
                value,node.weight);
        }
    }
    GridShellTemplates result;
    result.bin=bin;
    for (const GridTemplates& value:partial) {
        accumulate_grid_templates(result.diagrams,value,1.0);
    }
    result.shell_nodes=rule.nodes.size();
    result.invariant_shell_nodes=rule.diagnostics.invariant_nodes;
    result.orientation_nodes=rule.diagnostics.orientation_nodes;
    result.total_loop_nodes=result.diagrams.integration_nodes;
    result.zero_external_leg_pairs=
        rule.diagnostics.exact.zero_external_leg_pairs;
    result.closing_zero_pairs=rule.diagnostics.exact.closing_zero_pairs;
    result.valid_pair_fraction=rule.diagnostics.exact.valid_pair_fraction;
    return result;
}

eft::SparsePolynomial extract_marker_power_and_restore_b1_cubed(
    const eft::SparsePolynomial& polynomial,
    int requested_marker_power) {
    if (requested_marker_power<0) {
        throw std::invalid_argument("negative oracle marker power");
    }
    eft::SparsePolynomial selected;
    for (const auto& term:polynomial.terms()) {
        int marker_power=0;
        std::vector<eft::MonomialFactor> factors;
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==kOracleMarker) {
                marker_power=static_cast<int>(factor.power);
            } else {
                factors.push_back(factor);
            }
        }
        if (marker_power==requested_marker_power) {
            selected+=build_term(term.second,factors);
        }
    }
    const eft::SparsePolynomial b1=
        eft::SparsePolynomial::variable(eft::ParameterId::B1);
    return selected*b1*b1*b1;
}

eft::EftShellTemplates compute_tagged_oracle_shell_templates(
    const PowerSpectrum& power,
    const shell::ShellBin& bin,
    const eft::FieldKernelProvider& provider,
    const eft::ReconstructedShellIntegrationConfig& integration) {
    const eft::HaarOrientedLatticeShellRule rule=
        eft::make_haar_oriented_lattice_shell_rule(bin,integration.shell);
    if (rule.nodes.empty()) {
        throw std::runtime_error("tagged oracle shell has no projection nodes");
    }
    eft::EftShellTemplates result;
    result.bin=bin;
    result.shell_nodes=rule.nodes.size();
    result.diagrams.ir_safe=integration.loop.ir_safe;
    for (const shell::ShellNode& node:rule.nodes) {
        eft::DiagramTemplates value=
            eft::evaluate_direct(
                power,node.closed_vectors,provider,integration.loop);
        const eft::DiagramAssembler tree(
            power,node.closed_vectors,provider);
        value.tree=tree.tree();
        const double weight=node.weight;
        result.diagrams.tree+=weight*value.tree;
        result.diagrams.B222+=weight*value.B222;
        result.diagrams.B321I+=weight*value.B321I;
        result.diagrams.B321II+=weight*value.B321II;
        result.diagrams.B411+=weight*value.B411;
        result.diagrams.B321II_bare+=weight*value.B321II_bare;
        result.diagrams.B411_bare+=weight*value.B411_bare;
        result.diagrams.B321II_uv_subtraction+=
            weight*value.B321II_uv_subtraction;
        result.diagrams.B411_uv_subtraction+=
            weight*value.B411_uv_subtraction;
        result.diagrams.B321II_uv_restoration+=
            weight*value.B321II_uv_restoration;
        result.diagrams.B411_uv_restoration+=
            weight*value.B411_uv_restoration;
        result.total_loop_nodes+=value.integration_nodes;
    }
    result.diagrams.uv_subtraction_total=
        result.diagrams.B321II_uv_subtraction
        +result.diagrams.B411_uv_subtraction;
    result.diagrams.uv_restoration_total=
        result.diagrams.B321II_uv_restoration
        +result.diagrams.B411_uv_restoration;
    result.diagrams.one_loop=
        result.diagrams.B222+result.diagrams.B321I
        +result.diagrams.B321II+result.diagrams.B411;
    result.diagrams.total=
        result.diagrams.tree+result.diagrams.one_loop;
    result.diagrams.integration_nodes=result.total_loop_nodes;
    result.fft_lattice_radial_measure=true;
    result.exact_k1_k2_mu_lattice_measure=true;
    result.haar_oriented_cic_projection=true;
    result.invariant_shell_nodes=rule.diagnostics.invariant_nodes;
    result.orientation_nodes=rule.diagnostics.orientation_nodes;
    result.exact_lattice_radial_order=
        integration.shell.exact.radial_order;
    result.exact_lattice_angular_order=
        integration.shell.exact.angular_order;
    result.zero_external_leg_pairs=
        rule.diagnostics.exact.zero_external_leg_pairs;
    result.closing_zero_pairs=
        rule.diagnostics.exact.closing_zero_pairs;
    result.exact_lattice_valid_pair_fraction=
        rule.diagnostics.exact.valid_pair_fraction;
    result.exact_lattice_total_variation=
        rule.diagnostics.exact.total_variation;
    return result;
}

std::string file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot hash input: " + path);
    }
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return eft::sha256_hex(payload);
}

class TabulatedPower final : public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open power table: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double value = 0.0;
            if (!(row >> k >> value) || !(k > 0.0 && value > 0.0)) {
                throw std::runtime_error("invalid positive power-table row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(value));
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("power table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k > 0.0)) return 0.0;
        const double x = std::log(k);
        std::size_t right = 1;
        if (x >= log_k_.back()) {
            right = log_k_.size() - 1;
        } else if (x > log_k_.front()) {
            right = static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(), log_k_.end(), x)
                - log_k_.begin());
        }
        const std::size_t left = right - 1;
        const double fraction =
            (x - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_value_[left]
            + fraction * (log_value_[right] - log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("tabulated power has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

class TabulatedTransfer final : public PowerSpectrum {
public:
    explicit TabulatedTransfer(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open local-PNG table: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double power = 0.0;
            double ignored = 0.0;
            double transfer = 0.0;
            if (!(row >> k >> power >> ignored >> transfer)
                || !(k > 0.0 && transfer > 0.0)) {
                throw std::runtime_error("invalid local-PNG transfer row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(transfer));
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("local-PNG table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k > 0.0)) return 0.0;
        const double x = std::log(k);
        std::size_t right = 1;
        if (x >= log_k_.back()) {
            right = log_k_.size() - 1;
        } else if (x > log_k_.front()) {
            right = static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(), log_k_.end(), x)
                - log_k_.begin());
        }
        const std::size_t left = right - 1;
        const double fraction =
            (x - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_value_[left]
            + fraction * (log_value_[right] - log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("tabulated transfer has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

int parse_int(const char* text, const std::string& name, int minimum) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size() || value < minimum) {
        throw std::invalid_argument("invalid integer for " + name);
    }
    return value;
}

double parse_double(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid number for " + name);
    }
    return value;
}

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open edge file: " + path);
    }
    std::vector<double> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        double value = 0.0;
        if (!(row >> value)) {
            throw std::runtime_error("invalid edge row");
        }
        result.push_back(value);
    }
    if (result.size() != 16) {
        throw std::runtime_error("edge file must contain 16 values");
    }
    for (std::size_t index = 1; index < result.size(); ++index) {
        if (!(result[index] > result[index - 1])) {
            throw std::runtime_error("edges must be strictly increasing");
        }
    }
    return result;
}

void write_json_string(const std::string& value) {
    std::cout << '"';
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            std::cout << '\\' << static_cast<char>(character);
        } else if (character < 0x20) {
            const auto flags = std::cout.flags();
            const char fill = std::cout.fill();
            std::cout << "\\u" << std::hex << std::setw(4)
                      << std::setfill('0') << static_cast<int>(character);
            std::cout.flags(flags);
            std::cout.fill(fill);
        } else {
            std::cout << static_cast<char>(character);
        }
    }
    std::cout << '"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout << '{';
    bool first = true;
    for (const auto& term : polynomial.terms()) {
        if (!first) std::cout << ',';
        first = false;
        write_json_string(term.first.canonical_string());
        std::cout << ':' << term.second;
    }
    std::cout << '}';
}

struct LeadingBshotShellShapes {
    double ashot=0.0;
    eft::SparsePolynomial pre;
    eft::SparsePolynomial post;
};

LeadingBshotShellShapes leading_bshot_shell_shapes(
    const PowerSpectrum& power,
    const shell::ShellBin& bin,
    const eft::FieldKernelProvider& provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const eft::HaarOrientedLatticeShellRuleConfig& config) {
    const eft::HaarOrientedLatticeShellRule rule =
        eft::make_haar_oriented_lattice_shell_rule(bin, config);
    if (rule.nodes.empty()) {
        throw std::runtime_error("leading Bshot shell has no projection nodes");
    }
    LeadingBshotShellShapes result;
    const eft::SparsePolynomial b1_squared =
        eft::SparsePolynomial::variable(eft::ParameterId::B1)
        * eft::SparsePolynomial::variable(eft::ParameterId::B1);
    for (const shell::ShellNode& node : rule.nodes) {
        result.ashot+=node.weight;
        double power_sum = 0.0;
        for (const eft::Vec3& wavevector : node.closed_vectors) {
            power_sum += power(marisa_b_halo_v1::norm(wavevector));
        }
        result.pre += node.weight * power_sum * b1_squared;
        result.post += node.weight
                       * eft::reconstructed_leading_bshot_residual_generating_shape(
                           power,
                           node.closed_vectors,
                           provider,
                           reconstruction,
                           false);
    }
    return result;
}

void write_bin(
    int index,
    const eft::EftShellTemplates& value,
    const LeadingBshotShellShapes& leading_bshot) {
    std::cout << std::setprecision(17)
              << "{\"record\":\"bin\",\"index\":" << index
              << ",\"edges\":[" << value.bin.k1_lower << ','
              << value.bin.k1_upper << ',' << value.bin.k2_lower << ','
              << value.bin.k2_upper << ']'
              << ",\"shell_nodes\":" << value.shell_nodes
              << ",\"invariant_shell_nodes\":" << value.invariant_shell_nodes
              << ",\"orientation_nodes\":" << value.orientation_nodes
              << ",\"total_loop_nodes\":" << value.total_loop_nodes
              << ",\"zero_external_leg_pairs\":" << value.zero_external_leg_pairs
              << ",\"closing_zero_pairs\":" << value.closing_zero_pairs
              << ",\"valid_pair_fraction\":"
              << value.exact_lattice_valid_pair_fraction
              << ",\"tree\":";
    write_polynomial(value.diagrams.tree);
    std::cout << ",\"B222\":";
    write_polynomial(value.diagrams.B222);
    std::cout << ",\"B321I\":";
    write_polynomial(value.diagrams.B321I);
    std::cout << ",\"B321II\":";
    write_polynomial(value.diagrams.B321II);
    std::cout << ",\"B411\":";
    write_polynomial(value.diagrams.B411);
    std::cout << ",\"B321II_bare\":";
    write_polynomial(value.diagrams.B321II_bare);
    std::cout << ",\"B411_bare\":";
    write_polynomial(value.diagrams.B411_bare);
    std::cout << ",\"B321II_uv_subtraction\":";
    write_polynomial(value.diagrams.B321II_uv_subtraction);
    std::cout << ",\"B411_uv_subtraction\":";
    write_polynomial(value.diagrams.B411_uv_subtraction);
    std::cout << ",\"B321II_uv_restoration\":";
    write_polynomial(value.diagrams.B321II_uv_restoration);
    std::cout << ",\"B411_uv_restoration\":";
    write_polynomial(value.diagrams.B411_uv_restoration);
    std::cout << ",\"Ashot_residual\":";
    write_polynomial(
        eft::SparsePolynomial::constant(leading_bshot.ashot));
    std::cout << ",\"Bshot_residual\":";
    write_polynomial(leading_bshot.post);
    std::cout << ",\"pre_Bshot_residual\":";
    write_polynomial(leading_bshot.pre);
    std::cout << "}\n";
    std::cout.flush();
}

void write_tagged_pair(
    const std::string& name,
    const eft::SparsePolynomial& tagged) {
    std::cout << ',';
    write_json_string(name+"_zero");
    std::cout << ':';
    write_polynomial(
        extract_marker_power_and_restore_b1_cubed(tagged,0));
    std::cout << ',';
    write_json_string(name+"_direction");
    std::cout << ':';
    write_polynomial(
        extract_marker_power_and_restore_b1_cubed(tagged,1));
}

void write_analytic_bin(
    int index,
    const eft::EftShellTemplates& value,
    const LeadingBshotShellShapes& leading_bshot) {
    std::cout << std::setprecision(17)
              << "{\"record\":\"bin\",\"index\":" << index
              << ",\"edges\":[" << value.bin.k1_lower << ','
              << value.bin.k1_upper << ',' << value.bin.k2_lower << ','
              << value.bin.k2_upper << ']'
              << ",\"shell_nodes\":" << value.shell_nodes
              << ",\"invariant_shell_nodes\":" << value.invariant_shell_nodes
              << ",\"orientation_nodes\":" << value.orientation_nodes
              << ",\"total_loop_nodes\":" << value.total_loop_nodes
              << ",\"zero_external_leg_pairs\":" << value.zero_external_leg_pairs
              << ",\"closing_zero_pairs\":" << value.closing_zero_pairs
              << ",\"valid_pair_fraction\":"
              << value.exact_lattice_valid_pair_fraction;
    write_tagged_pair("tree",value.diagrams.tree);
    write_tagged_pair("B222",value.diagrams.B222);
    write_tagged_pair("B321I",value.diagrams.B321I);
    write_tagged_pair("B321II",value.diagrams.B321II);
    write_tagged_pair("B411",value.diagrams.B411);
    write_tagged_pair(
        "B321II_bare",value.diagrams.B321II_bare);
    write_tagged_pair(
        "B411_bare",value.diagrams.B411_bare);
    write_tagged_pair(
        "B321II_uv_subtraction",
        value.diagrams.B321II_uv_subtraction);
    write_tagged_pair(
        "B411_uv_subtraction",
        value.diagrams.B411_uv_subtraction);
    write_tagged_pair(
        "B321II_uv_restoration",
        value.diagrams.B321II_uv_restoration);
    write_tagged_pair(
        "B411_uv_restoration",
        value.diagrams.B411_uv_restoration);
    std::cout << ",\"Ashot_residual_zero\":";
    write_polynomial(
        eft::SparsePolynomial::constant(leading_bshot.ashot));
    std::cout << ",\"Ashot_residual_direction\":";
    write_polynomial(eft::SparsePolynomial());
    std::cout << ",\"Bshot_residual_zero\":";
    write_polynomial(leading_bshot.post);
    std::cout << ",\"Bshot_residual_direction\":";
    write_polynomial(eft::SparsePolynomial());
    std::cout << ",\"pre_Bshot_residual_zero\":";
    write_polynomial(leading_bshot.pre);
    std::cout << "}\n";
    std::cout.flush();
}

void write_grid_array(const std::array<double,kGridB1.size()>& values) {
    std::cout<<'[';
    for (std::size_t index=0;index<values.size();++index) {
        if (index!=0) std::cout<<',';
        std::cout<<values[index];
    }
    std::cout<<']';
}

std::array<double,kGridB1.size()> restored_grid_values(
    const GridDual& value,bool direction) {
    std::array<double,kGridB1.size()> result{};
    const auto& source=direction?value.direction:value.value;
    for (std::size_t index=0;index<kGridB1.size();++index) {
        result[index]=source[index]
                      *kGridB1[index]*kGridB1[index]*kGridB1[index];
    }
    return result;
}

void write_grid_pair(const std::string& name,const GridDual& value) {
    std::cout<<',';
    write_json_string(name+"_zero_grid");
    std::cout<<':';
    write_grid_array(restored_grid_values(value,false));
    std::cout<<',';
    write_json_string(name+"_direction_grid");
    std::cout<<':';
    write_grid_array(restored_grid_values(value,true));
}

std::array<double,kGridB1.size()> evaluate_grid_polynomial(
    const eft::SparsePolynomial& polynomial) {
    std::array<double,kGridB1.size()> result{};
    for (std::size_t index=0;index<kGridB1.size();++index) {
        std::array<double,eft::kParameterCount> parameters{};
        parameters[static_cast<std::size_t>(eft::ParameterId::B1)]=
            kGridB1[index];
        result[index]=polynomial.evaluate(parameters);
    }
    return result;
}

void write_grid_bin(
    int index,
    const GridShellTemplates& value,
    const LeadingBshotShellShapes& leading_bshot) {
    std::cout<<std::setprecision(17)
             <<"{\"record\":\"bin\",\"index\":"<<index
             <<",\"edges\":["<<value.bin.k1_lower<<','
             <<value.bin.k1_upper<<','<<value.bin.k2_lower<<','
             <<value.bin.k2_upper<<']'
             <<",\"shell_nodes\":"<<value.shell_nodes
             <<",\"invariant_shell_nodes\":"<<value.invariant_shell_nodes
             <<",\"orientation_nodes\":"<<value.orientation_nodes
             <<",\"total_loop_nodes\":"<<value.total_loop_nodes
             <<",\"zero_external_leg_pairs\":"
             <<value.zero_external_leg_pairs
             <<",\"closing_zero_pairs\":"<<value.closing_zero_pairs
             <<",\"valid_pair_fraction\":"<<value.valid_pair_fraction;
    write_grid_pair("tree",value.diagrams.tree);
    write_grid_pair("B222",value.diagrams.B222);
    write_grid_pair("B321I",value.diagrams.B321I);
    write_grid_pair("B321II",value.diagrams.B321II);
    write_grid_pair("B411",value.diagrams.B411);
    write_grid_pair("B321II_bare",value.diagrams.B321II_bare);
    write_grid_pair("B411_bare",value.diagrams.B411_bare);
    write_grid_pair(
        "B321II_uv_subtraction",
        value.diagrams.B321II_uv_subtraction);
    write_grid_pair(
        "B411_uv_subtraction",
        value.diagrams.B411_uv_subtraction);
    write_grid_pair(
        "B321II_uv_restoration",
        value.diagrams.B321II_uv_restoration);
    write_grid_pair(
        "B411_uv_restoration",
        value.diagrams.B411_uv_restoration);
    std::array<double,kGridB1.size()> ashot{};
    ashot.fill(leading_bshot.ashot);
    std::array<double,kGridB1.size()> zeros{};
    std::cout<<",\"Ashot_residual_zero_grid\":";
    write_grid_array(ashot);
    std::cout<<",\"Ashot_residual_direction_grid\":";
    write_grid_array(zeros);
    std::cout<<",\"Bshot_residual_zero_grid\":";
    write_grid_array(evaluate_grid_polynomial(leading_bshot.post));
    std::cout<<",\"Bshot_residual_direction_grid\":";
    write_grid_array(zeros);
    std::cout<<",\"pre_Bshot_residual_zero_grid\":";
    write_grid_array(evaluate_grid_polynomial(leading_bshot.pre));
    std::cout<<"}\n";
    std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc!=7 &&argc!=8) {
            std::cerr
                << "usage: marisa_b_v0p9_oracle_matter_driver "
                << "POWER_TABLE EDGE_FILE PNG_TABLE START_BIN STOP_BIN "
                << "FNL_REC_OR_ANALYTIC [RADIAL_STOP]\n";
            return 2;
        }
        const std::string power_path = argv[1];
        const std::string edge_path = argv[2];
        const std::string png_path = argv[3];
        const int start = parse_int(argv[4], "start bin", 0);
        const int stop = parse_int(argv[5], "stop bin", 1);
        const bool analytic_grid=std::string(argv[6])=="analytic-grid";
        const bool analytic_symbolic=std::string(argv[6])=="analytic";
        const bool analytic_direction=analytic_symbolic ||analytic_grid;
        const double fnl_rec=analytic_direction
            ?0.0:parse_double(argv[6], "fNL_rec");
        const int radial_stop=argc==8
            ?parse_int(argv[7], "radial stop", 1)
            :kDefaultRadialStop;
        if (start < 0 || stop <= start || stop > 120) {
            throw std::invalid_argument("invalid bin range");
        }
        if (radial_stop < 1 || radial_stop > 15) {
            throw std::invalid_argument("invalid radial stop");
        }

        eft::ReconstructedShellIntegrationConfig integration;
        integration.shell.exact.radial_order = 2;
        integration.shell.exact.angular_order = 8;
        integration.shell.exact.fft_box_size = kBoxSize;
        integration.shell.exact.fft_mesh_size = kMeshSize;
        integration.shell.n_alpha = 2;
        integration.shell.n_cos_beta = 2;
        integration.shell.n_gamma = 2;
        integration.loop.n_radial = 8;
        integration.loop.n_mu = 32;
        integration.loop.n_phi = 24;
        integration.loop.qmin = 0.0001;
        integration.loop.qmax = 6.4;
        integration.loop.restore_uv_tail = true;
        integration.loop.uv_tail_kmax = 30.0;
        integration.loop.ir_safe = true;
        integration.loop.uv_subtract = true;
        integration.loop.exploit_phi_reflection = false;
        integration.loop.angular_rule = eft::DirectAngularRule::AntipodalFibonacci;
        integration.include_reconstructed_stochastic = false;
        integration.include_reconstructed_fixed_poisson = false;
        integration.include_bare_fixed_poisson_one_loop = false;

        eft::StochasticIntegrationConfig stochastic;
        stochastic.qmin = integration.loop.qmin;
        stochastic.qmax = integration.loop.qmax;
        stochastic.n_radial = integration.loop.n_radial;
        stochastic.n_mu = integration.loop.n_mu;
        stochastic.n_phi = integration.loop.n_phi;
        stochastic.restore_p13_uv_tail = true;
        stochastic.uv_tail_kmax = integration.loop.uv_tail_kmax;

        const TabulatedPower power(power_path);
        const TabulatedTransfer transfer(png_path);
        const std::vector<double> edges = read_edges(edge_path);
        marisa_b_halo_v1::ReconstructionConfig reconstruction;
        reconstruction.enabled = true;
        reconstruction.smoothing_radius = kSmoothingRadius;
        reconstruction.bias_recon = kHaloReconstructionBias;
        reconstruction.cell_size = kCellSize;
        const marisa_b_halo_v1::ReconstructionConfig fixed_reconstruction=
            reconstruction;
        if (!analytic_direction) {
            reconstruction.local_png_bias_transfer = &transfer;
            reconstruction.local_png_bias_amplitude =
                fnl_rec * kHaloReconstructionPhiBias;
            reconstruction.local_png_bias_kmin = kReconstructionMinimumK;
        }

        const LinearBiasMatterKernelProvider matter;
        const TaggedOracleMatterKernelProvider tagged(
            matter,fixed_reconstruction,transfer);
        const GridMatterKernelProvider grid_provider(
            fixed_reconstruction,transfer);
        std::cout << std::setprecision(17)
                  << "{\"record\":\"header\","
                  << "\"schema\":\""
                  << (analytic_grid
                      ?"marisa-b-v0p9-oracle-b1-grid-dual-v1"
                      :analytic_symbolic
                      ?"marisa-b-v0p9-oracle-pure-b1-analytic-v1"
                      :"marisa-b-v0p9-oracle-pure-b1-finite-v2")
                  << "\","
                  << "\"model\":\"pure-linear-bias matter SPT tree plus one-loop and selected stochastic\","
                  << "\"production_candidate\":false,"
                  << "\"bin_range\":[" << start << ',' << stop << "],"
                  << "\"radial_stop\":" << radial_stop << ','
                  << "\"fnl_rec\":" << fnl_rec << ','
                  << "\"analytic_direction\":"
                  << (analytic_direction?"true":"false") << ','
                  << "\"analytic_grid\":"
                  << (analytic_grid?"true":"false") << ','
                  << "\"response_carrier\":\""
                  << (analytic_grid?"numeric_dual":"gamma31") << "\","
                  << "\"gamma31_is_auxiliary_marker\":"
                  << (analytic_symbolic?"true":"false") << ','
                  << "\"per_kernel_common_b1_factored\":"
                  << (analytic_direction?"true":"false") << ','
                  << "\"b1_probes\":[1,2,3,4],"
                  << "\"maximum_cpu_threads\":8,"
                  << "\"b_rec_h\":" << reconstruction.bias_recon << ','
                  << "\"bphi_rec\":" << kHaloReconstructionPhiBias << ','
                  << "\"kmin\":" << kReconstructionMinimumK << ','
                  << "\"shell_orders\":[2,8,2,2,2],"
                  << "\"loop_orders\":[8,32,24],"
                  << "\"pure_b1_polynomial\":true,"
                  << "\"selected_stochastic\":[\"Ashot_residual\",\"Bshot_residual\"],"
                  << "\"q_range\":[0.0001,6.4],"
                  << "\"uv_tail_kmax\":30,"
                  << "\"source_hashes\":{"
                  << "\"linear_power\":\"" << file_sha256(power_path) << "\","
                  << "\"edge_file\":\"" << file_sha256(edge_path) << "\","
                  << "\"png_table\":\"" << file_sha256(png_path) << "\","
                  << "\"driver_executable\":\"" << file_sha256(argv[0]) << "\"}"
                  << "}\n";
        std::cout.flush();

        int flat = 0;
        for (int first = 0; first < 15; ++first) {
            for (int second = first; second < 15; ++second, ++flat) {
                if (flat < start || flat >= stop) continue;
                if (first >= radial_stop || second >= radial_stop) continue;
                const shell::ShellBin bin{
                    edges[static_cast<std::size_t>(first)],
                    edges[static_cast<std::size_t>(first + 1)],
                    edges[static_cast<std::size_t>(second)],
                    edges[static_cast<std::size_t>(second + 1)]};
                const LeadingBshotShellShapes leading_bshot =
                    leading_bshot_shell_shapes(
                        power,bin,matter,reconstruction,integration.shell);
                if (analytic_grid) {
                    const GridShellTemplates value=
                        compute_grid_shell_templates(
                            power,bin,grid_provider,integration);
                    write_grid_bin(flat,value,leading_bshot);
                } else if (analytic_symbolic) {
                    const eft::EftShellTemplates value=
                        compute_tagged_oracle_shell_templates(
                            power,bin,tagged,integration);
                    write_analytic_bin(flat,value,leading_bshot);
                } else {
                    const eft::EftShellTemplates value=
                        eft::compute_reconstructed_eft_shell_templates(
                            power,power,bin,matter,reconstruction,
                            integration,stochastic,0.30);
                    write_bin(flat, value, leading_bshot);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0.9 oracle matter driver failed: " << error.what() << '\n';
        return 1;
    }
}
