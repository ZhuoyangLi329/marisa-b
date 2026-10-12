#include "poisson_reconstruction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "PowerSpectrum.h"

namespace marisa_b_eft_v2 {
namespace {

constexpr double kPi=3.141592653589793238462643383279502884;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(
    int count,double lower,double upper) {
    if (count<1 ||!(upper>lower)) {
        throw std::invalid_argument(
            "invalid fixed-Poisson quadrature");
    }
    QuadratureRule result;
    result.nodes.resize(static_cast<std::size_t>(count));
    result.weights.resize(static_cast<std::size_t>(count));
    const int half=(count+1)/2;
    const double midpoint=0.5*(lower+upper);
    const double half_width=0.5*(upper-lower);
    for (int index=0;index<half;++index) {
        double root=std::cos(
            kPi*(index+0.75)/(count+0.5));
        double derivative=0.0;
        for (int iteration=0;iteration<100;++iteration) {
            double previous=1.0;
            double current=root;
            for (int order=2;order<=count;++order) {
                const double next=
                    ((2.0*order-1.0)*root*current
                     -(order-1.0)*previous)/order;
                previous=current;
                current=next;
            }
            const double pn=count==1?root:current;
            const double pnm1=count==1?1.0:previous;
            derivative=count*(root*pn-pnm1)
                       /(root*root-1.0);
            const double update=pn/derivative;
            root-=update;
            if (std::fabs(update)
                <4.0*std::numeric_limits<double>::epsilon()) {
                break;
            }
        }
        const double weight=
            2.0/((1.0-root*root)
                 *derivative*derivative);
        result.nodes[static_cast<std::size_t>(index)]=
            midpoint-half_width*root;
        result.nodes[
            static_cast<std::size_t>(count-1-index)]=
            midpoint+half_width*root;
        result.weights[static_cast<std::size_t>(index)]=
            half_width*weight;
        result.weights[
            static_cast<std::size_t>(count-1-index)]=
            half_width*weight;
    }
    return result;
}

Vec3 scaled(const Vec3& value,double factor) {
    return Vec3{
        factor*value.x,
        factor*value.y,
        factor*value.z};
}

class UnitPoissonMarkedProvider final:
    public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>& matter_momenta,
        const Vec3&) const override {
        if (!matter_momenta.empty()) return {};
        return SparsePolynomial::constant(1.0);
    }
    std::string_view name() const noexcept override {
        return "unit_local_poisson_K0";
    }
};

class ZeroPoissonMarkedProvider final:
    public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>&,
        const Vec3&) const override {
        return {};
    }
    std::string_view name() const noexcept override {
        return "zero_local_poisson_shift";
    }
};

struct LegAddress {
    int field=0;
    int local=0;
    int decoration_block=-1;
};

struct ContractionTopology {
    int target_order=0;
    int noise_cost=0;
    std::array<int,3> matter_count={};
    std::array<int,3> stochastic_count={};
    std::vector<LegAddress> matter_legs;
    std::vector<LegAddress> stochastic_legs;
    std::vector<int> decoration_matter_count;
    std::vector<std::vector<int>>
        decoration_matter_legs;
    std::vector<std::pair<int,int>> matter_pairs;
    std::vector<std::vector<int>> stochastic_blocks;
    std::vector<std::vector<int>> matter_coefficients;
    std::vector<std::vector<int>> stochastic_coefficients;
    std::array<std::vector<int>,3> external_coefficients;
    std::array<int,2> pivot_rows={{-1,-1}};
    std::array<int,2> pivot_columns={{-1,-1}};
    int free_column=-1;
    double jacobian=0.0;
};

void generate_pairings_recursive(
    const std::vector<int>& remaining,
    std::vector<std::pair<int,int>>& current,
    std::vector<std::vector<std::pair<int,int>>>& output) {
    if (remaining.empty()) {
        output.push_back(current);
        return;
    }
    const int first=remaining.front();
    for (std::size_t partner=1;
         partner<remaining.size();
         ++partner) {
        std::vector<int> next;
        next.reserve(remaining.size()-2);
        for (std::size_t index=1;
             index<remaining.size();
             ++index) {
            if (index!=partner) {
                next.push_back(remaining[index]);
            }
        }
        current.emplace_back(first,remaining[partner]);
        generate_pairings_recursive(
            next,current,output);
        current.pop_back();
    }
}

std::vector<std::vector<std::pair<int,int>>>
generate_pairings(int count) {
    if (count%2!=0) return {};
    std::vector<int> remaining(
        static_cast<std::size_t>(count));
    for (int index=0;index<count;++index) {
        remaining[static_cast<std::size_t>(index)]=index;
    }
    std::vector<std::pair<int,int>> current;
    std::vector<std::vector<std::pair<int,int>>> output;
    generate_pairings_recursive(
        remaining,current,output);
    return output;
}

void generate_noise_partitions_recursive(
    int next,int count,
    std::vector<std::vector<int>>& blocks,
    std::vector<std::vector<std::vector<int>>>& output) {
    if (next==count) {
        bool valid=!blocks.empty();
        for (const auto& block:blocks) {
            valid=valid
                &&block.size()>=2
                &&block.size()<=4;
        }
        if (valid) output.push_back(blocks);
        return;
    }
    for (std::size_t block=0;
         block<blocks.size();
         ++block) {
        if (blocks[block].size()>=4) continue;
        blocks[block].push_back(next);
        generate_noise_partitions_recursive(
            next+1,count,blocks,output);
        blocks[block].pop_back();
    }
    blocks.push_back({next});
    generate_noise_partitions_recursive(
        next+1,count,blocks,output);
    blocks.pop_back();
}

std::vector<std::vector<std::vector<int>>>
generate_noise_partitions(int count) {
    std::vector<std::vector<int>> blocks;
    std::vector<std::vector<std::vector<int>>> output;
    generate_noise_partitions_recursive(
        0,count,blocks,output);
    return output;
}

bool connected_external_graph(
    const ContractionTopology& topology) {
    const int node_count=
        3+static_cast<int>(
            topology.stochastic_blocks.size());
    std::vector<int> parent(
        static_cast<std::size_t>(node_count));
    for (int node=0;node<node_count;++node) {
        parent[static_cast<std::size_t>(node)]=node;
    }
    const auto root=[&parent](int node) {
        while (parent[static_cast<std::size_t>(node)]
               !=node) {
            node=parent[static_cast<std::size_t>(node)];
        }
        return node;
    };
    const auto unite=[&](int left,int right) {
        const int left_root=root(left);
        const int right_root=root(right);
        if (left_root!=right_root) {
            parent[static_cast<std::size_t>(
                right_root)]=left_root;
        }
    };
    const auto matter_node=[](
        const LegAddress& address) {
        return address.field>=0
            ?address.field
            :3+address.decoration_block;
    };
    for (const auto& pair:topology.matter_pairs) {
        unite(
            matter_node(
                topology.matter_legs[
                    static_cast<std::size_t>(
                        pair.first)]),
            matter_node(
                topology.matter_legs[
                    static_cast<std::size_t>(
                        pair.second)]));
    }
    for (std::size_t block_index=0;
         block_index<topology.stochastic_blocks.size();
         ++block_index) {
        const auto& block=
            topology.stochastic_blocks[block_index];
        const int cumulant_node=
            3+static_cast<int>(block_index);
        for (int leg:block) {
            unite(
                cumulant_node,
                topology.stochastic_legs[
                    static_cast<std::size_t>(leg)].field);
        }
    }
    return root(0)==root(1)&&root(0)==root(2);
}

bool choose_constraint_solution(
    ContractionTopology& topology) {
    const int variables=topology.target_order;
    double best_determinant=0.0;
    for (int first_row=0;
         first_row<3;
         ++first_row) {
        for (int second_row=first_row+1;
             second_row<3;
             ++second_row) {
            for (int free_column=-1;
                 free_column<variables;
                 ++free_column) {
                if ((variables==2&&free_column!=-1)
                    ||(variables==3&&free_column<0)) {
                    continue;
                }
                std::array<int,2> pivots={{-1,-1}};
                int found=0;
                for (int column=0;
                     column<variables;
                     ++column) {
                    if (column==free_column) continue;
                    pivots[static_cast<std::size_t>(found++)]=
                        column;
                }
                if (found!=2) continue;
                const auto& A=topology.external_coefficients;
                const double determinant=
                    static_cast<double>(
                        A[static_cast<std::size_t>(first_row)]
                         [static_cast<std::size_t>(pivots[0])]
                        *A[static_cast<std::size_t>(second_row)]
                          [static_cast<std::size_t>(pivots[1])]
                        -A[static_cast<std::size_t>(first_row)]
                          [static_cast<std::size_t>(pivots[1])]
                         *A[static_cast<std::size_t>(second_row)]
                           [static_cast<std::size_t>(pivots[0])]);
                if (std::fabs(determinant)
                    >std::fabs(best_determinant)) {
                    best_determinant=determinant;
                    topology.pivot_rows={{
                        first_row,second_row}};
                    topology.pivot_columns=pivots;
                    topology.free_column=free_column;
                }
            }
        }
    }
    if (best_determinant==0.0) return false;
    topology.jacobian=
        1.0/std::pow(std::fabs(best_determinant),3);
    return true;
}

bool finalize_topology(
    ContractionTopology& topology) {
    const int variables=topology.target_order;
    topology.matter_coefficients.assign(
        topology.matter_legs.size(),
        std::vector<int>(
            static_cast<std::size_t>(variables),0));
    topology.stochastic_coefficients.assign(
        topology.stochastic_legs.size(),
        std::vector<int>(
            static_cast<std::size_t>(variables),0));
    int variable=0;
    for (const auto& pair:topology.matter_pairs) {
        topology.matter_coefficients[
            static_cast<std::size_t>(pair.first)]
            [static_cast<std::size_t>(variable)]=1;
        topology.matter_coefficients[
            static_cast<std::size_t>(pair.second)]
            [static_cast<std::size_t>(variable)]=-1;
        ++variable;
    }
    for (std::size_t block_index=0;
         block_index<topology.stochastic_blocks.size();
         ++block_index) {
        const auto& block=
            topology.stochastic_blocks[block_index];
        for (std::size_t index=0;
             index+1<block.size();
             ++index) {
            topology.stochastic_coefficients[
                static_cast<std::size_t>(block[index])]
                [static_cast<std::size_t>(variable)]=1;
            topology.stochastic_coefficients[
                static_cast<std::size_t>(block.back())]
                [static_cast<std::size_t>(variable)]=-1;
            ++variable;
        }
        /*
         * A conditional Poisson cumulant is proportional to
         * [1+delta_h] at the coincident point.  Momentum conservation at a
         * decorated cumulant in the Fourier convention used here is
         *
         *   sum(noise momenta)-sum(decoration matter momenta)=0.
         *
         * The independent matter-pair coefficients store the decoration
         * momentum itself.  Solving the equation for the last noise leg
         * therefore adds those coefficients to that leg.  Using the
         * opposite sign violates global external-triangle closure and
         * silently discards otherwise connected decorated topologies.
         */
        for (int decoration_leg:
             topology.decoration_matter_legs[
                 block_index]) {
            for (int column=0;
                 column<variables;
                 ++column) {
                topology.stochastic_coefficients[
                    static_cast<std::size_t>(
                        block.back())]
                    [static_cast<std::size_t>(column)]
                    +=topology.matter_coefficients[
                        static_cast<std::size_t>(
                            decoration_leg)]
                      [static_cast<std::size_t>(column)];
            }
        }
    }
    if (variable!=variables) {
        throw std::logic_error(
            "fixed-Poisson topology has inconsistent power order");
    }
    for (auto& row:topology.external_coefficients) {
        row.assign(
            static_cast<std::size_t>(variables),0);
    }
    for (std::size_t leg=0;
         leg<topology.matter_legs.size();
         ++leg) {
        const int field=topology.matter_legs[leg].field;
        if (field<0) continue;
        for (int column=0;column<variables;++column) {
            topology.external_coefficients[
                static_cast<std::size_t>(field)]
                [static_cast<std::size_t>(column)]
                +=topology.matter_coefficients[leg]
                  [static_cast<std::size_t>(column)];
        }
    }
    for (std::size_t leg=0;
         leg<topology.stochastic_legs.size();
         ++leg) {
        const int field=topology.stochastic_legs[leg].field;
        for (int column=0;column<variables;++column) {
            topology.external_coefficients[
                static_cast<std::size_t>(field)]
                [static_cast<std::size_t>(column)]
                +=topology.stochastic_coefficients[leg]
                  [static_cast<std::size_t>(column)];
        }
    }
    return connected_external_graph(topology)
        &&choose_constraint_solution(topology);
}

void generate_decoration_orders_recursive(
    std::size_t block,
    std::size_t block_count,
    int remaining_matter,
    bool conditional,
    std::vector<int>& current,
    std::vector<std::vector<int>>& output) {
    if (block==block_count) {
        output.push_back(current);
        return;
    }
    const int maximum=conditional
        ?std::min(4,remaining_matter)
        :0;
    for (int order=0;order<=maximum;++order) {
        current.push_back(order);
        generate_decoration_orders_recursive(
            block+1,block_count,
            remaining_matter-order,
            conditional,current,output);
        current.pop_back();
    }
}

std::vector<std::vector<int>>
generate_decoration_orders(
    std::size_t block_count,
    int remaining_matter,
    bool conditional) {
    std::vector<int> current;
    std::vector<std::vector<int>> output;
    generate_decoration_orders_recursive(
        0,block_count,remaining_matter,
        conditional,current,output);
    return output;
}

std::vector<ContractionTopology>
generate_topologies(
    int target_order,
    PoissonIntensityMode intensity_mode) {
    const bool conditional=
        intensity_mode
        ==PoissonIntensityMode::ConditionalTracer;
    std::vector<ContractionTopology> output;
    for (int m0=0;m0<=4;++m0) {
        for (int n0=0;n0<=4-m0;++n0) {
            if (m0+n0==0) continue;
            for (int m1=0;m1<=4;++m1) {
                for (int n1=0;n1<=4-m1;++n1) {
                    if (m1+n1==0) continue;
                    for (int m2=0;m2<=4;++m2) {
                        for (int n2=0;n2<=4-m2;++n2) {
                            if (m2+n2==0) continue;
                            const std::array<int,3> matter={{
                                m0,m1,m2}};
                            const std::array<int,3> noise={{
                                n0,n1,n2}};
                            const int external_matter=m0+m1+m2;
                            const int total_noise=n0+n1+n2;
                            if (external_matter>2*target_order
                                ||total_noise<2
                                ||total_noise>2*target_order) {
                                continue;
                            }
                            for (const auto& blocks:
                                 generate_noise_partitions(
                                     total_noise)) {
                                int noise_cost=0;
                                for (const auto& block:blocks) {
                                    noise_cost+=
                                        static_cast<int>(
                                            block.size())-1;
                                }
                                if (noise_cost>target_order) {
                                    continue;
                                }
                                const int maximum_matter=
                                    2*(target_order-noise_cost);
                                if (external_matter
                                    >maximum_matter) {
                                    continue;
                                }
                                for (const auto& decorations:
                                     generate_decoration_orders(
                                         blocks.size(),
                                         maximum_matter
                                             -external_matter,
                                         conditional)) {
                                    int total_matter=
                                        external_matter;
                                    for (int order:decorations) {
                                        total_matter+=order;
                                    }
                                    if (total_matter%2!=0
                                        ||total_matter/2
                                              +noise_cost
                                          !=target_order) {
                                        continue;
                                    }
                                    const auto pairings=
                                        generate_pairings(
                                            total_matter);
                                    for (const auto& pairing:
                                         pairings) {
                                    ContractionTopology topology;
                                    topology.target_order=
                                        target_order;
                                    topology.noise_cost=
                                        noise_cost;
                                    topology.matter_count=
                                        matter;
                                    topology.stochastic_count=
                                        noise;
                                    topology.decoration_matter_count=
                                        decorations;
                                    topology.decoration_matter_legs.resize(
                                        blocks.size());
                                    for (int field=0;
                                         field<3;
                                         ++field) {
                                        for (int local=0;
                                             local<matter[
                                                static_cast<std::size_t>(
                                                    field)];
                                             ++local) {
                                            topology.matter_legs.push_back(
                                                {field,local,-1});
                                        }
                                        for (int local=0;
                                             local<noise[
                                                static_cast<std::size_t>(
                                                    field)];
                                             ++local) {
                                            topology.stochastic_legs.push_back(
                                                {field,local,-1});
                                        }
                                    }
                                    for (std::size_t block=0;
                                         block<decorations.size();
                                         ++block) {
                                        for (int local=0;
                                             local<decorations[block];
                                             ++local) {
                                            const int leg=
                                                static_cast<int>(
                                                    topology.matter_legs
                                                        .size());
                                            topology.matter_legs.push_back(
                                                {-1,local,
                                                 static_cast<int>(
                                                     block)});
                                            topology
                                                .decoration_matter_legs[
                                                    block]
                                                .push_back(leg);
                                        }
                                    }
                                    topology.matter_pairs=
                                        pairing;
                                    topology.stochastic_blocks=
                                        blocks;
                                    if (finalize_topology(topology)) {
                                        output.push_back(
                                            std::move(topology));
                                    }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}

Vec3 linear_combination(
    const std::vector<int>& coefficients,
    const std::vector<Vec3>& variables) {
    Vec3 result{};
    for (std::size_t index=0;
         index<coefficients.size();
         ++index) {
        result=marisa_b_halo_v1::add(
            result,
            scaled(
                variables[index],
                static_cast<double>(
                    coefficients[index])));
    }
    return result;
}

bool solve_variables(
    const ContractionTopology& topology,
    const std::array<Vec3,3>& external,
    const Vec3& free_value,
    std::vector<Vec3>& variables) {
    variables.assign(
        static_cast<std::size_t>(
            topology.target_order),Vec3{});
    if (topology.free_column>=0) {
        variables[static_cast<std::size_t>(
            topology.free_column)]=free_value;
    }
    const int r0=topology.pivot_rows[0];
    const int r1=topology.pivot_rows[1];
    const int c0=topology.pivot_columns[0];
    const int c1=topology.pivot_columns[1];
    const auto& A=topology.external_coefficients;
    Vec3 rhs0=external[static_cast<std::size_t>(r0)];
    Vec3 rhs1=external[static_cast<std::size_t>(r1)];
    if (topology.free_column>=0) {
        rhs0=marisa_b_halo_v1::subtract(
            rhs0,scaled(
                free_value,
                static_cast<double>(
                    A[static_cast<std::size_t>(r0)]
                     [static_cast<std::size_t>(
                         topology.free_column)])));
        rhs1=marisa_b_halo_v1::subtract(
            rhs1,scaled(
                free_value,
                static_cast<double>(
                    A[static_cast<std::size_t>(r1)]
                     [static_cast<std::size_t>(
                         topology.free_column)])));
    }
    const double a00=
        A[static_cast<std::size_t>(r0)]
         [static_cast<std::size_t>(c0)];
    const double a01=
        A[static_cast<std::size_t>(r0)]
         [static_cast<std::size_t>(c1)];
    const double a10=
        A[static_cast<std::size_t>(r1)]
         [static_cast<std::size_t>(c0)];
    const double a11=
        A[static_cast<std::size_t>(r1)]
         [static_cast<std::size_t>(c1)];
    const double determinant=a00*a11-a01*a10;
    if (determinant==0.0) return false;
    variables[static_cast<std::size_t>(c0)]=
        scaled(
            marisa_b_halo_v1::subtract(
                scaled(rhs0,a11),
                scaled(rhs1,a01)),
            1.0/determinant);
    variables[static_cast<std::size_t>(c1)]=
        scaled(
            marisa_b_halo_v1::subtract(
                scaled(rhs1,a00),
                scaled(rhs0,a10)),
            1.0/determinant);

    double scale_external=1.0;
    for (const Vec3& value:external) {
        scale_external=std::max(
            scale_external,
            marisa_b_halo_v1::norm(value));
    }
    for (int field=0;field<3;++field) {
        const Vec3 reconstructed=
            linear_combination(
                A[static_cast<std::size_t>(field)],
                variables);
        if (marisa_b_halo_v1::norm(
                marisa_b_halo_v1::subtract(
                    reconstructed,
                    external[
                        static_cast<std::size_t>(
                            field)]))
            >2.0e-11*scale_external) {
            return false;
        }
    }
    return true;
}

bool estimator_allows(
    const ContractionTopology& topology,
    const std::array<
        const RoleResolvedMarkedFieldTerm*,3>& terms) {
    for (const auto& block:
         topology.stochastic_blocks) {
        int external_density_marks=0;
        for (int leg:block) {
            const LegAddress address=
                topology.stochastic_legs[
                    static_cast<std::size_t>(leg)];
            const auto* term=
                terms[static_cast<std::size_t>(
                    address.field)];
            if (term->stochastic_roles.at(
                    static_cast<std::size_t>(
                        address.local))
                ==StochasticBlockRole::Density) {
                ++external_density_marks;
            }
        }
        /*
         * S122/S121/S113/S111 make the three external catalog-particle
         * indices distinct.  Internal displacement-source coincidences are
         * not removed.  Hence a same-source Poisson cumulant may contain at
         * most one external-density mark.
         */
        if (external_density_marks>=2) return false;
    }
    return true;
}

std::vector<std::vector<StochasticBlockRole>>
structural_role_assignments(
    int matter_count,
    int stochastic_count,
    ReconstructedStochasticMap map,
    bool reconstruction_enabled) {
    /*
     * Production fixed-Poisson metadata only reaches this helper after
     * reconstructed_fixed_poisson_bispectrum() has required
     * reconstruction.enabled=true.  The disabled branch is retained as the
     * structural analogue of an ordinary density mark, but it intentionally
     * does not encode provider-specific K0 support and must not be used to
     * advertise a future pre-reconstruction fixed-Poisson baseline without a
     * separate provider-capability check.
     */
    std::vector<std::vector<StochasticBlockRole>> result;
    if (!reconstruction_enabled) {
        if (stochastic_count==0) result.push_back({});
        if (stochastic_count==1) {
            result.push_back(
                {StochasticBlockRole::Density});
        }
        return result;
    }
    const int total_count=matter_count+stochastic_count;
    for (int shift_count=0;
         shift_count<total_count;
         ++shift_count) {
        const int block_count=shift_count+1;
        std::uint64_t assignment_count=1;
        for (int index=0;index<total_count;++index) {
            assignment_count*=
                static_cast<std::uint64_t>(block_count);
        }
        for (std::uint64_t code=0;
             code<assignment_count;
             ++code) {
            std::uint64_t remaining=code;
            std::vector<int> block_sizes(
                static_cast<std::size_t>(block_count),0);
            for (int index=0;index<matter_count;++index) {
                const int block=static_cast<int>(
                    remaining
                    %static_cast<std::uint64_t>(
                        block_count));
                remaining/=
                    static_cast<std::uint64_t>(
                        block_count);
                ++block_sizes[
                    static_cast<std::size_t>(block)];
            }
            bool invalid=false;
            std::vector<int> stochastic_assignment(
                static_cast<std::size_t>(
                    stochastic_count),-1);
            std::vector<int> stochastic_per_block(
                static_cast<std::size_t>(block_count),0);
            for (int index=0;
                 index<stochastic_count;
                 ++index) {
                const int block=static_cast<int>(
                    remaining
                    %static_cast<std::uint64_t>(
                        block_count));
                remaining/=
                    static_cast<std::uint64_t>(
                        block_count);
                stochastic_assignment[
                    static_cast<std::size_t>(
                        index)]=block;
                ++block_sizes[
                    static_cast<std::size_t>(block)];
                if (++stochastic_per_block[
                        static_cast<std::size_t>(
                            block)]>1
                    ||(map==ReconstructedStochasticMap::
                            DensityOnly
                       &&block>0)) {
                    invalid=true;
                    break;
                }
            }
            if (invalid
                ||std::any_of(
                    block_sizes.begin(),
                    block_sizes.end(),
                    [](int size) { return size==0; })) {
                continue;
            }
            std::vector<StochasticBlockRole> roles;
            roles.reserve(
                static_cast<std::size_t>(
                    stochastic_count));
            for (int block:stochastic_assignment) {
                roles.push_back(
                    block==0
                    ?StochasticBlockRole::Density
                    :StochasticBlockRole::Shift);
            }
            if (std::find(
                    result.begin(),result.end(),roles)
                ==result.end()) {
                result.push_back(std::move(roles));
            }
        }
    }
    return result;
}

bool structurally_estimator_allowed(
    const ContractionTopology& topology,
    ReconstructedStochasticMap map,
    bool reconstruction_enabled) {
    std::array<int,3> matter_count{{0,0,0}};
    std::array<int,3> stochastic_count{{0,0,0}};
    for (const LegAddress& address:
         topology.matter_legs) {
        if (address.field>=0) {
            ++matter_count[
                static_cast<std::size_t>(
                    address.field)];
        }
    }
    for (const LegAddress& address:
         topology.stochastic_legs) {
        ++stochastic_count[
            static_cast<std::size_t>(
                address.field)];
    }
    std::array<
        std::vector<std::vector<
            StochasticBlockRole>>,3> choices;
    for (int field=0;field<3;++field) {
        choices[static_cast<std::size_t>(field)]=
            structural_role_assignments(
                matter_count[
                    static_cast<std::size_t>(field)],
                stochastic_count[
                    static_cast<std::size_t>(field)],
                map,reconstruction_enabled);
        if (choices[
                static_cast<std::size_t>(field)]
                .empty()) {
            return false;
        }
    }
    for (const auto& role0:choices[0]) {
        for (const auto& role1:choices[1]) {
            for (const auto& role2:choices[2]) {
                const std::array<
                    const std::vector<
                        StochasticBlockRole>*,3>
                    selected={{&role0,&role1,&role2}};
                bool allowed=true;
                for (const auto& block:
                     topology.stochastic_blocks) {
                    int density_marks=0;
                    for (int leg:block) {
                        const LegAddress address=
                            topology.stochastic_legs[
                                static_cast<std::size_t>(
                                    leg)];
                        if (selected[
                                static_cast<std::size_t>(
                                    address.field)]
                                ->at(
                                    static_cast<std::size_t>(
                                        address.local))
                            ==StochasticBlockRole::
                                Density) {
                            ++density_marks;
                        }
                    }
                    if (density_marks>=2) {
                        allowed=false;
                        break;
                    }
                }
                if (allowed) return true;
            }
        }
    }
    return false;
}

SparsePolynomial evaluate_topology(
    const ContractionTopology& topology,
    const PowerSpectrum& power,
    const std::array<Vec3,3>& external,
    const Vec3& free_value,
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    ReconstructedStochasticMap map,
    bool& has_estimator_allowed_term) {
    std::vector<Vec3> variables;
    if (!solve_variables(
            topology,external,free_value,variables)) {
        return {};
    }
    double prefactor=topology.jacobian;
    for (std::size_t pair=0;
         pair<topology.matter_pairs.size();
         ++pair) {
        const double k=marisa_b_halo_v1::norm(
            variables[pair]);
        if (!(k>1.0e-12&&std::isfinite(k))) return {};
        const double value=power(k);
        if (!std::isfinite(value)) return {};
        prefactor*=value;
    }

    std::array<std::vector<Vec3>,3> matter;
    std::array<std::vector<Vec3>,3> noise;
    std::vector<std::vector<Vec3>> decoration_matter(
        topology.stochastic_blocks.size());
    for (std::size_t leg=0;
         leg<topology.matter_legs.size();
         ++leg) {
        const LegAddress address=
            topology.matter_legs[leg];
        const Vec3 momentum=
            linear_combination(
                topology.matter_coefficients[leg],
                variables);
        if (address.field>=0) {
            matter[
                static_cast<std::size_t>(
                    address.field)].push_back(
                        momentum);
        } else {
            decoration_matter[
                static_cast<std::size_t>(
                    address.decoration_block)]
                .push_back(momentum);
        }
    }
    for (std::size_t leg=0;
         leg<topology.stochastic_legs.size();
         ++leg) {
        const LegAddress address=
            topology.stochastic_legs[leg];
        noise[static_cast<std::size_t>(address.field)]
            .push_back(
                linear_combination(
                    topology.stochastic_coefficients[leg],
                    variables));
    }

    static const UnitPoissonMarkedProvider marked;
    static const ZeroPoissonMarkedProvider zero_marked;
    const MarkedFieldKernelProvider& shift_marked=
        map==ReconstructedStochasticMap::
                TiedDensityAndShift
        ?static_cast<const MarkedFieldKernelProvider&>(
            marked)
        :static_cast<const MarkedFieldKernelProvider&>(
            zero_marked);
    std::array<
        std::vector<RoleResolvedMarkedFieldTerm>,3> field_terms;
    for (int field=0;field<3;++field) {
        field_terms[static_cast<std::size_t>(field)]=
            reconstructed_multimarked_field_terms(
                base,marked,base,shift_marked,
                reconstruction,
                matter[static_cast<std::size_t>(field)],
                noise[static_cast<std::size_t>(field)]);
        if (field_terms[
                static_cast<std::size_t>(field)].empty()) {
            return {};
        }
    }
    SparsePolynomial intensity_product=
        SparsePolynomial::constant(1.0);
    for (const auto& decoration:
         decoration_matter) {
        if (!decoration.empty()) {
            intensity_product=
                intensity_product
                *base.deterministic(decoration);
        }
    }
    SparsePolynomial result;
    for (const auto& term0:field_terms[0]) {
        for (const auto& term1:field_terms[1]) {
            for (const auto& term2:field_terms[2]) {
                const std::array<
                    const RoleResolvedMarkedFieldTerm*,3>
                    selected={{
                        &term0,&term1,&term2}};
                if (!estimator_allows(
                        topology,selected)) {
                    continue;
                }
                has_estimator_allowed_term=true;
                result+=prefactor
                    *(intensity_product
                      *term0.value*term1.value
                      *term2.value);
            }
        }
    }
    return result;
}

FixedPoissonOrderTemplates evaluate_tree(
    const std::vector<ContractionTopology>& topologies,
    const PowerSpectrum& power,
    const std::array<Vec3,3>& external,
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    ReconstructedStochasticMap map) {
    FixedPoissonOrderTemplates result;
    result.generated_topologies=topologies.size();
    for (const auto& topology:topologies) {
        bool allowed=false;
        result.by_inverse_number_density[
            static_cast<std::size_t>(
                topology.noise_cost)]
            +=evaluate_topology(
                topology,power,external,Vec3{},
                base,reconstruction,map,allowed);
        result.estimator_allowed_topologies+=
            structurally_estimator_allowed(
                topology,map,reconstruction.enabled);
    }
    return result;
}

FixedPoissonOrderTemplates evaluate_one_loop(
    const std::vector<ContractionTopology>& topologies,
    const PowerSpectrum& power,
    const std::array<Vec3,3>& external,
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& config,
    ReconstructedStochasticMap map) {
    FixedPoissonOrderTemplates result;
    result.generated_topologies=topologies.size();
    result.estimator_allowed_topologies=
        static_cast<std::size_t>(
            std::count_if(
                topologies.begin(),topologies.end(),
                [&](const ContractionTopology& topology) {
                    return structurally_estimator_allowed(
                        topology,map,
                        reconstruction.enabled);
                }));
    const QuadratureRule radial=
        gauss_legendre(
            config.n_radial,
            std::log(config.qmin),
            std::log(config.qmax));
    const QuadratureRule mu=
        gauss_legendre(config.n_mu,-1.0,1.0);
    const QuadratureRule phi=
        gauss_legendre(
            config.n_phi,0.0,2.0*kPi);
    const double measure=
        1.0/(8.0*kPi*kPi*kPi);
    for (std::size_t ir=0;
         ir<radial.nodes.size();
         ++ir) {
        const double radius=std::exp(
            radial.nodes[ir]);
        for (std::size_t imu=0;
             imu<mu.nodes.size();
             ++imu) {
            const double cosine=mu.nodes[imu];
            const double sine=std::sqrt(
                std::max(
                    0.0,1.0-cosine*cosine));
            for (std::size_t iphi=0;
                 iphi<phi.nodes.size();
                 ++iphi) {
                const Vec3 q{
                    radius*sine*std::cos(
                        phi.nodes[iphi]),
                    radius*sine*std::sin(
                        phi.nodes[iphi]),
                    radius*cosine};
                const double weight=
                    radial.weights[ir]
                    *radius*radius*radius
                    *mu.weights[imu]
                    *phi.weights[iphi]
                    *measure;
                for (std::size_t index=0;
                     index<topologies.size();
                     ++index) {
                    bool allowed=false;
                    const auto& topology=
                        topologies[index];
                    result.by_inverse_number_density[
                        static_cast<std::size_t>(
                            topology.noise_cost)]
                        +=weight*evaluate_topology(
                            topology,power,external,q,
                            base,reconstruction,map,
                            allowed);
                }
                ++result.integration_nodes;
            }
        }
    }
    return result;
}

}  // namespace

std::array<SparsePolynomial,4>
ReconstructedFixedPoissonTemplates::
total_by_inverse_number_density() const {
    std::array<SparsePolynomial,4> result;
    for (std::size_t index=1;
         index<result.size();
         ++index) {
        result[index]=
            tree.by_inverse_number_density[index]
            +one_loop.by_inverse_number_density[index];
    }
    return result;
}

double ReconstructedFixedPoissonTemplates::evaluate(
    const std::array<double,kParameterCount>& parameter_values,
    double number_density) const {
    if (!(number_density>0.0
          &&std::isfinite(number_density))) {
        throw std::invalid_argument(
            "fixed-Poisson number density must be "
            "positive and finite");
    }
    const auto total=
        total_by_inverse_number_density();
    double result=0.0;
    double inverse_power=1.0;
    for (std::size_t index=1;
         index<total.size();
         ++index) {
        inverse_power/=number_density;
        result+=inverse_power
            *total[index].evaluate(parameter_values);
    }
    return result;
}

ReconstructedFixedPoissonTemplates
reconstructed_fixed_poisson_bispectrum(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const StochasticIntegrationConfig& integration,
    ReconstructedStochasticMap map,
    PoissonIntensityMode intensity_mode,
    bool include_bare_one_loop) {
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "fixed post-reconstruction Poisson baseline "
            "requires enabled reconstruction");
    }
    if (!(integration.qmin>0.0
          &&integration.qmax>integration.qmin
          &&integration.n_radial>0
          &&integration.n_mu>0
          &&integration.n_phi>0)) {
        throw std::invalid_argument(
            "invalid fixed-Poisson integration configuration");
    }
    Vec3 closure{};
    for (const Vec3& value:closed_triangle) {
        closure=marisa_b_halo_v1::add(
            closure,value);
    }
    if (marisa_b_halo_v1::norm(closure)>1.0e-11) {
        throw std::invalid_argument(
            "fixed-Poisson triangle does not close");
    }
    static const std::vector<ContractionTopology>
        unit_tree_topologies=generate_topologies(
            2,PoissonIntensityMode::UnitK0);
    static const std::vector<ContractionTopology>
        unit_loop_topologies=generate_topologies(
            3,PoissonIntensityMode::UnitK0);
    static const std::vector<ContractionTopology>
        conditional_tree_topologies=generate_topologies(
            2,PoissonIntensityMode::ConditionalTracer);
    static const std::vector<ContractionTopology>
        conditional_loop_topologies=generate_topologies(
            3,PoissonIntensityMode::ConditionalTracer);
    const auto& tree_topologies=
        intensity_mode==PoissonIntensityMode::UnitK0
        ?unit_tree_topologies
        :conditional_tree_topologies;
    const auto& loop_topologies=
        intensity_mode==PoissonIntensityMode::UnitK0
        ?unit_loop_topologies
        :conditional_loop_topologies;
    ReconstructedFixedPoissonTemplates result;
    result.tree=evaluate_tree(
        tree_topologies,linear_power,
        closed_triangle,base_provider,
        reconstruction,map);
    if (include_bare_one_loop) {
        result.one_loop=evaluate_one_loop(
            loop_topologies,linear_power,
            closed_triangle,base_provider,
            reconstruction,integration,map);
    }
    return result;
}

}  // namespace marisa_b_eft_v2
