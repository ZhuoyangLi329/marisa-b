#include "field_kernel_provider.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "halo_v1.h"

namespace marisa_b_eft_v2 {
namespace {

bool has_opposite_pair(const std::vector<Vec3>& momenta) {
    for (std::size_t left=0;left<momenta.size();++left) {
        for (std::size_t right=left+1;right<momenta.size();++right) {
            const double scale=std::max({
                marisa_b_halo_v1::norm(momenta[left]),
                marisa_b_halo_v1::norm(momenta[right]),1.0});
            if (marisa_b_halo_v1::norm(
                    marisa_b_halo_v1::add(momenta[left],momenta[right]))
                <=1.0e-14*scale) return true;
        }
    }
    return false;
}

double factorial(int value) {
    double result=1.0;
    for (int factor=2;factor<=value;++factor) {
        result*=static_cast<double>(factor);
    }
    return result;
}

void generate_partitions_recursive(
    int next,
    int count,
    std::vector<std::vector<int>>& blocks,
    std::vector<std::vector<std::vector<int>>>& output) {
    if (next==count) {
        output.push_back(blocks);
        return;
    }
    for (std::size_t index=0;index<blocks.size();++index) {
        blocks[index].push_back(next);
        generate_partitions_recursive(next+1,count,blocks,output);
        blocks[index].pop_back();
    }
    blocks.push_back(std::vector<int>{next});
    generate_partitions_recursive(next+1,count,blocks,output);
    blocks.pop_back();
}

const std::vector<std::vector<std::vector<int>>>&
set_partitions(int count) {
    if (count<1 || count>4) {
        throw std::invalid_argument(
            "reconstructed EFT kernel order must be one through four");
    }
    static const std::array<
        std::vector<std::vector<std::vector<int>>>,5> cache=[] {
        std::array<std::vector<std::vector<std::vector<int>>>,5> result;
        for (int order=1;order<=4;++order) {
            std::vector<std::vector<int>> blocks;
            generate_partitions_recursive(
                0,order,blocks,result[static_cast<std::size_t>(order)]);
        }
        return result;
    }();
    return cache[static_cast<std::size_t>(count)];
}

std::vector<Vec3> block_vectors(
    const std::vector<Vec3>& momenta,
    const std::vector<int>& block) {
    std::vector<Vec3> result;
    result.reserve(block.size());
    for (int index:block) {
        result.push_back(momenta.at(static_cast<std::size_t>(index)));
    }
    return result;
}

bool is_zero_block(const std::vector<Vec3>& vectors) {
    const Vec3 total=sum(vectors);
    double scale=1.0;
    for (const Vec3& value:vectors) {
        scale=std::max(scale,marisa_b_halo_v1::norm(value));
    }
    return marisa_b_halo_v1::norm(total)<=1.0e-14*scale;
}

void validate_reconstruction(
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction) {
    if (reconstruction.enabled
        &&(!(reconstruction.smoothing_radius>=0.0)
           ||!(reconstruction.bias_recon>0.0)
           ||!(reconstruction.cell_size>=0.0)
           ||!std::isfinite(reconstruction.smoothing_radius)
           ||!std::isfinite(reconstruction.bias_recon)
           ||!std::isfinite(reconstruction.cell_size))) {
        throw std::invalid_argument(
            "invalid reconstructed EFT kernel configuration");
    }
}

FieldKernelVariation multiply(
    const FieldKernelVariation& left,
    const FieldKernelVariation& right) {
    return FieldKernelVariation{
        left.value*right.value,
        left.direction*right.value+left.value*right.direction};
}

class ZeroFieldKernelProvider final : public FieldKernelProvider {
public:
    SparsePolynomial deterministic(
        const std::vector<Vec3>&) const override {
        return {};
    }
    std::string_view name() const noexcept override {
        return "zero_direction";
    }
};

}  // namespace

SparsePolynomial EftBiasKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    return deterministic_kernel(momenta);
}

SparsePolynomial MatterKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    return SparsePolynomial::constant(spt_F(momenta));
}

SparsePolynomial V1CompatibilityKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    SparsePolynomial result=adapt_v1_polynomial(
        marisa_b_halo_v1::pre_reconstruction_kernel(momenta).regular);
    if (momenta.size()==3 && has_opposite_pair(momenta)) {
        // halo-v1 stores this matter term as a P13 tadpole rather than in its
        // regular polynomial.  EFT-v2 has the analytic pointwise limit.
        result+=spt_F(momenta)*SparsePolynomial::variable(ParameterId::B1);
    }
    return result;
}

ReconstructedFieldKernelProvider::ReconstructedFieldKernelProvider(
    const FieldKernelProvider& base,
    marisa_b_halo_v1::ReconstructionConfig reconstruction)
    :base_(base),
     reconstruction_(std::move(reconstruction)),
     name_("reconstructed("+std::string(base.name())+")") {
    validate_reconstruction(reconstruction_);
}

FieldKernelVariation reconstructed_field_kernel_variation(
    const FieldKernelProvider& base,
    const FieldKernelProvider& direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta) {
    return reconstructed_field_kernel_variation_with_shift(
        base,direction,base,direction,
        reconstruction,momenta);
}

FieldKernelVariation reconstructed_field_kernel_variation_with_shift(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_direction,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta) {
    validate_reconstruction(reconstruction);
    const int order=static_cast<int>(momenta.size());
    if (order<1 || order>4) {
        throw std::invalid_argument(
            "reconstructed EFT kernel order must be one through four");
    }
    if (!reconstruction.enabled) {
        return FieldKernelVariation{
            density_base.deterministic(momenta),
            density_direction.deterministic(momenta)};
    }

    const Vec3 output=sum(momenta);
    FieldKernelVariation result;
    for (const std::vector<std::vector<int>>& partition:
         set_partitions(order)) {
        double symmetry=1.0/factorial(order);
        for (const std::vector<int>& block:partition) {
            symmetry*=factorial(static_cast<int>(block.size()));
        }
        for (std::size_t density_index=0;
             density_index<partition.size();++density_index) {
            const std::vector<Vec3> density_vectors=
                block_vectors(momenta,partition[density_index]);
            if (is_zero_block(density_vectors)) continue;

            FieldKernelVariation term{
                density_base.deterministic(density_vectors),
                density_direction.deterministic(
                    density_vectors)};
            double shift_product=symmetry;
            bool vanishes=false;
            for (std::size_t block_index=0;
                 block_index<partition.size();++block_index) {
                if (block_index==density_index) continue;
                const std::vector<Vec3> shift_vectors=
                    block_vectors(momenta,partition[block_index]);
                const Vec3 block_momentum=sum(shift_vectors);
                const double shift=
                    marisa_b_halo_v1::reconstruction_shift_factor(
                        output,block_momentum,reconstruction);
                if (shift==0.0) {
                    vanishes=true;
                    break;
                }
                term=multiply(
                    term,
                    FieldKernelVariation{
                        shift_base.deterministic(
                            shift_vectors),
                        shift_direction.deterministic(
                            shift_vectors)});
                shift_product*=shift;
            }
            if (!vanishes) {
                result.value+=shift_product*term.value;
                result.direction+=shift_product*term.direction;
            }
        }
    }
    return result;
}

SparsePolynomial reconstructed_marked_field_kernel(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const Vec3& stochastic_momentum) {
    validate_reconstruction(reconstruction);
    const int matter_order=
        static_cast<int>(matter_momenta.size());
    if (matter_order<0 || matter_order>3) {
        throw std::invalid_argument(
            "reconstructed marked EFT kernel supports zero through "
            "three matter legs");
    }
    if (!reconstruction.enabled) {
        return density_marked.marked(
            matter_momenta,stochastic_momentum);
    }

    std::vector<Vec3> labelled=matter_momenta;
    labelled.push_back(stochastic_momentum);
    const int marked_index=matter_order;
    const int total_order=matter_order+1;
    const Vec3 output=sum(labelled);
    SparsePolynomial result;
    for (const std::vector<std::vector<int>>& partition:
         set_partitions(total_order)) {
        std::size_t marked_block_index=partition.size();
        double symmetry=1.0/factorial(matter_order);
        for (std::size_t block_index=0;
             block_index<partition.size();
             ++block_index) {
            int matter_count=0;
            bool contains_mark=false;
            for (int index:partition[block_index]) {
                if (index==marked_index) contains_mark=true;
                else ++matter_count;
            }
            symmetry*=factorial(matter_count);
            if (contains_mark) {
                if (marked_block_index!=partition.size()) {
                    throw std::logic_error(
                        "marked reconstruction partition contains "
                        "the stochastic leg twice");
                }
                marked_block_index=block_index;
            }
        }
        if (marked_block_index==partition.size()) {
            throw std::logic_error(
                "marked reconstruction partition lost stochastic leg");
        }

        const auto marked_matter=[&]() {
            std::vector<Vec3> result;
            for (int index:partition[marked_block_index]) {
                if (index!=marked_index) {
                    result.push_back(
                        matter_momenta.at(
                            static_cast<std::size_t>(index)));
                }
            }
            return result;
        }();
        const std::vector<Vec3> marked_vectors=[&]() {
            std::vector<Vec3> result=marked_matter;
            result.push_back(stochastic_momentum);
            return result;
        }();

        // The marked block supplies the observed-density factor.
        if (!is_zero_block(marked_vectors)) {
            SparsePolynomial term=
                density_marked.marked(
                    marked_matter,stochastic_momentum);
            double prefactor=symmetry;
            bool vanishes=false;
            for (std::size_t block_index=0;
                 block_index<partition.size();
                 ++block_index) {
                if (block_index==marked_block_index) continue;
                const std::vector<Vec3> vectors=
                    block_vectors(
                        matter_momenta,
                        partition[block_index]);
                const double shift=
                    marisa_b_halo_v1::
                        reconstruction_shift_factor(
                            output,sum(vectors),reconstruction);
                if (shift==0.0) {
                    vanishes=true;
                    break;
                }
                term=term*shift_base.deterministic(vectors);
                prefactor*=shift;
            }
            if (!vanishes) result+=prefactor*term;
        }

        // Each ordinary block may instead supply the observed-density
        // factor, leaving the marked block as one displacement source.
        for (std::size_t density_index=0;
             density_index<partition.size();
             ++density_index) {
            if (density_index==marked_block_index) continue;
            const std::vector<Vec3> density_vectors=
                block_vectors(
                    matter_momenta,partition[density_index]);
            if (is_zero_block(density_vectors)) continue;
            SparsePolynomial term=
                density_base.deterministic(density_vectors);
            double prefactor=symmetry;
            bool vanishes=false;
            for (std::size_t block_index=0;
                 block_index<partition.size();
                 ++block_index) {
                if (block_index==density_index) continue;
                if (block_index==marked_block_index) {
                    const double shift=
                        marisa_b_halo_v1::
                            reconstruction_shift_factor(
                                output,sum(marked_vectors),
                                reconstruction);
                    if (shift==0.0) {
                        vanishes=true;
                        break;
                    }
                    term=term*shift_marked.marked(
                        marked_matter,stochastic_momentum);
                    prefactor*=shift;
                } else {
                    const std::vector<Vec3> vectors=
                        block_vectors(
                            matter_momenta,
                            partition[block_index]);
                    const double shift=
                        marisa_b_halo_v1::
                            reconstruction_shift_factor(
                                output,sum(vectors),
                                reconstruction);
                    if (shift==0.0) {
                        vanishes=true;
                        break;
                    }
                    term=term*shift_base.deterministic(vectors);
                    prefactor*=shift;
                }
            }
            if (!vanishes) result+=prefactor*term;
        }
    }
    return result;
}

std::vector<RoleResolvedMarkedFieldTerm>
reconstructed_multimarked_field_terms(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta) {
    validate_reconstruction(reconstruction);
    const int matter_count=
        static_cast<int>(matter_momenta.size());
    const int stochastic_count=
        static_cast<int>(stochastic_momenta.size());
    const int total_count=matter_count+stochastic_count;
    if (total_count<1 ||total_count>4) {
        throw std::invalid_argument(
            "reconstructed multimarked EFT kernel total order "
            "must be one through four");
    }
    if (!reconstruction.enabled) {
        if (stochastic_count==0) {
            return {{
                density_base.deterministic(matter_momenta),
                {}}};
        }
        if (stochastic_count==1) {
            return {{
                density_marked.marked(
                    matter_momenta,stochastic_momenta[0]),
                {StochasticBlockRole::Density}}};
        }
        return {};
    }
    if (stochastic_count==0) {
        static const ZeroFieldKernelProvider zero;
        return {{
            reconstructed_field_kernel_variation_with_shift(
                density_base,zero,shift_base,zero,
                reconstruction,matter_momenta).value,
            {}}};
    }

    Vec3 output=sum(matter_momenta);
    for (const Vec3& momentum:stochastic_momenta) {
        output=add(output,momentum);
    }
    std::vector<RoleResolvedMarkedFieldTerm> result;
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
            std::vector<std::vector<Vec3>> matter_blocks(
                static_cast<std::size_t>(block_count));
            std::vector<std::vector<Vec3>> stochastic_blocks(
                static_cast<std::size_t>(block_count));
            std::vector<int> stochastic_assignment(
                static_cast<std::size_t>(stochastic_count),-1);
            for (int index=0;index<matter_count;++index) {
                const int block=
                    static_cast<int>(
                        remaining
                        %static_cast<std::uint64_t>(
                            block_count));
                remaining/=
                    static_cast<std::uint64_t>(
                        block_count);
                matter_blocks[
                    static_cast<std::size_t>(block)]
                    .push_back(
                        matter_momenta[
                            static_cast<std::size_t>(
                                index)]);
            }
            bool invalid=false;
            for (int index=0;
                 index<stochastic_count;
                 ++index) {
                const int block=
                    static_cast<int>(
                        remaining
                        %static_cast<std::uint64_t>(
                            block_count));
                remaining/=
                    static_cast<std::uint64_t>(
                        block_count);
                auto& values=stochastic_blocks[
                    static_cast<std::size_t>(block)];
                stochastic_assignment[
                    static_cast<std::size_t>(index)]=block;
                values.push_back(
                    stochastic_momenta[
                        static_cast<std::size_t>(
                            index)]);
                if (values.size()>1) {
                    invalid=true;
                    break;
                }
            }
            if (invalid) continue;
            for (int block=0;
                 block<block_count;
                 ++block) {
                if (matter_blocks[
                        static_cast<std::size_t>(block)]
                        .empty()
                    &&stochastic_blocks[
                        static_cast<std::size_t>(block)]
                        .empty()) {
                    invalid=true;
                    break;
                }
            }
            if (invalid) continue;

            double coefficient=
                1.0/(factorial(matter_count)
                     *factorial(stochastic_count)
                     *factorial(shift_count));
            for (const auto& block:matter_blocks) {
                coefficient*=factorial(
                    static_cast<int>(block.size()));
            }
            SparsePolynomial term;
            bool initialized=false;
            bool vanishes=false;
            for (int block=0;
                 block<block_count;
                 ++block) {
                const auto& matter=matter_blocks[
                    static_cast<std::size_t>(block)];
                const auto& stochastic=
                    stochastic_blocks[
                        static_cast<std::size_t>(block)];
                SparsePolynomial kernel=
                    stochastic.empty()
                    ?(block==0
                      ?density_base.deterministic(matter)
                      :shift_base.deterministic(matter))
                    :(block==0
                      ?density_marked.marked(
                          matter,stochastic[0])
                      :shift_marked.marked(
                          matter,stochastic[0]));
                if (block>0) {
                    std::vector<Vec3> vectors=matter;
                    if (!stochastic.empty()) {
                        vectors.push_back(stochastic[0]);
                    }
                    const double shift=
                        marisa_b_halo_v1::
                            reconstruction_shift_factor(
                                output,sum(vectors),
                                reconstruction);
                    if (shift==0.0) {
                        vanishes=true;
                        break;
                    }
                    kernel*=shift;
                }
                if (!initialized) {
                    term=kernel;
                    initialized=true;
                } else {
                    term=term*kernel;
                }
            }
            if (!vanishes &&initialized) {
                RoleResolvedMarkedFieldTerm resolved;
                resolved.value=coefficient*term;
                resolved.stochastic_roles.reserve(
                    static_cast<std::size_t>(
                        stochastic_count));
                for (int block:stochastic_assignment) {
                    if (block<0) {
                        throw std::logic_error(
                            "multimarked reconstruction lost "
                            "a stochastic assignment");
                    }
                    resolved.stochastic_roles.push_back(
                        block==0
                        ?StochasticBlockRole::Density
                        :StochasticBlockRole::Shift);
                }
                result.push_back(std::move(resolved));
            }
        }
    }
    return result;
}

SparsePolynomial reconstructed_multimarked_field_kernel(
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta) {
    SparsePolynomial result;
    for (const RoleResolvedMarkedFieldTerm& term:
         reconstructed_multimarked_field_terms(
             density_base,density_marked,
             shift_base,shift_marked,
             reconstruction,matter_momenta,
             stochastic_momenta)) {
        result+=term.value;
    }
    return result;
}

SparsePolynomial ReconstructedFieldKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    static const ZeroFieldKernelProvider zero;
    return reconstructed_field_kernel_variation(
        base_,zero,reconstruction_,momenta).value;
}

}  // namespace marisa_b_eft_v2
