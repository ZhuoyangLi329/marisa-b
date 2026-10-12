#include "field_kernel_provider.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "PowerSpectrum.h"
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
           ||!(reconstruction.local_png_bias_kmin>=0.0)
           ||!std::isfinite(reconstruction.smoothing_radius)
           ||!std::isfinite(reconstruction.bias_recon)
           ||!std::isfinite(reconstruction.cell_size)
           ||!std::isfinite(
               reconstruction.local_png_bias_amplitude)
           ||!std::isfinite(
               reconstruction.local_png_bias_kmin)
           ||(reconstruction.local_png_bias_transfer==nullptr
              &&reconstruction.local_png_bias_amplitude!=0.0))) {
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

constexpr double kCollapseThreshold=1.686;

SparsePolynomial local_png_bphi(double p) {
    return 2.0*kCollapseThreshold
           *(SparsePolynomial::variable(ParameterId::B1)
             -SparsePolynomial::constant(p));
}

SparsePolynomial local_png_bphidelta(double p) {
    const SparsePolynomial one=SparsePolynomial::constant(1.0);
    const SparsePolynomial b1=SparsePolynomial::variable(ParameterId::B1);
    const SparsePolynomial b2=SparsePolynomial::variable(ParameterId::B2);
    const SparsePolynomial gamma2=
        SparsePolynomial::variable(ParameterId::Gamma2);
    const SparsePolynomial b2_native=b2-(4.0/3.0)*gamma2;
    return local_png_bphi(p)
           +2.0*(
               kCollapseThreshold
                   *(b2_native-(8.0/21.0)*(b1-one))
               -b1+one);
}

const char* local_png_sector_name(LocalPngTreeDirectionSector sector) {
    switch (sector) {
        case LocalPngTreeDirectionSector::ExternalBphi:
            return "external_bphi";
        case LocalPngTreeDirectionSector::LocalIc:
            return "local_ic";
        case LocalPngTreeDirectionSector::BphiAdvection:
            return "bphi_advection";
        case LocalPngTreeDirectionSector::Bphidelta:
            return "bphidelta";
        case LocalPngTreeDirectionSector::Total:
            return "total";
    }
    throw std::logic_error("unknown local-PNG tree direction sector");
}

const char* delta2_png_sector_name(Delta2PngTreeDirectionSector sector) {
    switch (sector) {
        case Delta2PngTreeDirectionSector::ExternalBPsi:
            return "external_bPsi";
        case Delta2PngTreeDirectionSector::BPsiAdvection:
            return "bPsi_advection";
        case Delta2PngTreeDirectionSector::BPsiDelta:
            return "bPsiDelta";
        case Delta2PngTreeDirectionSector::BPsiS:
            return "bPsiS";
        case Delta2PngTreeDirectionSector::Total:
            return "total";
    }
    throw std::logic_error("unknown Delta=2 PNG tree direction sector");
}

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

LocalPngHaloTreeDirectionProvider::LocalPngHaloTreeDirectionProvider(
    const PowerSpectrum& transfer,
    double p_bphi,
    double p_bphidelta,
    LocalPngTreeDirectionSector sector)
    :transfer_(transfer),
     p_bphi_(p_bphi),
     p_bphidelta_(p_bphidelta),
     sector_(sector),
     name_("local_png_halo_tree_direction_"
           +std::string(local_png_sector_name(sector))) {
    if (!std::isfinite(p_bphi_) ||!std::isfinite(p_bphidelta_)) {
        throw std::invalid_argument(
            "local-PNG universality offsets must be finite");
    }
}

SparsePolynomial LocalPngHaloTreeDirectionProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    if (momenta.size()==1) {
        if (sector_!=LocalPngTreeDirectionSector::Total
            &&sector_!=LocalPngTreeDirectionSector::ExternalBphi) return {};
        const double transfer=transfer_(
            marisa_b_halo_v1::norm(momenta[0]));
        if (!(transfer>0.0)) return {};
        return (1.0/transfer)*local_png_bphi(p_bphi_);
    }
    if (momenta.size()!=2) return {};
    const double k1=marisa_b_halo_v1::norm(momenta[0]);
    const double k2=marisa_b_halo_v1::norm(momenta[1]);
    const double m1=transfer_(k1);
    const double m2=transfer_(k2);
    if (!(k1>0.0 &&k2>0.0 &&m1>0.0 &&m2>0.0)) return {};

    SparsePolynomial result;
    const Vec3 combined=marisa_b_halo_v1::add(
        momenta[0],momenta[1]);
    const double combined_k=marisa_b_halo_v1::norm(combined);
    if ((sector_==LocalPngTreeDirectionSector::Total
         ||sector_==LocalPngTreeDirectionSector::LocalIc)
        &&combined_k>1.0e-12) {
        const double combined_transfer=transfer_(combined_k);
        if (combined_transfer>0.0) {
            result+=(combined_transfer/(m1*m2))
                    *SparsePolynomial::variable(ParameterId::B1);
        }
    }
    if (sector_==LocalPngTreeDirectionSector::Total
        ||sector_==LocalPngTreeDirectionSector::BphiAdvection) {
        const double mu=marisa_b_halo_v1::dot(
            momenta[0],momenta[1])/(k1*k2);
        const double advection=0.5*mu
            *(k1/(k2*m1)+k2/(k1*m2));
        result+=advection*local_png_bphi(p_bphi_);
    }
    if (sector_==LocalPngTreeDirectionSector::Total
        ||sector_==LocalPngTreeDirectionSector::Bphidelta) {
        result+=0.5*(1.0/m1+1.0/m2)
                *local_png_bphidelta(p_bphidelta_);
    }
    return result;
}

Delta2PngHaloTreeDirectionProvider::Delta2PngHaloTreeDirectionProvider(
    const PowerSpectrum& transfer,
    double bPsi,
    double bPsiDelta,
    double bPsiS,
    double k_pivot,
    Delta2PngTreeDirectionSector sector)
    :transfer_(transfer),
     bPsi_(bPsi),
     bPsiDelta_(bPsiDelta),
     bPsiS_(bPsiS),
     k_pivot_(k_pivot),
     sector_(sector),
     name_("delta2_png_halo_tree_direction_"
           +std::string(delta2_png_sector_name(sector))) {
    if (!std::isfinite(bPsi_)||!std::isfinite(bPsiDelta_)
        ||!std::isfinite(bPsiS_)) {
        throw std::invalid_argument(
            "Delta=2 PNG bias coefficients must be finite");
    }
    if (!(k_pivot_>0.0)||!std::isfinite(k_pivot_)) {
        throw std::invalid_argument(
            "Delta=2 PNG direction requires positive finite k_pivot");
    }
}

SparsePolynomial Delta2PngHaloTreeDirectionProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    if (momenta.size()!=1&&momenta.size()!=2) return {};
    const double k1=marisa_b_halo_v1::norm(momenta[0]);
    const double m1=transfer_(k1);
    if (!(k1>0.0&&m1>0.0)) return {};
    const double s1=(k1/k_pivot_)*(k1/k_pivot_)/m1;
    if (momenta.size()==1) {
        if (sector_!=Delta2PngTreeDirectionSector::Total
            &&sector_!=Delta2PngTreeDirectionSector::ExternalBPsi) {
            return {};
        }
        return SparsePolynomial::constant(bPsi_*s1);
    }

    const double k2=marisa_b_halo_v1::norm(momenta[1]);
    const double m2=transfer_(k2);
    if (!(k2>0.0&&m2>0.0)) return {};
    const double s2=(k2/k_pivot_)*(k2/k_pivot_)/m2;
    const double mu=marisa_b_halo_v1::dot(
        momenta[0],momenta[1])/(k1*k2);
    SparsePolynomial result;
    if (sector_==Delta2PngTreeDirectionSector::Total
        ||sector_==Delta2PngTreeDirectionSector::BPsiAdvection) {
        result+=SparsePolynomial::constant(
            0.5*bPsi_*mu*(k1/k2*s1+k2/k1*s2));
    }
    if (sector_==Delta2PngTreeDirectionSector::Total
        ||sector_==Delta2PngTreeDirectionSector::BPsiDelta) {
        result+=SparsePolynomial::constant(
            0.5*bPsiDelta_*(s1+s2));
    }
    if (sector_==Delta2PngTreeDirectionSector::Total
        ||sector_==Delta2PngTreeDirectionSector::BPsiS) {
        const double p2=0.5*(3.0*mu*mu-1.0);
        result+=SparsePolynomial::constant(
            0.5*bPsiS_*p2*(s1+s2));
    }
    return result;
}

ReconstructedFieldKernelProvider::ReconstructedFieldKernelProvider(
    const FieldKernelProvider& base,
    marisa_b_halo_v1::ReconstructionConfig reconstruction)
    :ReconstructedFieldKernelProvider(base,base,std::move(reconstruction)) {}

ReconstructedFieldKernelProvider::ReconstructedFieldKernelProvider(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& shift_base,
    marisa_b_halo_v1::ReconstructionConfig reconstruction)
    :density_base_(density_base),
     shift_base_(shift_base),
     reconstruction_(std::move(reconstruction)),
     name_(
         &density_base==&shift_base
             ?"reconstructed("+std::string(density_base.name())+")"
             :"reconstructed(density="+std::string(density_base.name())
                 +",shift="+std::string(shift_base.name())+")") {
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

RoleResolvedFieldKernelVariation
reconstructed_field_kernel_role_variation(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_response,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_response,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta) {
    static const ZeroFieldKernelProvider zero;
    const FieldKernelVariation density=
        reconstructed_field_kernel_variation_with_shift(
            density_base,density_response,
            shift_base,zero,reconstruction,momenta);
    const FieldKernelVariation shift=
        reconstructed_field_kernel_variation_with_shift(
            density_base,zero,
            shift_base,shift_response,reconstruction,momenta);

    /*
     * Both calls traverse the identical base reconstruction forest.  Return
     * one value rather than adding it twice; the regression test below this
     * API checks that the two independently evaluated values remain equal.
     */
    return RoleResolvedFieldKernelVariation{
        density.value,density.direction,shift.direction};
}

FieldKernelVariation shifted_random_field_kernel_variation(
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta) {
    validate_reconstruction(reconstruction);
    const int order=static_cast<int>(momenta.size());
    if (order<1 ||order>4) {
        throw std::invalid_argument(
            "shifted-random EFT kernel order must be one through four");
    }
    if (!reconstruction.enabled) return {};

    const Vec3 output=sum(momenta);
    FieldKernelVariation result;
    for (const std::vector<std::vector<int>>& partition:
         set_partitions(order)) {
        double coefficient=1.0/factorial(order);
        FieldKernelVariation term{
            SparsePolynomial::constant(1.0),{}};
        bool vanishes=false;
        for (const std::vector<int>& block:partition) {
            const std::vector<Vec3> vectors=
                block_vectors(momenta,block);
            const Vec3 block_momentum=sum(vectors);
            const double shift=
                marisa_b_halo_v1::reconstruction_shift_factor(
                    output,block_momentum,reconstruction);
            if (shift==0.0) {
                vanishes=true;
                break;
            }
            coefficient*=factorial(
                static_cast<int>(block.size()));
            term=multiply(
                term,
                FieldKernelVariation{
                    shift*shift_base.deterministic(vectors),
                    shift*shift_direction.deterministic(vectors)});
        }
        if (!vanishes) {
            result.value+=coefficient*term.value;
            result.direction+=coefficient*term.direction;
        }
    }
    return result;
}

FieldKernelVariation displaced_data_field_kernel_variation(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& density_direction,
    const FieldKernelProvider& shift_base,
    const FieldKernelProvider& shift_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& momenta) {
    const FieldKernelVariation reconstructed=
        reconstructed_field_kernel_variation_with_shift(
            density_base,density_direction,
            shift_base,shift_direction,
            reconstruction,momenta);
    const FieldKernelVariation random=
        shifted_random_field_kernel_variation(
            shift_base,shift_direction,
            reconstruction,momenta);
    return FieldKernelVariation{
        reconstructed.value+random.value,
        reconstructed.direction+random.direction};
}

FieldKernelVariation
reconstructed_field_kernel_local_png_denominator_variation(
    const FieldKernelProvider& base,
    const marisa_b_halo_v1::ReconstructionConfig& fixed_reconstruction,
    const PowerSpectrum& transfer,
    double kmin,
    const std::vector<Vec3>& momenta) {
    validate_reconstruction(fixed_reconstruction);
    const int order=static_cast<int>(momenta.size());
    if (order<1 ||order>4) {
        throw std::invalid_argument(
            "adaptive reconstructed EFT kernel order must be one through four");
    }
    if (!fixed_reconstruction.enabled) {
        return FieldKernelVariation{
            base.deterministic(momenta),{}};
    }
    if (fixed_reconstruction.local_png_bias_transfer!=nullptr
        ||fixed_reconstruction.local_png_bias_amplitude!=0.0) {
        throw std::invalid_argument(
            "adaptive denominator variation requires a fixed base map");
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
                block_vectors(
                    momenta,partition[density_index]);
            if (is_zero_block(density_vectors)) continue;
            FieldKernelVariation term{
                base.deterministic(density_vectors),{}};
            bool vanishes=false;
            for (std::size_t block_index=0;
                 block_index<partition.size();++block_index) {
                if (block_index==density_index) continue;
                const std::vector<Vec3> shift_vectors=
                    block_vectors(
                        momenta,partition[block_index]);
                const Vec3 block_momentum=sum(shift_vectors);
                const auto shift=
                    marisa_b_halo_v1::
                    reconstruction_shift_factor_local_png_variation(
                        output,block_momentum,
                        fixed_reconstruction,transfer,kmin);
                if (shift.value==0.0
                    &&shift.local_png_denominator_direction==0.0) {
                    vanishes=true;
                    break;
                }
                const SparsePolynomial source=
                    base.deterministic(shift_vectors);
                term=multiply(
                    term,
                    FieldKernelVariation{
                        shift.value*source,
                        shift.local_png_denominator_direction
                            *source});
            }
            if (!vanishes) {
                result.value+=symmetry*term.value;
                result.direction+=symmetry*term.direction;
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

namespace {

std::vector<RoleResolvedMarkedFieldTerm>
shifted_random_multimarked_terms_impl(
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta) {
    const int matter_count=static_cast<int>(matter_momenta.size());
    const int stochastic_count=static_cast<int>(stochastic_momenta.size());
    const int total_count=matter_count+stochastic_count;
    if (total_count<1 ||total_count>4) {
        throw std::invalid_argument(
            "shifted-random multimarked total order must be one through four");
    }
    if (!reconstruction.enabled) return {};

    Vec3 output=sum(matter_momenta);
    for (const Vec3& momentum:stochastic_momenta) {
        output=add(output,momentum);
    }
    std::vector<RoleResolvedMarkedFieldTerm> result;
    for (int block_count=1;block_count<=total_count;++block_count) {
        std::uint64_t assignment_count=1;
        for (int index=0;index<total_count;++index) {
            assignment_count*=static_cast<std::uint64_t>(block_count);
        }
        for (std::uint64_t code=0;code<assignment_count;++code) {
            std::uint64_t remaining=code;
            std::vector<std::vector<Vec3>> matter_blocks(
                static_cast<std::size_t>(block_count));
            std::vector<std::vector<Vec3>> stochastic_blocks(
                static_cast<std::size_t>(block_count));
            std::vector<int> stochastic_assignment(
                static_cast<std::size_t>(stochastic_count),-1);
            for (int index=0;index<matter_count;++index) {
                const int block=static_cast<int>(
                    remaining%static_cast<std::uint64_t>(block_count));
                remaining/=static_cast<std::uint64_t>(block_count);
                matter_blocks[static_cast<std::size_t>(block)].push_back(
                    matter_momenta[static_cast<std::size_t>(index)]);
            }
            bool invalid=false;
            for (int index=0;index<stochastic_count;++index) {
                const int block=static_cast<int>(
                    remaining%static_cast<std::uint64_t>(block_count));
                remaining/=static_cast<std::uint64_t>(block_count);
                auto& values=stochastic_blocks[static_cast<std::size_t>(block)];
                stochastic_assignment[static_cast<std::size_t>(index)]=block;
                values.push_back(stochastic_momenta[static_cast<std::size_t>(index)]);
                if (values.size()>1) {
                    invalid=true;
                    break;
                }
            }
            if (invalid) continue;
            for (int block=0;block<block_count;++block) {
                if (matter_blocks[static_cast<std::size_t>(block)].empty()
                    &&stochastic_blocks[static_cast<std::size_t>(block)].empty()) {
                    invalid=true;
                    break;
                }
            }
            if (invalid) continue;

            double coefficient=1.0/
                (factorial(matter_count)*factorial(stochastic_count)
                 *factorial(block_count));
            SparsePolynomial value=SparsePolynomial::constant(1.0);
            bool vanishes=false;
            for (int block=0;block<block_count;++block) {
                const auto& matter=matter_blocks[static_cast<std::size_t>(block)];
                const auto& stochastic=stochastic_blocks[
                    static_cast<std::size_t>(block)];
                coefficient*=factorial(static_cast<int>(matter.size()));
                SparsePolynomial kernel=stochastic.empty()
                    ?shift_base.deterministic(matter)
                    :shift_marked.marked(matter,stochastic[0]);
                std::vector<Vec3> vectors=matter;
                if (!stochastic.empty()) vectors.push_back(stochastic[0]);
                const double shift=
                    marisa_b_halo_v1::reconstruction_shift_factor(
                        output,sum(vectors),reconstruction);
                if (shift==0.0) {
                    vanishes=true;
                    break;
                }
                value=value*(shift*kernel);
            }
            if (vanishes) continue;
            RoleResolvedMarkedFieldTerm term;
            term.value=coefficient*value;
            term.stochastic_roles.assign(
                static_cast<std::size_t>(stochastic_count),
                StochasticBlockRole::Shift);
            result.push_back(std::move(term));
        }
    }
    return result;
}

std::vector<RoleResolvedMarkedFieldTerm>
shifted_random_density_marked_terms_impl(
    const MarkedFieldKernelProvider& random_density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta) {
    const int matter_count=static_cast<int>(matter_momenta.size());
    const int stochastic_count=static_cast<int>(stochastic_momenta.size());
    const int total_count=matter_count+stochastic_count;
    if (total_count<1 ||total_count>4) {
        throw std::invalid_argument(
            "random-density multimarked total order must be one through four");
    }
    if (stochastic_count==0) return {};

    Vec3 output=sum(matter_momenta);
    for (const Vec3& momentum:stochastic_momenta) {
        output=add(output,momentum);
    }
    std::vector<RoleResolvedMarkedFieldTerm> result;
    for (int density_stochastic=0;
         density_stochastic<stochastic_count;
         ++density_stochastic) {
        const SparsePolynomial density=
            random_density_marked.marked(
                {},stochastic_momenta[
                    static_cast<std::size_t>(density_stochastic)]);
        if (density.empty()) continue;
        const int remaining_stochastic=stochastic_count-1;
        const int remaining_total=matter_count+remaining_stochastic;
        if (remaining_total==0) {
            RoleResolvedMarkedFieldTerm term;
            term.value=density;
            term.stochastic_roles.assign(
                static_cast<std::size_t>(stochastic_count),
                StochasticBlockRole::Shift);
            term.stochastic_roles[
                static_cast<std::size_t>(density_stochastic)]=
                    StochasticBlockRole::Density;
            result.push_back(std::move(term));
            continue;
        }
        if (!reconstruction.enabled) continue;
        for (int block_count=1;
             block_count<=remaining_total;
             ++block_count) {
            std::uint64_t assignment_count=1;
            for (int index=0;index<remaining_total;++index) {
                assignment_count*=static_cast<std::uint64_t>(block_count);
            }
            for (std::uint64_t code=0;code<assignment_count;++code) {
                std::uint64_t remaining=code;
                std::vector<std::vector<Vec3>> matter_blocks(
                    static_cast<std::size_t>(block_count));
                std::vector<std::vector<Vec3>> stochastic_blocks(
                    static_cast<std::size_t>(block_count));
                std::vector<int> stochastic_assignment(
                    static_cast<std::size_t>(stochastic_count),-1);
                stochastic_assignment[
                    static_cast<std::size_t>(density_stochastic)]=-2;
                for (int index=0;index<matter_count;++index) {
                    const int block=static_cast<int>(
                        remaining%static_cast<std::uint64_t>(block_count));
                    remaining/=static_cast<std::uint64_t>(block_count);
                    matter_blocks[static_cast<std::size_t>(block)].push_back(
                        matter_momenta[static_cast<std::size_t>(index)]);
                }
                bool invalid=false;
                for (int source=0;source<stochastic_count;++source) {
                    if (source==density_stochastic) continue;
                    const int block=static_cast<int>(
                        remaining%static_cast<std::uint64_t>(block_count));
                    remaining/=static_cast<std::uint64_t>(block_count);
                    auto& values=stochastic_blocks[
                        static_cast<std::size_t>(block)];
                    stochastic_assignment[static_cast<std::size_t>(source)]=block;
                    values.push_back(stochastic_momenta[
                        static_cast<std::size_t>(source)]);
                    if (values.size()>1) {
                        invalid=true;
                        break;
                    }
                }
                if (invalid) continue;
                for (int block=0;block<block_count;++block) {
                    if (matter_blocks[static_cast<std::size_t>(block)].empty()
                        &&stochastic_blocks[static_cast<std::size_t>(block)].empty()) {
                        invalid=true;
                        break;
                    }
                }
                if (invalid) continue;
                double coefficient=1.0/
                    (factorial(matter_count)*factorial(stochastic_count)
                     *factorial(block_count));
                SparsePolynomial value=density;
                bool vanishes=false;
                for (int block=0;block<block_count;++block) {
                    const auto& matter=matter_blocks[
                        static_cast<std::size_t>(block)];
                    const auto& stochastic=stochastic_blocks[
                        static_cast<std::size_t>(block)];
                    coefficient*=factorial(static_cast<int>(matter.size()));
                    SparsePolynomial kernel=stochastic.empty()
                        ?shift_base.deterministic(matter)
                        :shift_marked.marked(matter,stochastic[0]);
                    std::vector<Vec3> vectors=matter;
                    if (!stochastic.empty()) vectors.push_back(stochastic[0]);
                    const double shift=
                        marisa_b_halo_v1::reconstruction_shift_factor(
                            output,sum(vectors),reconstruction);
                    if (shift==0.0) {
                        vanishes=true;
                        break;
                    }
                    value=value*(shift*kernel);
                }
                if (vanishes) continue;
                RoleResolvedMarkedFieldTerm term;
                term.value=coefficient*value;
                term.stochastic_roles.assign(
                    static_cast<std::size_t>(stochastic_count),
                    StochasticBlockRole::Shift);
                term.stochastic_roles[
                    static_cast<std::size_t>(density_stochastic)]=
                        StochasticBlockRole::Density;
                result.push_back(std::move(term));
            }
        }
    }
    return result;
}

}  // namespace

std::vector<RoleResolvedMarkedFieldTerm>
estimator_field_multimarked_terms(
    EstimatorFieldRole field,
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& data_density_marked,
    const MarkedFieldKernelProvider& random_density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const std::vector<Vec3>& stochastic_momenta) {
    validate_reconstruction(reconstruction);
    std::vector<RoleResolvedMarkedFieldTerm> shift_terms=
        shifted_random_multimarked_terms_impl(
            shift_base,shift_marked,reconstruction,
            matter_momenta,stochastic_momenta);
    if (field==EstimatorFieldRole::DisplacedData) {
        std::vector<RoleResolvedMarkedFieldTerm> result=
            reconstructed_multimarked_field_terms(
                density_base,data_density_marked,
                shift_base,shift_marked,reconstruction,
                matter_momenta,stochastic_momenta);
        result.insert(
            result.end(),
            std::make_move_iterator(shift_terms.begin()),
            std::make_move_iterator(shift_terms.end()));
        return result;
    }
    std::vector<RoleResolvedMarkedFieldTerm> result=std::move(shift_terms);
    std::vector<RoleResolvedMarkedFieldTerm> density_terms=
        shifted_random_density_marked_terms_impl(
            random_density_marked,shift_base,shift_marked,
            reconstruction,matter_momenta,stochastic_momenta);
    result.insert(
        result.end(),
        std::make_move_iterator(density_terms.begin()),
        std::make_move_iterator(density_terms.end()));
    return result;
}

SparsePolynomial estimator_field_marked_kernel(
    EstimatorFieldRole field,
    const FieldKernelProvider& density_base,
    const MarkedFieldKernelProvider& data_density_marked,
    const MarkedFieldKernelProvider& random_density_marked,
    const FieldKernelProvider& shift_base,
    const MarkedFieldKernelProvider& shift_marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter_momenta,
    const Vec3& stochastic_momentum) {
    SparsePolynomial result;
    for (const RoleResolvedMarkedFieldTerm& term:
         estimator_field_multimarked_terms(
             field,density_base,data_density_marked,
             random_density_marked,shift_base,shift_marked,
             reconstruction,matter_momenta,
             std::vector<Vec3>{stochastic_momentum})) {
        result+=term.value;
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
    return reconstructed_field_kernel_variation_with_shift(
        density_base_,zero,shift_base_,zero,
        reconstruction_,momenta).value;
}

ShiftedRandomFieldKernelProvider::ShiftedRandomFieldKernelProvider(
    const FieldKernelProvider& shift_base,
    marisa_b_halo_v1::ReconstructionConfig reconstruction)
    :shift_base_(shift_base),
     reconstruction_(std::move(reconstruction)),
     name_("shifted_random(shift="
           +std::string(shift_base.name())+")") {
    validate_reconstruction(reconstruction_);
}

SparsePolynomial ShiftedRandomFieldKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    static const ZeroFieldKernelProvider zero;
    return shifted_random_field_kernel_variation(
        shift_base_,zero,reconstruction_,momenta).value;
}

DisplacedDataFieldKernelProvider::DisplacedDataFieldKernelProvider(
    const FieldKernelProvider& density_base,
    const FieldKernelProvider& shift_base,
    marisa_b_halo_v1::ReconstructionConfig reconstruction)
    :density_base_(density_base),
     shift_base_(shift_base),
     reconstruction_(std::move(reconstruction)),
     name_("displaced_data(density="
           +std::string(density_base.name())
           +",shift="+std::string(shift_base.name())+")") {
    validate_reconstruction(reconstruction_);
}

SparsePolynomial DisplacedDataFieldKernelProvider::deterministic(
    const std::vector<Vec3>& momenta) const {
    static const ZeroFieldKernelProvider zero;
    return displaced_data_field_kernel_variation(
        density_base_,zero,shift_base_,zero,
        reconstruction_,momenta).value;
}

}  // namespace marisa_b_eft_v2
