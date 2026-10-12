#include "dr_stochastic_roles.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "poisson_reconstruction.h"
#include "stochastic.h"

namespace marisa_b_eft_v2 {
namespace {

class UnitMarkedProvider final:public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>& matter,
        const Vec3&) const override {
        return matter.empty()
            ?SparsePolynomial::constant(1.0)
            :SparsePolynomial{};
    }
    std::string_view name() const noexcept override {
        return "dr_unit_K0_mark";
    }
};

class NegativeUnitMarkedProvider final:public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>& matter,
        const Vec3&) const override {
        return matter.empty()
            ?SparsePolynomial::constant(-1.0)
            :SparsePolynomial{};
    }
    std::string_view name() const noexcept override {
        return "dr_negative_unit_K0_mark";
    }
};

class ZeroMarkedProvider final:public MarkedFieldKernelProvider {
public:
    SparsePolynomial marked(
        const std::vector<Vec3>&,
        const Vec3&) const override {
        return {};
    }
    std::string_view name() const noexcept override {
        return "dr_zero_mark";
    }
};

class BshotResidualMarkedProvider final:public MarkedFieldKernelProvider {
public:
    explicit BshotResidualMarkedProvider(
        const FieldKernelProvider& base):base_(base) {}

    SparsePolynomial marked(
        const std::vector<Vec3>& matter,
        const Vec3&) const override {
        if (matter.size()!=1) return {};
        return 0.5*base_.deterministic(matter);
    }
    std::string_view name() const noexcept override {
        return "dr_Bshot_residual_epsilon_delta";
    }
private:
    const FieldKernelProvider& base_;
};

/* Unit-amplitude [epsilon_Psi Psi] marked kernel for Delta=2. */
class Delta2EpsilonPsiMarkedProvider final:
    public MarkedFieldKernelProvider {
public:
    Delta2EpsilonPsiMarkedProvider(
        const PowerSpectrum& transfer,double k_pivot)
        :transfer_(transfer),k_pivot_(k_pivot) {
        if (!(k_pivot_>0.0)||!std::isfinite(k_pivot_)) {
            throw std::invalid_argument(
                "epsilon_Psi direction requires positive finite k_pivot");
        }
    }

    SparsePolynomial marked(
        const std::vector<Vec3>& matter,
        const Vec3&) const override {
        if (matter.size()!=1) return {};
        const double k=marisa_b_halo_v1::norm(matter[0]);
        const double transfer=transfer_(k);
        if (!(k>0.0)||!(transfer>0.0)) return {};
        const double shape=(k/k_pivot_)*(k/k_pivot_)/transfer;
        return SparsePolynomial::constant(shape);
    }
    std::string_view name() const noexcept override {
        return "dr_delta2_epsilon_Psi_Psi";
    }
private:
    const PowerSpectrum& transfer_;
    double k_pivot_=0.05;
};

class LinearCombinationProvider final:public FieldKernelProvider {
public:
    LinearCombinationProvider(
        const FieldKernelProvider& base,
        const FieldKernelProvider& direction,
        double amplitude):
        base_(base),direction_(direction),amplitude_(amplitude) {}

    SparsePolynomial deterministic(
        const std::vector<Vec3>& momenta) const override {
        return base_.deterministic(momenta)
            +amplitude_*direction_.deterministic(momenta);
    }
    std::string_view name() const noexcept override {
        return "dr_linear_combination_provider";
    }
private:
    const FieldKernelProvider& base_;
    const FieldKernelProvider& direction_;
    double amplitude_=0.0;
};

struct NoiseProviders {
    const MarkedFieldKernelProvider& data_density;
    const MarkedFieldKernelProvider& random_density;
    const MarkedFieldKernelProvider& shift;
};

struct RawAndRetained {
    SparsePolynomial raw;
    SparsePolynomial retained;
};

Vec3 negative(const Vec3& value) {
    return Vec3{-value.x,-value.y,-value.z};
}

void validate_triangle(const std::array<Vec3,3>& triangle) {
    Vec3 closure{};
    double scale=1.0;
    for (const Vec3& value:triangle) {
        closure=marisa_b_halo_v1::add(closure,value);
        scale=std::max(scale,marisa_b_halo_v1::norm(value));
    }
    if (marisa_b_halo_v1::norm(closure)>1.0e-12*scale) {
        throw std::invalid_argument(
            "D/R stochastic triangle does not close");
    }
}

const FieldKernelProvider& deterministic_provider(
    EstimatorFieldRole role,
    const DisplacedDataFieldKernelProvider& displaced,
    const ShiftedRandomFieldKernelProvider& random) {
    return role==EstimatorFieldRole::DisplacedData
        ?static_cast<const FieldKernelProvider&>(displaced)
        :static_cast<const FieldKernelProvider&>(random);
}

std::vector<RoleResolvedMarkedFieldTerm> marked_terms(
    EstimatorFieldRole role,
    const FieldKernelProvider& base,
    const NoiseProviders& providers,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const std::vector<Vec3>& matter,
    const Vec3& noise) {
    return estimator_field_multimarked_terms(
        role,base,providers.data_density,
        providers.random_density,base,providers.shift,
        reconstruction,matter,std::vector<Vec3>{noise});
}

RawAndRetained pure_three_point(
    const std::array<EstimatorFieldRole,3>& fields,
    const std::array<Vec3,3>& triangle,
    const FieldKernelProvider& base,
    const NoiseProviders& providers,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    bool apply_production_filter) {
    std::array<std::vector<RoleResolvedMarkedFieldTerm>,3> terms;
    for (std::size_t field=0;field<3;++field) {
        terms[field]=marked_terms(
            fields[field],base,providers,reconstruction,
            {},negative(triangle[field]));
    }
    RawAndRetained result;
    for (const auto& first:terms[0]) {
        for (const auto& second:terms[1]) {
            for (const auto& third:terms[2]) {
                const SparsePolynomial value=
                    first.value*second.value*third.value;
                result.raw+=value;
                int density_marks=0;
                for (const auto* term:{&first,&second,&third}) {
                    if (term->stochastic_roles.size()!=1) {
                        throw std::logic_error(
                            "leading pure contraction lost its mark");
                    }
                    density_marks+=
                        term->stochastic_roles[0]
                            ==StochasticBlockRole::Density;
                }
                if (!apply_production_filter ||density_marks<=1) {
                    result.retained+=value;
                }
            }
        }
    }
    return result;
}

RawAndRetained mixed_two_point(
    const PowerSpectrum& power,
    const std::array<EstimatorFieldRole,3>& fields,
    const std::array<Vec3,3>& triangle,
    const FieldKernelProvider& base,
    const NoiseProviders& pure_providers,
    const NoiseProviders& mixed_providers,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    bool apply_production_filter) {
    const DisplacedDataFieldKernelProvider displaced(
        base,base,reconstruction);
    const ShiftedRandomFieldKernelProvider random(
        base,reconstruction);
    RawAndRetained result;
    for (std::size_t matter_leg=0;matter_leg<3;++matter_leg) {
        const Vec3 matter=negative(triangle[matter_leg]);
        const double tree_power=power(
            marisa_b_halo_v1::norm(matter));
        const SparsePolynomial deterministic=
            deterministic_provider(
                fields[matter_leg],displaced,random)
                .deterministic({matter});
        for (std::size_t noise_leg=0;noise_leg<3;++noise_leg) {
            if (noise_leg==matter_leg) continue;
            const std::size_t mixed_leg=3-matter_leg-noise_leg;
            const Vec3 noise=negative(triangle[noise_leg]);
            const auto pure=marked_terms(
                fields[noise_leg],base,pure_providers,
                reconstruction,{},noise);
            const auto mixed=marked_terms(
                fields[mixed_leg],base,mixed_providers,
                reconstruction,{matter},noise);
            for (const auto& pure_term:pure) {
                for (const auto& mixed_term:mixed) {
                    const SparsePolynomial value=
                        tree_power*deterministic
                        *pure_term.value*mixed_term.value;
                    result.raw+=value;
                    if (pure_term.stochastic_roles.size()!=1
                        ||mixed_term.stochastic_roles.size()!=1) {
                        throw std::logic_error(
                            "leading mixed contraction lost its marks");
                    }
                    const int density_marks=
                        (pure_term.stochastic_roles[0]
                             ==StochasticBlockRole::Density)
                        +(mixed_term.stochastic_roles[0]
                             ==StochasticBlockRole::Density);
                    if (!apply_production_filter ||density_marks<=1) {
                        result.retained+=value;
                    }
                }
            }
        }
    }
    return result;
}

std::array<EstimatorFieldRole,3> fields_from_bits(int bits) {
    std::array<EstimatorFieldRole,3> result;
    for (int field=0;field<3;++field) {
        result[static_cast<std::size_t>(field)]=
            bits&(1<<field)
            ?EstimatorFieldRole::ShiftedRandom
            :EstimatorFieldRole::DisplacedData;
    }
    return result;
}

int random_count(int bits) {
    int count=0;
    for (int field=0;field<3;++field) {
        count+=(bits&(1<<field))!=0;
    }
    return count;
}

SignedDrRole role_for_count(int count) {
    switch (count) {
        case 0:return SignedDrRole::DDD;
        case 1:return SignedDrRole::DDR;
        case 2:return SignedDrRole::DRR;
        case 3:return SignedDrRole::RRR;
    }
    throw std::logic_error("invalid D/R random count");
}

double signed_factor(int count) {
    return count%2==0?1.0:-1.0;
}

template <class Evaluator>
SignedDrRoleTemplates signed_roles(
    const Evaluator& evaluate,
    const SparsePolynomial& tied) {
    SignedDrRoleTemplates result;
    for (int bits=0;bits<8;++bits) {
        const int count=random_count(bits);
        result[role_for_count(count)]+=signed_factor(count)
            *evaluate(fields_from_bits(bits));
    }
    result[SignedDrRole::TiedF]=tied;
    return result;
}

SignedDrRoleTemplates residual_bshot_roles(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction) {
    static const UnitMarkedProvider unit;
    static const ZeroMarkedProvider zero;
    const BshotResidualMarkedProvider bshot(base_provider);
    const NoiseProviders residual_pure{unit,zero,zero};
    const NoiseProviders residual_mixed{bshot,zero,bshot};
    const auto evaluate=[&](
        const std::array<EstimatorFieldRole,3>& fields) {
        return mixed_two_point(
            linear_power,fields,closed_triangle,base_provider,
            residual_pure,residual_mixed,reconstruction,false).raw;
    };
    const SparsePolynomial tied=
        reconstructed_leading_bshot_residual_generating_shape(
            linear_power,closed_triangle,base_provider,
            reconstruction,true);
    return signed_roles(evaluate,tied);
}

SparsePolynomial tied_delta2_epsilon_psi_shape(
    const PowerSpectrum& power,
    const std::array<Vec3,3>& triangle,
    const FieldKernelProvider& base,
    const Delta2EpsilonPsiMarkedProvider& marked,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction) {
    SparsePolynomial result;
    for (std::size_t matter_leg=0;matter_leg<3;++matter_leg) {
        const Vec3 matter=negative(triangle[matter_leg]);
        const double tree_power=power(
            marisa_b_halo_v1::norm(matter));
        const SparsePolynomial deterministic=
            base.deterministic({matter});
        for (std::size_t noise_leg=0;noise_leg<3;++noise_leg) {
            if (noise_leg==matter_leg) continue;
            const Vec3 noise=negative(triangle[noise_leg]);
            const SparsePolynomial mixed=
                reconstructed_marked_field_kernel(
                    base,marked,base,marked,reconstruction,
                    {matter},noise);
            result+=tree_power*deterministic*mixed;
        }
    }
    return result;
}

struct FixedSpeciesResult {
    std::array<SignedDrRoleTemplates,3> raw;
    std::array<SparsePolynomial,3> retained;
};

FixedSpeciesResult fixed_species(
    const PowerSpectrum& power,
    const std::array<Vec3,3>& triangle,
    const FieldKernelProvider& base,
    const NoiseProviders& providers,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction) {
    FixedSpeciesResult result;
    std::array<SparsePolynomial,3> retained_role_sum;
    for (int bits=0;bits<8;++bits) {
        const int count=random_count(bits);
        const double sign=signed_factor(count);
        const SignedDrRole role=role_for_count(count);
        const auto fields=fields_from_bits(bits);
        const RawAndRetained mixed=mixed_two_point(
            power,fields,triangle,base,providers,providers,
            reconstruction,true);
        const RawAndRetained pure=pure_three_point(
            fields,triangle,base,providers,reconstruction,true);
        result.raw[1][role]+=sign*mixed.raw;
        result.raw[2][role]+=sign*pure.raw;
        retained_role_sum[1]+=sign*mixed.retained;
        retained_role_sum[2]+=sign*pure.retained;
    }
    for (std::size_t inverse=1;inverse<3;++inverse) {
        result.raw[inverse][SignedDrRole::TiedF]=
            signed_dr_role_sum(result.raw[inverse]);
        result.retained[inverse]=retained_role_sum[inverse];
    }
    return result;
}

}  // namespace

SparsePolynomial signed_dr_role_sum(
    const SignedDrRoleTemplates& roles) {
    return roles[SignedDrRole::DDD]
        +roles[SignedDrRole::DDR]
        +roles[SignedDrRole::DRR]
        +roles[SignedDrRole::RRR];
}

DrLeadingStochasticPoissonTemplates
dr_leading_stochastic_poisson_tree(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const RandomCatalogConfig& random_catalog) {
    validate_triangle(closed_triangle);
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "D/R stochastic roles require enabled reconstruction");
    }
    if (!(random_catalog.random_to_halo_density>0.0)
        ||!std::isfinite(random_catalog.random_to_halo_density)) {
        throw std::invalid_argument(
            "random-to-halo density ratio must be positive and finite");
    }

    static const UnitMarkedProvider unit;
    static const NegativeUnitMarkedProvider negative_unit;
    static const ZeroMarkedProvider zero;
    const NoiseProviders residual_pure{unit,zero,zero};
    const NoiseProviders halo_poisson{unit,zero,unit};
    const NoiseProviders random_poisson{zero,unit,negative_unit};

    const auto residual_A=[&](const std::array<EstimatorFieldRole,3>& fields) {
        return pure_three_point(
            fields,closed_triangle,base_provider,residual_pure,
            reconstruction,false).raw;
    };
    const SparsePolynomial tied_A=SparsePolynomial::constant(1.0);

    DrLeadingStochasticPoissonTemplates result;
    result.random_catalog=random_catalog;
    result.Ashot_residual=signed_roles(residual_A,tied_A);
    result.Bshot_residual=residual_bshot_roles(
        linear_power,closed_triangle,base_provider,reconstruction);

    const FixedSpeciesResult halo=fixed_species(
        linear_power,closed_triangle,base_provider,
        halo_poisson,reconstruction);
    result.halo_poisson.raw_by_inverse_density=halo.raw;
    for (std::size_t inverse=1;inverse<3;++inverse) {
        result.halo_poisson.legacy_theory_retained[inverse]=
            halo.retained[inverse];
        result.halo_poisson.legacy_theory_filter_complement[inverse]=
            halo.retained[inverse]
            -signed_dr_role_sum(halo.raw[inverse]);
    }

    if (random_catalog.convention
        ==RandomCatalogConvention::FiniteIndependentPoisson) {
        const FixedSpeciesResult random=fixed_species(
            linear_power,closed_triangle,base_provider,
            random_poisson,reconstruction);
        result.random_poisson.raw_by_inverse_density=random.raw;
        for (std::size_t inverse=1;inverse<3;++inverse) {
            result.random_poisson.legacy_theory_retained[inverse]=
                random.retained[inverse];
            result.random_poisson.legacy_theory_filter_complement[inverse]=
                random.retained[inverse]
                -signed_dr_role_sum(random.raw[inverse]);
        }
    }
    return result;
}

DrLeadingStochasticPngResponses
dr_leading_stochastic_png_response(
    const PowerSpectrum& linear_power,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const FieldKernelProvider& local_png_direction,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const RandomCatalogConfig& random_catalog) {
    validate_triangle(closed_triangle);
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "D/R stochastic PNG response requires enabled reconstruction");
    }
    /*
     * Every fully-tree reconstructed leading contraction is a polynomial of
     * degree at most six in a common K1 marker.  The symmetric six-point
     * coefficient extractor below is therefore the exact derivative at zero,
     * not a small-step finite difference and not a fitted response gain.
     * Pairing +/- amplitudes before accumulation cancels every even/constant
     * contribution exactly, including the leading Ashot K0 direction.
     */
    std::array<DrLeadingStochasticPoissonTemplates,3> plus;
    std::array<DrLeadingStochasticPoissonTemplates,3> minus;
    for (std::size_t sample=0;sample<3;++sample) {
        const double amplitude=static_cast<double>(sample+1);
        const LinearCombinationProvider positive(
            base_provider,local_png_direction,amplitude);
        const LinearCombinationProvider negative(
            base_provider,local_png_direction,-amplitude);
        plus[sample]=dr_leading_stochastic_poisson_tree(
            linear_power,closed_triangle,positive,
            reconstruction,random_catalog);
        minus[sample]=dr_leading_stochastic_poisson_tree(
            linear_power,closed_triangle,negative,
            reconstruction,random_catalog);
    }
    const auto derivative=[&](const auto& member) {
        return (
            45.0*(member(plus[0])-member(minus[0]))
            -9.0*(member(plus[1])-member(minus[1]))
            +(member(plus[2])-member(minus[2]))
        )*(1.0/60.0);
    };
    DrLeadingStochasticPngResponses result;
    for (std::size_t role=0;role<kSignedDrRoleCount;++role) {
        result.Ashot_residual.value[role]=derivative(
            [role](const auto& value) {
                return value.Ashot_residual.value[role];
            });
        result.Bshot_residual.value[role]=derivative(
            [role](const auto& value) {
                return value.Bshot_residual.value[role];
            })*0.5;
    }
    /*
     * The strict generating shape differentiates both factors of
     * H=(K1)^2 sum(P), giving the sum of the alpha3 and alpha3PNG branches.
     * The frozen fully-tree main context deliberately uses the historical
     * ``legacy_accepted_pre`` policy, in which only the Gaussian-alpha3
     * branch is tied to Bshot_residual.  Its production linear response is
     * therefore one half of the strict generating derivative.  Fixed
     * Poisson is parameter-free and retains its complete derivative below.
     */
    for (std::size_t inverse=1;inverse<3;++inverse) {
        for (std::size_t role=0;role<kSignedDrRoleCount;++role) {
            result.halo_poisson.raw_by_inverse_density[inverse]
                .value[role]=derivative(
                    [inverse,role](const auto& value) {
                        return value.halo_poisson
                            .raw_by_inverse_density[inverse].value[role];
                    });
            result.random_poisson.raw_by_inverse_density[inverse]
                .value[role]=derivative(
                    [inverse,role](const auto& value) {
                        return value.random_poisson
                            .raw_by_inverse_density[inverse].value[role];
                    });
        }
        result.halo_poisson.legacy_theory_filter_complement[inverse]=derivative(
            [inverse](const auto& value) {
                return value.halo_poisson
                    .legacy_theory_filter_complement[inverse];
            });
        result.halo_poisson.legacy_theory_retained[inverse]=derivative(
            [inverse](const auto& value) {
                return value.halo_poisson.legacy_theory_retained[inverse];
            });
        result.random_poisson.legacy_theory_filter_complement[inverse]=derivative(
            [inverse](const auto& value) {
                return value.random_poisson
                    .legacy_theory_filter_complement[inverse];
            });
        result.random_poisson.legacy_theory_retained[inverse]=derivative(
            [inverse](const auto& value) {
                return value.random_poisson.legacy_theory_retained[inverse];
            });
    }
    return result;
}

SignedDrRoleTemplates dr_delta2_epsilon_psi_stochastic_tree(
    const PowerSpectrum& linear_power,
    const PowerSpectrum& transfer_m,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    double k_pivot) {
    validate_triangle(closed_triangle);
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "D/R epsilon_Psi stochastic roles require enabled reconstruction");
    }
    static const UnitMarkedProvider unit;
    static const ZeroMarkedProvider zero;
    const Delta2EpsilonPsiMarkedProvider epsilon_psi(
        transfer_m,k_pivot);
    const NoiseProviders pure{unit,zero,zero};
    /* epsilon_Psi is carried by both output density and halo shift source. */
    const NoiseProviders mixed{epsilon_psi,zero,epsilon_psi};
    const auto evaluate=[&](
        const std::array<EstimatorFieldRole,3>& fields) {
        return mixed_two_point(
            linear_power,fields,closed_triangle,base_provider,
            pure,mixed,reconstruction,false).raw;
    };
    const SparsePolynomial tied=tied_delta2_epsilon_psi_shape(
        linear_power,closed_triangle,base_provider,
        epsilon_psi,reconstruction);
    return signed_roles(evaluate,tied);
}

std::array<SparsePolynomial,3>
fixed_poisson_in_halo_density_units(
    const DrFixedPoissonSpeciesTemplates& halo,
    const DrFixedPoissonSpeciesTemplates& random,
    double random_to_halo_density) {
    if (!(random_to_halo_density>0.0)
        ||!std::isfinite(random_to_halo_density)) {
        throw std::invalid_argument(
            "random-to-halo density ratio must be positive and finite");
    }
    std::array<SparsePolynomial,3> result;
    double inverse_random_power=1.0;
    for (std::size_t inverse=1;inverse<3;++inverse) {
        inverse_random_power/=random_to_halo_density;
        result[inverse]=halo.legacy_theory_retained[inverse]
            +inverse_random_power*random.legacy_theory_retained[inverse];
    }
    return result;
}

}  // namespace marisa_b_eft_v2
