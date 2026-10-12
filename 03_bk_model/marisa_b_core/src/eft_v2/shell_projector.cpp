#include "shell_projector.h"

#include <algorithm>
#include <exception>
#include <map>
#include <stdexcept>
#include <vector>

#include "PowerSpectrum.h"
#include "diagram_assembler.h"

namespace marisa_b_eft_v2 {
namespace {

struct NodeTemplates {
    DiagramTemplates diagrams;
    CountertermTemplates counterterms;
    StochasticTemplates stochastic;
    StochasticTemplates stochastic_density_only;
    StochasticTemplates stochastic_noisy_shift;
    ReconstructedFixedPoissonTemplates fixed_poisson;
};

void add_weighted_stochastic(
    StochasticTemplates& destination,
    const StochasticTemplates& source,
    double weight) {
    for (std::size_t index=0;
         index<kStochasticCount;
         ++index) {
        destination.pure[index]+=
            weight*source.pure[index];
        destination.mixed_tree[index]+=
            weight*source.mixed_tree[index];
        destination.mixed_one_loop[index]+=
            weight*source.mixed_one_loop[index];
        destination.derivative_mixed[index]+=
            weight*source.derivative_mixed[index];
        destination.raw[index]+=
            weight*source.raw[index];
    }
    destination.bshot_bnabla2_cross+=
        weight*source.bshot_bnabla2_cross;
}

void add_weighted_fixed_poisson(
    ReconstructedFixedPoissonTemplates& destination,
    const ReconstructedFixedPoissonTemplates& source,
    double weight) {
    for (std::size_t inverse_nbar=1;
         inverse_nbar<4;
         ++inverse_nbar) {
        destination.tree
            .by_inverse_number_density[inverse_nbar]
            +=weight*source.tree
                .by_inverse_number_density[inverse_nbar];
        destination.one_loop
            .by_inverse_number_density[inverse_nbar]
            +=weight*source.one_loop
                .by_inverse_number_density[inverse_nbar];
    }
    destination.tree.generated_topologies=
        std::max(
            destination.tree.generated_topologies,
            source.tree.generated_topologies);
    destination.tree.estimator_allowed_topologies=
        std::max(
            destination.tree.estimator_allowed_topologies,
            source.tree.estimator_allowed_topologies);
    destination.tree.integration_nodes+=
        source.tree.integration_nodes;
    destination.one_loop.generated_topologies=
        std::max(
            destination.one_loop.generated_topologies,
            source.one_loop.generated_topologies);
    destination.one_loop.estimator_allowed_topologies=
        std::max(
            destination.one_loop.estimator_allowed_topologies,
            source.one_loop.estimator_allowed_topologies);
    destination.one_loop.integration_nodes+=
        source.one_loop.integration_nodes;
}

std::array<double,3> external_power(
    const PowerSpectrum& power,const std::array<Vec3,3>& triangle) {
    std::array<double,3> result={};
    for (int index=0;index<3;++index) {
        result[index]=power(marisa_b_halo_v1::norm(triangle[index]));
    }
    return result;
}

template<typename Result,typename Compute>
std::vector<Result> parallel_compute(std::size_t count,Compute compute) {
    std::vector<Result> result(count);
    std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic,1)
    for (std::ptrdiff_t index=0;index<static_cast<std::ptrdiff_t>(count);++index) {
        try {
            result[static_cast<std::size_t>(index)]=compute(
                static_cast<std::size_t>(index));
        } catch (...) {
#pragma omp critical(marisa_b_eft_v2_shell_failure)
            { if (!failure) failure=std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
    return result;
}

std::vector<double> unique_wavenumbers(
    const std::vector<marisa_b_shell_v1::ShellNode>& nodes) {
    std::vector<double> result;
    result.reserve(3*nodes.size());
    for (const auto& node:nodes) {
        for (const Vec3& value:node.closed_vectors) {
            result.push_back(marisa_b_halo_v1::norm(value));
        }
    }
    std::sort(result.begin(),result.end());
    result.erase(
        std::unique(result.begin(),result.end()),result.end());
    return result;
}

std::map<double,std::size_t> wavenumber_index(
    const std::vector<double>& values) {
    std::map<double,std::size_t> result;
    for (std::size_t index=0;index<values.size();++index) {
        result.emplace(values[index],index);
    }
    return result;
}

EftShellTemplates assemble_shell_templates(
    const marisa_b_shell_v1::ShellBin& bin,
    const std::vector<marisa_b_shell_v1::ShellNode>& nodes,
    const std::vector<NodeTemplates>& values,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    bool ir_safe,
    bool hybrid_factorized_analytic_tadpoles=false,
    std::uint64_t cached_p13_nodes=0,
    bool fftlog_analytic_convolutions=false) {
    EftShellTemplates result;
    result.bin=bin;
    result.shell_nodes=nodes.size();
    result.cached_p13_nodes=cached_p13_nodes;
    result.total_loop_nodes=cached_p13_nodes;
    result.hybrid_factorized_analytic_tadpoles=
        hybrid_factorized_analytic_tadpoles;
    result.fftlog_analytic_convolutions=
        fftlog_analytic_convolutions;
    result.fft_lattice_radial_measure=
        shell_config.radial_measure
        ==marisa_b_shell_v1::ShellRadialMeasure::FftLattice;
    result.diagrams.ir_safe=ir_safe;
    for (std::size_t node=0;node<nodes.size();++node) {
        const double weight=nodes[node].weight;
        const auto& value=values[node];
        result.diagrams.tree+=weight*value.diagrams.tree;
        result.diagrams.B222+=weight*value.diagrams.B222;
        result.diagrams.B321I+=weight*value.diagrams.B321I;
        result.diagrams.B321II+=weight*value.diagrams.B321II;
        result.diagrams.B411+=weight*value.diagrams.B411;
        result.diagrams.B321II_bare+=
            weight*value.diagrams.B321II_bare;
        result.diagrams.B411_bare+=
            weight*value.diagrams.B411_bare;
        result.diagrams.B321II_uv_subtraction+=
            weight*value.diagrams.B321II_uv_subtraction;
        result.diagrams.B411_uv_subtraction+=
            weight*value.diagrams.B411_uv_subtraction;
        result.diagrams.B321II_uv_restoration+=
            weight*value.diagrams.B321II_uv_restoration;
        result.diagrams.B411_uv_restoration+=
            weight*value.diagrams.B411_uv_restoration;
        result.total_loop_nodes+=value.diagrams.integration_nodes;
        for (std::size_t index=0;
             index<kCountertermCount;++index) {
            result.counterterms.BctrI[index]+=
                weight*value.counterterms.BctrI[index];
            result.counterterms.BctrII[index]+=
                weight*value.counterterms.BctrII[index];
            result.counterterms.total[index]+=
                weight*value.counterterms.total[index];
        }
        add_weighted_stochastic(
            result.stochastic,
            value.stochastic,weight);
        add_weighted_stochastic(
            result.stochastic_density_only,
            value.stochastic_density_only,weight);
        add_weighted_stochastic(
            result.stochastic_noisy_shift,
            value.stochastic_noisy_shift,weight);
        add_weighted_fixed_poisson(
            result.fixed_poisson,
            value.fixed_poisson,weight);
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
    return result;
}

void accumulate_multilevel_node(
    EftShellTemplates& result,
    const NodeTemplates& value,
    double weight,
    bool cheap_sector) {
    if (cheap_sector) {
        result.diagrams.tree+=weight*value.diagrams.tree;
        result.diagrams.B321II+=weight*value.diagrams.B321II;
        result.diagrams.B321II_bare+=
            weight*value.diagrams.B321II_bare;
        result.diagrams.B321II_uv_subtraction+=
            weight*value.diagrams.B321II_uv_subtraction;
        result.diagrams.B321II_uv_restoration+=
            weight*value.diagrams.B321II_uv_restoration;
        result.diagrams.B411_uv_restoration+=
            weight*value.diagrams.B411_uv_restoration;
        for (std::size_t index=0;
             index<kCountertermCount;++index) {
            result.counterterms.BctrI[index]+=
                weight*value.counterterms.BctrI[index];
            result.counterterms.BctrII[index]+=
                weight*value.counterterms.BctrII[index];
            result.counterterms.total[index]+=
                weight*value.counterterms.total[index];
        }
        for (std::size_t index=0;
             index<kStochasticCount;++index) {
            result.stochastic.pure[index]+=
                weight*value.stochastic.pure[index];
            result.stochastic.mixed_tree[index]+=
                weight*value.stochastic.mixed_tree[index];
            result.stochastic.mixed_one_loop[index]+=
                weight*value.stochastic.mixed_one_loop[index];
            result.stochastic.derivative_mixed[index]+=
                weight*value.stochastic.derivative_mixed[index];
            result.stochastic.raw[index]+=
                weight*value.stochastic.raw[index];
        }
        result.stochastic.bshot_bnabla2_cross+=
            weight*value.stochastic.bshot_bnabla2_cross;
    } else {
        result.diagrams.B222+=weight*value.diagrams.B222;
        result.diagrams.B321I+=weight*value.diagrams.B321I;
        result.diagrams.B411_bare+=
            weight*value.diagrams.B411_bare;
    }
    result.total_loop_nodes+=value.diagrams.integration_nodes;
}

EftShellTemplates compute_multilevel_exact_lattice_shell(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& provider,
    const FftlogDrOracle& b411_oracle,
    const HybridShellIntegrationConfig& loop_config,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    const StochasticIntegrationConfig& stochastic_config,
    double k_nl) {
    if (loop_config.analytic_convolutions) {
        throw std::invalid_argument(
            "exact-lattice multilevel projection requires "
            "the hybrid direct B222/B321I path");
    }
    if (shell_config.radial_measure
            !=marisa_b_shell_v1::ShellRadialMeasure::FftLattice
        ||shell_config.average_grid_orientation
        ||shell_config.n_radial<1
        ||shell_config.n_internal_mu<1
        ||loop_config.expensive_lattice_radial_order<1
        ||loop_config.expensive_k3_interpolation_order<1) {
        throw std::invalid_argument(
            "invalid exact-lattice multilevel shell configuration");
    }
    ExactLatticeShellRuleConfig cheap_config;
    cheap_config.radial_order=shell_config.n_radial;
    cheap_config.angular_order=shell_config.n_internal_mu;
    cheap_config.fft_box_size=shell_config.fft_box_size;
    cheap_config.fft_mesh_size=shell_config.fft_mesh_size;
    const ExactLatticeShellRule cheap=
        make_exact_lattice_shell_rule(bin,cheap_config);

    MultilevelLatticeShellRuleConfig expensive_config;
    expensive_config.exact=cheap_config;
    expensive_config.exact.radial_order=
        loop_config.expensive_lattice_radial_order;
    expensive_config.interpolation_order=
        loop_config.expensive_k3_interpolation_order;
    const MultilevelLatticeShellRule expensive=
        make_multilevel_lattice_shell_rule(
            bin,expensive_config);
    if (cheap.diagnostics.first_full_modes
            !=expensive.diagnostics.exact.first_full_modes
        ||cheap.diagnostics.second_full_modes
            !=expensive.diagnostics.exact.second_full_modes
        ||cheap.diagnostics.zero_external_leg_pairs
            !=expensive.diagnostics.exact.zero_external_leg_pairs
        ||cheap.diagnostics.closing_zero_pairs
            !=expensive.diagnostics.exact.closing_zero_pairs) {
        throw std::logic_error(
            "cheap and expensive exact-lattice measures disagree");
    }

    const std::vector<double> unique_k=
        unique_wavenumbers(cheap.nodes);
    const std::map<double,std::size_t> unique_index=
        wavenumber_index(unique_k);
    const auto cross_power=
        parallel_compute<MixedStochasticPowerTemplates>(
            unique_k.size(),[&](std::size_t index) {
                return mixed_stochastic_cross_power(
                    loop_power,unique_k[index],
                    stochastic_config);
            });
    const auto p13_cache=
        parallel_compute<RenormalizedTracerP13Template>(
            unique_k.size(),[&](std::size_t index) {
                return renormalized_tracer_p13_template(
                    loop_power,unique_k[index],provider,
                    loop_config.factorized_p13);
            });
    std::uint64_t cached_p13_nodes=0;
    for (const auto& value:p13_cache) {
        cached_p13_nodes+=value.integration_nodes;
    }
    const double uv_moment=
        loop_config.analytic.uv_restoration.enabled
        ?fftlog_normalized_linear_power_uv_moment(
            loop_power,
            loop_config.analytic.uv_restoration)
        :0.0;
    const auto cheap_values=parallel_compute<NodeTemplates>(
        cheap.nodes.size(),[&](std::size_t index) {
            const auto& triangle=
                cheap.nodes[index].closed_vectors;
            NodeTemplates value;
            const marisa_b_halo_v1::CanonicalTriangle canonical=
                marisa_b_halo_v1::canonicalize_closed_vectors(
                    triangle);
            const std::array<Vec3,3> canonical_external={{
                canonical.k1,canonical.k2,canonical.k3}};
            std::array<
                RenormalizedTracerP13Template,3> legs;
            std::array<
                MixedStochasticPowerTemplates,3> cross;
            for (int leg=0;leg<3;++leg) {
                const double k=marisa_b_halo_v1::norm(
                    canonical_external[
                        static_cast<std::size_t>(leg)]);
                legs[static_cast<std::size_t>(leg)]=
                    p13_cache.at(unique_index.at(k));
            }
            for (int leg=0;leg<3;++leg) {
                const double k=marisa_b_halo_v1::norm(
                    triangle[static_cast<std::size_t>(leg)]);
                cross[static_cast<std::size_t>(leg)]=
                    cross_power.at(unique_index.at(k));
            }
            const FactorizedB321IITemplates factorized=
                assemble_factorized_b321ii_templates(
                    loop_power,triangle,provider,legs);
            value.diagrams.B321II=factorized.value;
            value.diagrams.B321II_bare=factorized.bare;
            value.diagrams.B321II_uv_subtraction=
                factorized.bias_subtraction;
            value.diagrams.B321II_uv_restoration=
                factorized.uv_tail_restoration;
            value.diagrams.B411_uv_restoration=
                loop_config.analytic.uv_restoration.enabled
                ?fftlog_b411_exact_uv_restoration(
                    loop_power,triangle,provider,
                    uv_moment)
                :SparsePolynomial();
            const DiagramAssembler tree_assembler(
                tree_power,triangle,provider);
            value.diagrams.tree=tree_assembler.tree();
            value.counterterms=counterterm_bispectrum(
                triangle,external_power(loop_power,triangle),
                k_nl);
            value.stochastic=stochastic_bispectrum(
                loop_power,triangle,k_nl,
                stochastic_config,&cross,&tree_power);
            return value;
        });

    DirectIntegrationConfig convolution=
        loop_config.convolution;
    convolution.diagram_mask=kDirectB222|kDirectB321I;
    const double coarse_epsilon=
        loop_config.analytic.regulator.coarse_epsilon;
    const double ratio=
        loop_config.analytic.regulator.refinement_ratio;
    const double fine_epsilon=coarse_epsilon/ratio;
    const auto expensive_values=
        parallel_compute<NodeTemplates>(
            expensive.nodes.size(),[&](std::size_t index) {
                const auto& triangle=
                    expensive.nodes[index].closed_vectors;
                NodeTemplates value;
                value.diagrams=evaluate_direct(
                    loop_power,triangle,provider,
                    convolution);
                const FftlogAnalyticDiagramResult coarse=
                    b411_oracle.evaluate_b411_analytic(
                        triangle,provider,{},
                        loop_config.analytic.master,
                        loop_config
                            .analytic
                            .mode_coefficient_relative_tolerance,
                        coarse_epsilon);
                const FftlogAnalyticDiagramResult fine=
                    b411_oracle.evaluate_b411_analytic(
                        triangle,provider,{},
                        loop_config.analytic.master,
                        loop_config
                            .analytic
                            .mode_coefficient_relative_tolerance,
                        fine_epsilon);
                value.diagrams.B411_bare=
                    (1.0/(ratio-1.0))
                    *(ratio*fine.value-coarse.value);
                value.diagrams.integration_nodes+=
                    coarse.master_integral_evaluations
                    +fine.master_integral_evaluations;
                return value;
            });

    EftShellTemplates result;
    result.bin=bin;
    result.shell_nodes=
        cheap.nodes.size()+expensive.nodes.size();
    result.cheap_shell_nodes=cheap.nodes.size();
    result.expensive_shell_nodes=expensive.nodes.size();
    result.cached_p13_nodes=cached_p13_nodes;
    result.total_loop_nodes=cached_p13_nodes;
    result.fft_lattice_radial_measure=true;
    result.hybrid_factorized_analytic_tadpoles=true;
    result.fftlog_analytic_convolutions=false;
    result.exact_joint_lattice_measure=true;
    result.multilevel_external_projection=true;
    result.exact_lattice_radial_order=
        cheap_config.radial_order;
    result.exact_lattice_angular_order=
        cheap_config.angular_order;
    result.expensive_lattice_radial_order=
        expensive_config.exact.radial_order;
    result.expensive_k3_interpolation_order=
        expensive_config.interpolation_order;
    result.zero_external_leg_pairs=
        cheap.diagnostics.zero_external_leg_pairs;
    result.closing_zero_pairs=
        cheap.diagnostics.closing_zero_pairs;
    result.exact_lattice_valid_pair_fraction=
        cheap.diagnostics.valid_pair_fraction;
    result.exact_lattice_total_variation=
        cheap.diagnostics.total_variation;
    result.expensive_total_variation=
        expensive.diagnostics.total_variation;
    result.expensive_maximum_sampled_lebesgue=
        expensive.diagnostics.maximum_sampled_lebesgue;
    result.diagrams.ir_safe=convolution.ir_safe;
    for (std::size_t index=0;
         index<cheap.nodes.size();++index) {
        accumulate_multilevel_node(
            result,cheap_values[index],
            cheap.nodes[index].weight,true);
    }
    for (std::size_t index=0;
         index<expensive.nodes.size();++index) {
        accumulate_multilevel_node(
            result,expensive_values[index],
            expensive.nodes[index].weight,false);
    }
    result.diagrams.B411=
        result.diagrams.B411_bare
        +result.diagrams.B411_uv_restoration;
    result.diagrams.uv_subtraction_total=
        result.diagrams.B321II_uv_subtraction
        +result.diagrams.B411_uv_subtraction;
    result.diagrams.uv_restoration_total=
        result.diagrams.B321II_uv_restoration
        +result.diagrams.B411_uv_restoration;
    result.diagrams.one_loop=
        result.diagrams.B222
        +result.diagrams.B321I
        +result.diagrams.B321II
        +result.diagrams.B411;
    result.diagrams.total=
        result.diagrams.tree
        +result.diagrams.one_loop;
    result.diagrams.integration_nodes=
        result.total_loop_nodes;
    return result;
}

}  // namespace

EftShellTemplates compute_eft_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& loop_config,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    const StochasticIntegrationConfig& stochastic_config,
    double k_nl) {
    const std::vector<marisa_b_shell_v1::ShellNode> nodes=
        marisa_b_shell_v1::make_shell_nodes(bin,shell_config);
    if (nodes.empty()) throw std::runtime_error("EFT-v2 shell has no quadrature nodes");

    const std::vector<double> unique_k=
        unique_wavenumbers(nodes);
    const auto cross_power=parallel_compute<MixedStochasticPowerTemplates>(
        unique_k.size(),[&](std::size_t index) {
            return mixed_stochastic_cross_power(
                loop_power,unique_k[index],stochastic_config);
        });
    const std::map<double,std::size_t> cross_index=
        wavenumber_index(unique_k);

    const auto values=parallel_compute<NodeTemplates>(nodes.size(),[&](std::size_t index) {
        const auto& triangle=nodes[index].closed_vectors;
        NodeTemplates value;
        value.diagrams=evaluate_direct(loop_power,triangle,provider,loop_config);
        const DiagramAssembler tree_assembler(tree_power,triangle,provider);
        value.diagrams.tree=tree_assembler.tree();
        value.diagrams.total=value.diagrams.tree+value.diagrams.one_loop;
        value.counterterms=counterterm_bispectrum(
            triangle,external_power(loop_power,triangle),k_nl);
        std::array<MixedStochasticPowerTemplates,3> cross={};
        for (int leg=0;leg<3;++leg) {
            const double k=marisa_b_halo_v1::norm(triangle[leg]);
            cross[leg]=cross_power.at(cross_index.at(k));
        }
        value.stochastic=stochastic_bispectrum(
            loop_power,triangle,k_nl,stochastic_config,&cross,&tree_power);
        return value;
    });

    return assemble_shell_templates(
        bin,nodes,values,shell_config,loop_config.ir_safe);
}

EftShellTemplates compute_reconstructed_eft_shell_templates(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& base_provider,
    const marisa_b_halo_v1::ReconstructionConfig& reconstruction,
    const ReconstructedShellIntegrationConfig& integration,
    const StochasticIntegrationConfig& stochastic_config,
    double k_nl) {
    if (!reconstruction.enabled) {
        throw std::invalid_argument(
            "post shell projection requires enabled reconstruction");
    }
    const HaarOrientedLatticeShellRule rule=
        make_haar_oriented_lattice_shell_rule(
            bin,integration.shell);
    if (rule.nodes.empty()) {
        throw std::runtime_error(
            "post EFT shell has no Haar-oriented lattice nodes");
    }
    const ReconstructedFieldKernelProvider provider(
        base_provider,reconstruction);
    const auto values=parallel_compute<NodeTemplates>(
        rule.nodes.size(),[&](std::size_t index) {
            const auto& triangle=
                rule.nodes[index].closed_vectors;
            NodeTemplates value;
            value.diagrams=evaluate_direct(
                loop_power,triangle,provider,
                integration.loop);
            const DiagramAssembler tree_assembler(
                tree_power,triangle,provider);
            value.diagrams.tree=tree_assembler.tree();
            value.diagrams.total=
                value.diagrams.tree
                +value.diagrams.one_loop;
            value.counterterms=
                reconstructed_counterterm_bispectrum(
                    base_provider,reconstruction,triangle,
                    external_power(loop_power,triangle),
                    k_nl);
            if (integration.include_reconstructed_stochastic) {
                const ReconstructedStochasticTemplates
                    stochastic=
                    reconstructed_stochastic_bispectrum_decomposition(
                        loop_power,triangle,base_provider,
                        reconstruction,k_nl,
                        stochastic_config,&tree_power);
                value.stochastic=stochastic.tied;
                value.stochastic_density_only=
                    stochastic.density_only;
                value.stochastic_noisy_shift=
                    stochastic.noisy_shift;
            }
            if (integration.include_reconstructed_fixed_poisson) {
                value.fixed_poisson=
                    reconstructed_fixed_poisson_bispectrum(
                        loop_power,triangle,base_provider,
                        reconstruction,stochastic_config,
                        ReconstructedStochasticMap::
                            TiedDensityAndShift,
                        PoissonIntensityMode::
                            ConditionalTracer,
                        integration
                            .include_bare_fixed_poisson_one_loop);
            }
            return value;
        });

    marisa_b_shell_v1::ShellQuadratureConfig metadata;
    metadata.radial_measure=
        marisa_b_shell_v1::ShellRadialMeasure::FftLattice;
    metadata.fft_box_size=
        integration.shell.exact.fft_box_size;
    metadata.fft_mesh_size=
        integration.shell.exact.fft_mesh_size;
    EftShellTemplates result=assemble_shell_templates(
        bin,rule.nodes,values,metadata,
        integration.loop.ir_safe);
    /*
     * The k1-k2-mu marginal, zero-mode atoms, and estimator denominator are
     * inherited from the exact FFT-lattice rule.  The conditional orientation
     * about the cubic mesh is a converged Haar cubature, not a literal
     * enumeration of every cubic-lattice orbit.
     */
    result.exact_joint_lattice_measure=false;
    result.exact_k1_k2_mu_lattice_measure=true;
    result.conditional_cubic_orientation_exact=false;
    result.haar_oriented_cic_projection=true;
    result.reconstructed_counterterms=true;
    result.reconstructed_stochastic=
        integration.include_reconstructed_stochastic;
    result.reconstructed_stochastic_decomposed=
        integration.include_reconstructed_stochastic;
    result.reconstructed_fixed_poisson=
        integration.include_reconstructed_fixed_poisson;
    result.invariant_shell_nodes=
        rule.diagnostics.invariant_nodes;
    result.orientation_nodes=
        rule.diagnostics.orientation_nodes;
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

EftShellTemplates compute_eft_shell_templates_hybrid(
    const PowerSpectrum& loop_power,
    const PowerSpectrum& tree_power,
    const marisa_b_shell_v1::ShellBin& bin,
    const FieldKernelProvider& provider,
    const FftlogDrOracle& b411_oracle,
    const HybridShellIntegrationConfig& loop_config,
    const marisa_b_shell_v1::ShellQuadratureConfig& shell_config,
    const StochasticIntegrationConfig& stochastic_config,
    double k_nl) {
    if (loop_config.exact_lattice_multilevel) {
        return compute_multilevel_exact_lattice_shell(
            loop_power,tree_power,bin,provider,b411_oracle,
            loop_config,shell_config,stochastic_config,k_nl);
    }
    if (!(loop_config.analytic.regulator.refinement_ratio>1.0
          &&loop_config.analytic.regulator.coarse_epsilon>0.0)) {
        throw std::invalid_argument(
            "hybrid shell has an invalid analytic regulator");
    }
    const std::vector<marisa_b_shell_v1::ShellNode> nodes=
        marisa_b_shell_v1::make_shell_nodes(bin,shell_config);
    if (nodes.empty()) {
        throw std::runtime_error(
            "EFT-v2 hybrid shell has no quadrature nodes");
    }
    const std::vector<double> unique_k=
        unique_wavenumbers(nodes);
    const std::map<double,std::size_t> unique_index=
        wavenumber_index(unique_k);
    const auto cross_power=
        parallel_compute<MixedStochasticPowerTemplates>(
            unique_k.size(),[&](std::size_t index) {
                return mixed_stochastic_cross_power(
                    loop_power,unique_k[index],
                    stochastic_config);
            });
    const auto p13_cache=
        parallel_compute<RenormalizedTracerP13Template>(
            unique_k.size(),[&](std::size_t index) {
                return renormalized_tracer_p13_template(
                    loop_power,unique_k[index],provider,
                    loop_config.factorized_p13);
            });
    std::uint64_t cached_p13_nodes=0;
    for (const auto& value:p13_cache) {
        cached_p13_nodes+=value.integration_nodes;
    }
    const double uv_moment=
        loop_config.analytic.uv_restoration.enabled
        ?fftlog_normalized_linear_power_uv_moment(
            loop_power,
            loop_config.analytic.uv_restoration)
        :0.0;
    DirectIntegrationConfig convolution=
        loop_config.convolution;
    convolution.diagram_mask=kDirectB222|kDirectB321I;
    const double coarse_epsilon=
        loop_config.analytic.regulator.coarse_epsilon;
    const double ratio=
        loop_config.analytic.regulator.refinement_ratio;
    const double fine_epsilon=coarse_epsilon/ratio;
    const auto values=parallel_compute<NodeTemplates>(
        nodes.size(),[&](std::size_t index) {
            const auto& triangle=
                nodes[index].closed_vectors;
            NodeTemplates value;
            if (!loop_config.analytic_convolutions) {
                value.diagrams=evaluate_direct(
                    loop_power,triangle,provider,
                    convolution);
            }
            const marisa_b_halo_v1::CanonicalTriangle canonical=
                marisa_b_halo_v1::canonicalize_closed_vectors(
                    triangle);
            const std::array<Vec3,3> canonical_external={{
                canonical.k1,canonical.k2,canonical.k3}};
            std::array<
                RenormalizedTracerP13Template,3> legs;
            std::array<
                MixedStochasticPowerTemplates,3> cross;
            for (int leg=0;leg<3;++leg) {
                const double k=marisa_b_halo_v1::norm(
                    canonical_external[
                        static_cast<std::size_t>(leg)]);
                const std::size_t cache_index=
                    unique_index.at(k);
                legs[static_cast<std::size_t>(leg)]=
                    p13_cache.at(cache_index);
            }
            for (int leg=0;leg<3;++leg) {
                const double k=marisa_b_halo_v1::norm(
                    triangle[static_cast<std::size_t>(leg)]);
                const std::size_t cache_index=
                    unique_index.at(k);
                cross[static_cast<std::size_t>(leg)]=
                    cross_power.at(cache_index);
            }
            const FactorizedB321IITemplates factorized=
                assemble_factorized_b321ii_templates(
                    loop_power,triangle,provider,legs);
            value.diagrams.B321II=factorized.value;
            value.diagrams.B321II_bare=factorized.bare;
            value.diagrams.B321II_uv_subtraction=
                factorized.bias_subtraction;
            value.diagrams.B321II_uv_restoration=
                factorized.uv_tail_restoration;
            value.diagrams.B411_uv_restoration=
                loop_config
                    .analytic.uv_restoration.enabled
                ?fftlog_b411_exact_uv_restoration(
                    loop_power,triangle,provider,
                    uv_moment)
                :SparsePolynomial();
            if (loop_config.analytic_convolutions) {
                const FftlogAnalyticConvolutionResult analytic=
                    b411_oracle.evaluate_convolutions_analytic(
                        triangle,provider,
                        loop_config.analytic);
                value.diagrams.B222=analytic.B222.value;
                value.diagrams.B321I=
                    analytic.B321I.extrapolated.value;
                value.diagrams.B411_bare=
                    analytic.B411.extrapolated.value;
                value.diagrams.B411=
                    value.diagrams.B411_bare
                    +value.diagrams.B411_uv_restoration;
                value.diagrams.integration_nodes=
                    analytic.master_integral_evaluations;
                value.diagrams.ir_safe=false;
            } else {
                const FftlogAnalyticDiagramResult coarse=
                    b411_oracle.evaluate_b411_analytic(
                        triangle,provider,{},
                        loop_config.analytic.master,
                        loop_config
                            .analytic
                            .mode_coefficient_relative_tolerance,
                        coarse_epsilon);
                const FftlogAnalyticDiagramResult fine=
                    b411_oracle.evaluate_b411_analytic(
                        triangle,provider,{},
                        loop_config.analytic.master,
                        loop_config
                            .analytic
                            .mode_coefficient_relative_tolerance,
                        fine_epsilon);
                value.diagrams.B411=
                    (1.0/(ratio-1.0))
                        *(ratio*fine.value-coarse.value)
                    +value.diagrams.B411_uv_restoration;
                value.diagrams.B411_bare=
                    value.diagrams.B411
                    -value.diagrams.B411_uv_restoration;
                value.diagrams.integration_nodes+=
                    coarse.master_integral_evaluations
                    +fine.master_integral_evaluations;
            }
            value.diagrams.one_loop=
                value.diagrams.B222
                +value.diagrams.B321I
                +value.diagrams.B321II
                +value.diagrams.B411;
            const DiagramAssembler tree_assembler(
                tree_power,triangle,provider);
            value.diagrams.tree=
                tree_assembler.tree();
            value.diagrams.total=
                value.diagrams.tree
                +value.diagrams.one_loop;
            value.counterterms=counterterm_bispectrum(
                triangle,
                external_power(loop_power,triangle),
                k_nl);
            value.stochastic=stochastic_bispectrum(
                loop_power,triangle,k_nl,
                stochastic_config,&cross,&tree_power);
            return value;
        });
    return assemble_shell_templates(
        bin,nodes,values,shell_config,
        loop_config.analytic_convolutions
            ?false:convolution.ir_safe,
        true,cached_p13_nodes,
        loop_config.analytic_convolutions);
}

EftShellValues evaluate_eft_shell_templates(
    const EftShellTemplates& templates,
    const std::array<double,kParameterCount>& parameters,
    double number_density) {
    EftShellValues result;
    result.bin=templates.bin;
    result.tree=templates.diagrams.tree.evaluate(parameters);
    result.B222=templates.diagrams.B222.evaluate(parameters);
    result.B321I=templates.diagrams.B321I.evaluate(parameters);
    result.B321II=templates.diagrams.B321II.evaluate(parameters);
    result.B411=templates.diagrams.B411.evaluate(parameters);
    result.renormalized_loop=result.B222+result.B321I+result.B321II+result.B411;
    result.counterterm_shapes=templates.counterterms.linear_shapes(parameters);
    result.counterterm=templates.counterterms.evaluate(parameters);
    result.stochastic_shapes=templates.stochastic.linear_shapes(parameters,number_density);
    result.stochastic=templates.stochastic.evaluate(parameters,number_density);
    result.total=result.tree+result.renormalized_loop+result.counterterm+result.stochastic;
    return result;
}

}  // namespace marisa_b_eft_v2
