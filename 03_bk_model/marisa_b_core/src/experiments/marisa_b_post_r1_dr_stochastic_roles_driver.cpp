/*
 * Additive D/R leading-stochastic and fixed-Poisson shell driver.
 *
 * Only the global Ashot_residual and Bshot_residual nuisance directions are
 * exported.  Halo and finite-random Poisson contributions are fixed
 * diagnostics, with n_random=Nran*n_halo kept explicit.  Their historical
 * theory-filter complement is not identified with the measured jaxpower
 * fifth channel, so none is activated in the frozen model.  No loop
 * integrator is called and no historical tied production API is changed.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "PowerSpectrum.h"
#include "dr_stochastic_roles.h"
#include "lattice_shell_rule.h"
#include "parameter_registry.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace shell=marisa_b_shell_v1;

namespace {

constexpr std::size_t kRoleCount=eft::kSignedDrRoleCount;
const std::array<const char*,kRoleCount> kRoleNames{{
    "DDD","DDR","DRR","RRR","tied_F"
}};

int parse_int(const char* text,const std::string& label,int minimum=1) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size() ||value<minimum) {
        throw std::invalid_argument("invalid integer for "+label);
    }
    return value;
}

double parse_double(const char* text,const std::string& label) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size() ||!std::isfinite(value)) {
        throw std::invalid_argument("invalid number for "+label);
    }
    return value;
}

std::string file_sha256(const std::string& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash input: "+path);
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return eft::sha256_hex(payload);
}

class LogTable final:public PowerSpectrum {
public:
    LogTable(const std::string& path,int column,int required_columns):
        column_(column),required_columns_(required_columns) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() ||line[0]=='#') continue;
            std::istringstream row(line);
            std::vector<double> values(
                static_cast<std::size_t>(required_columns_),0.0);
            bool valid=true;
            for (double& value:values) {
                if (!(row>>value)) {
                    valid=false;
                    break;
                }
            }
            if (!valid ||!(values[0]>0.0)
                ||!(values[static_cast<std::size_t>(column_)]>0.0)) {
                throw std::runtime_error("invalid positive power-table row");
            }
            log_k_.push_back(std::log(values[0]));
            log_value_.push_back(std::log(
                values[static_cast<std::size_t>(column_)]));
        }
        if (log_k_.size()<2) throw std::runtime_error("power table is too short");
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) {
            right=log_k_.size()-1;
        } else if (x>log_k_.front()) {
            right=static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(),log_k_.end(),x)
                -log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])
            /(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]
            +fraction*(log_value_[right]-log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("LogTable has no Cosmology object");
    }
private:
    int column_;
    int required_columns_;
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open edge table: "+path);
    std::vector<double> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() ||line[0]=='#') continue;
        std::istringstream row(line);
        double value=0.0;
        if (!(row>>value)) throw std::runtime_error("invalid edge row");
        result.push_back(value);
    }
    if (result.size()!=16) {
        throw std::runtime_error("D/R stochastic driver requires 16 edges");
    }
    return result;
}

std::pair<int,int> pair_from_flat(int target) {
    int flat=0;
    for (int first=0;first<15;++first) {
        for (int second=first;second<15;++second,++flat) {
            if (flat==target) return {first,second};
        }
    }
    throw std::out_of_range("flat B000 index is outside [0,120)");
}

struct Comparison {
    double absolute=0.0;
    double scale=0.0;
    double relative() const {
        return absolute/std::max(scale,1.0e-300);
    }
};

Comparison compare(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected) {
    Comparison result;
    const eft::SparsePolynomial difference=actual-expected;
    for (const auto& term:difference.terms()) {
        result.absolute=std::max(result.absolute,std::fabs(term.second));
    }
    for (const auto& term:actual.terms()) {
        result.scale=std::max(result.scale,std::fabs(term.second));
    }
    for (const auto& term:expected.terms()) {
        result.scale=std::max(result.scale,std::fabs(term.second));
    }
    return result;
}

double polynomial_scale(const eft::SparsePolynomial& value) {
    double result=0.0;
    for (const auto& term:value.terms()) {
        result=std::max(result,std::fabs(term.second));
    }
    return result;
}

double role_scale(const eft::SignedDrRoleTemplates& roles) {
    double result=0.0;
    for (const auto& role:roles.value) {
        result=std::max(result,polynomial_scale(role));
    }
    return result;
}

void require_close(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    const std::string& label) {
    const Comparison value=compare(actual,expected);
    if (value.absolute>2.0e-9 &&value.relative()>2.0e-12) {
        throw std::runtime_error(
            label+" failed: abs="+std::to_string(value.absolute)
            +", rel="+std::to_string(value.relative()));
    }
}

void add_weighted(
    eft::SignedDrRoleTemplates& destination,
    const eft::SignedDrRoleTemplates& source,
    double weight) {
    for (std::size_t role=0;role<kRoleCount;++role) {
        destination.value[role]+=weight*source.value[role];
    }
}

void add_weighted(
    eft::DrFixedPoissonSpeciesTemplates& destination,
    const eft::DrFixedPoissonSpeciesTemplates& source,
    double weight) {
    for (std::size_t inverse=1;inverse<3;++inverse) {
        add_weighted(
            destination.raw_by_inverse_density[inverse],
            source.raw_by_inverse_density[inverse],weight);
        destination.legacy_theory_filter_complement[inverse]+=
            weight*source.legacy_theory_filter_complement[inverse];
        destination.legacy_theory_retained[inverse]+=
            weight*source.legacy_theory_retained[inverse];
    }
}

void write_json_string(const std::string& value) {
    std::cout<<'"';
    for (char character:value) {
        if (character=='"' ||character=='\\') std::cout<<'\\';
        std::cout<<character;
    }
    std::cout<<'"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout<<'{';
    bool first=true;
    for (const auto& term:polynomial.terms()) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(term.first.canonical_string());
        std::cout<<':'<<term.second;
    }
    std::cout<<'}';
}

void write_roles(const eft::SignedDrRoleTemplates& roles) {
    std::cout<<'{';
    for (std::size_t role=0;role<kRoleCount;++role) {
        if (role) std::cout<<',';
        write_json_string(kRoleNames[role]);
        std::cout<<':';
        write_polynomial(roles.value[role]);
    }
    std::cout<<'}';
}

void write_fixed_species(
    const eft::DrFixedPoissonSpeciesTemplates& species) {
    std::cout<<"{\"raw_zero_shot_roles\":{";
    for (std::size_t inverse=1;inverse<3;++inverse) {
        if (inverse>1) std::cout<<',';
        write_json_string("inverse_density_"+std::to_string(inverse));
        std::cout<<':';
        write_roles(species.raw_by_inverse_density[inverse]);
    }
    std::cout<<"},\"legacy_theory_filter_complement\":{";
    for (std::size_t inverse=1;inverse<3;++inverse) {
        if (inverse>1) std::cout<<',';
        write_json_string("inverse_density_"+std::to_string(inverse));
        std::cout<<':';
        write_polynomial(
            species.legacy_theory_filter_complement[inverse]);
    }
    std::cout<<"},\"legacy_theory_retained\":{";
    for (std::size_t inverse=1;inverse<3;++inverse) {
        if (inverse>1) std::cout<<',';
        write_json_string("inverse_density_"+std::to_string(inverse));
        std::cout<<':';
        write_polynomial(species.legacy_theory_retained[inverse]);
    }
    std::cout<<"},\"frozen_legacy_exclusion\":{";
    for (std::size_t inverse=1;inverse<3;++inverse) {
        if (inverse>1) std::cout<<',';
        write_json_string("inverse_density_"+std::to_string(inverse));
        std::cout<<':';
        write_polynomial(-1.0*species.legacy_theory_retained[inverse]);
    }
    std::cout<<"},\"frozen_active_tied\":{"
             <<"\"inverse_density_1\":{},"
             <<"\"inverse_density_2\":{}}}";
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=16) {
            std::cerr
                <<"usage: marisa_b_post_r1_dr_stochastic_roles_driver "
                <<"POWER_TABLE PNG_TABLE EDGE_FILE FLAT_INDEX SHELL_NRAD SHELL_NMU "
                <<"N_ALPHA N_COS_BETA N_GAMMA R_SMOOTH B_REC CELL_SIZE "
                <<"CIC_POWER NRAN P_BPHI\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string png_path=argv[2];
        const std::string edge_path=argv[3];
        const int flat=parse_int(argv[4],"flat index",0);
        if (flat>=120) throw std::invalid_argument("flat index must be <120");
        eft::HaarOrientedLatticeShellRuleConfig shell_config;
        shell_config.exact.radial_order=parse_int(argv[5],"shell nrad");
        shell_config.exact.angular_order=parse_int(argv[6],"shell nmu");
        shell_config.exact.fft_box_size=1000.0;
        shell_config.exact.fft_mesh_size=256;
        shell_config.n_alpha=parse_int(argv[7],"alpha order");
        shell_config.n_cos_beta=parse_int(argv[8],"cos-beta order");
        shell_config.n_gamma=parse_int(argv[9],"gamma order");
        hv1::ReconstructionConfig reconstruction;
        reconstruction.enabled=true;
        reconstruction.smoothing_radius=parse_double(argv[10],"R smooth");
        reconstruction.bias_recon=parse_double(argv[11],"b rec");
        reconstruction.cell_size=parse_double(argv[12],"cell size");
        const int cic_power=parse_int(argv[13],"CIC power",0);
        const int nran=parse_int(argv[14],"Nran");
        const double p_bphi=parse_double(argv[15],"p bphi");
        if (!(reconstruction.smoothing_radius>0.0)
            ||!(reconstruction.bias_recon>0.0)
            ||reconstruction.cell_size<0.0
            ||(reconstruction.cell_size>0.0 &&cic_power!=4)
            ||(reconstruction.cell_size==0.0 &&cic_power!=0)) {
            throw std::invalid_argument("invalid reconstruction configuration");
        }

        const std::vector<double> edges=read_edges(edge_path);
        const auto pair=pair_from_flat(flat);
        const shell::ShellBin bin{
            edges[static_cast<std::size_t>(pair.first)],
            edges[static_cast<std::size_t>(pair.first+1)],
            edges[static_cast<std::size_t>(pair.second)],
            edges[static_cast<std::size_t>(pair.second+1)]};
        const eft::HaarOrientedLatticeShellRule rule=
            eft::make_haar_oriented_lattice_shell_rule(bin,shell_config);
        const LogTable power(power_path,1,2);
        const LogTable transfer(png_path,3,4);
        const eft::EftBiasKernelProvider base;
        const eft::LocalPngHaloTreeDirectionProvider png_direction(
            transfer,p_bphi,1.0,
            eft::LocalPngTreeDirectionSector::ExternalBphi);
        eft::RandomCatalogConfig continuum_config;
        continuum_config.convention=eft::RandomCatalogConvention::Continuum;
        continuum_config.random_to_halo_density=static_cast<double>(nran);
        eft::RandomCatalogConfig finite_config;
        finite_config.convention=
            eft::RandomCatalogConvention::FiniteIndependentPoisson;
        finite_config.random_to_halo_density=static_cast<double>(nran);

        struct ThreadAccumulation {
            eft::DrLeadingStochasticPoissonTemplates accumulated;
            eft::DrLeadingStochasticPngResponses accumulated_response;
            double maximum_A_role_abs=0.0;
            double maximum_A_role_rel=0.0;
            double maximum_B_role_abs=0.0;
            double maximum_B_role_rel=0.0;
            double maximum_fixed_abs=0.0;
            double maximum_fixed_rel=0.0;
            double maximum_continuum_random=0.0;
            double maximum_response_A_abs=0.0;
            double maximum_response_B_abs=0.0;
            double maximum_response_B_rel=0.0;
            double maximum_response_fixed_abs=0.0;
            double maximum_response_fixed_rel=0.0;
        };
        const auto process_node=[&](
            const shell::ShellNode& node,
            ThreadAccumulation& local) {
            const auto continuum=eft::dr_leading_stochastic_poisson_tree(
                power,node.closed_vectors,base,reconstruction,
                continuum_config);
            const auto finite=eft::dr_leading_stochastic_poisson_tree(
                power,node.closed_vectors,base,reconstruction,
                finite_config);
            const auto response=eft::dr_leading_stochastic_png_response(
                power,node.closed_vectors,base,png_direction,
                reconstruction,finite_config);
            require_close(
                finite.Ashot_residual[eft::SignedDrRole::TiedF],
                eft::signed_dr_role_sum(finite.Ashot_residual),
                "node Ashot signed role sum");
            require_close(
                finite.Bshot_residual[eft::SignedDrRole::TiedF],
                eft::signed_dr_role_sum(finite.Bshot_residual),
                "node Bshot signed role sum");
            require_close(
                finite.Ashot_residual[eft::SignedDrRole::TiedF],
                continuum.Ashot_residual[eft::SignedDrRole::TiedF],
                "finite random Ashot invariance");
            require_close(
                finite.Bshot_residual[eft::SignedDrRole::TiedF],
                continuum.Bshot_residual[eft::SignedDrRole::TiedF],
                "finite random Bshot invariance");
            require_close(
                response.Ashot_residual[eft::SignedDrRole::TiedF],
                eft::signed_dr_role_sum(response.Ashot_residual),
                "node Ashot PNG response signed role sum");
            require_close(
                response.Bshot_residual[eft::SignedDrRole::TiedF],
                eft::signed_dr_role_sum(response.Bshot_residual),
                "node Bshot PNG response signed role sum");
            const Comparison A=compare(
                eft::signed_dr_role_sum(finite.Ashot_residual),
                finite.Ashot_residual[eft::SignedDrRole::TiedF]);
            const Comparison B=compare(
                eft::signed_dr_role_sum(finite.Bshot_residual),
                finite.Bshot_residual[eft::SignedDrRole::TiedF]);
            local.maximum_A_role_abs=std::max(
                local.maximum_A_role_abs,A.absolute);
            local.maximum_A_role_rel=std::max(
                local.maximum_A_role_rel,A.relative());
            local.maximum_B_role_abs=std::max(
                local.maximum_B_role_abs,B.absolute);
            local.maximum_B_role_rel=std::max(
                local.maximum_B_role_rel,B.relative());
            local.maximum_response_A_abs=std::max(
                local.maximum_response_A_abs,
                polynomial_scale(
                    response.Ashot_residual[
                        eft::SignedDrRole::TiedF]));
            const Comparison response_B=compare(
                eft::signed_dr_role_sum(response.Bshot_residual),
                response.Bshot_residual[eft::SignedDrRole::TiedF]);
            local.maximum_response_B_abs=std::max(
                local.maximum_response_B_abs,response_B.absolute);
            local.maximum_response_B_rel=std::max(
                local.maximum_response_B_rel,response_B.relative());
            for (std::size_t inverse=1;inverse<3;++inverse) {
                for (const auto* species:{
                     &finite.halo_poisson,&finite.random_poisson}) {
                    const eft::SparsePolynomial reconstructed=
                        eft::signed_dr_role_sum(
                            species->raw_by_inverse_density[inverse])
                        +species
                            ->legacy_theory_filter_complement[inverse];
                    const Comparison closure=compare(
                        reconstructed,
                        species->legacy_theory_retained[inverse]);
                    local.maximum_fixed_abs=std::max(
                        local.maximum_fixed_abs,closure.absolute);
                    local.maximum_fixed_rel=std::max(
                        local.maximum_fixed_rel,
                        closure.absolute/std::max({
                            role_scale(
                                species->raw_by_inverse_density[inverse]),
                            polynomial_scale(
                                species
                                    ->legacy_theory_filter_complement[inverse]),
                            polynomial_scale(
                                species->legacy_theory_retained[inverse]),
                            1.0e-300}));
                    require_close(
                        reconstructed,
                        species->legacy_theory_retained[inverse],
                        "node fixed Poisson raw plus legacy filter");
                }
                for (const auto* species:{
                     &response.halo_poisson,&response.random_poisson}) {
                    const eft::SparsePolynomial reconstructed=
                        eft::signed_dr_role_sum(
                            species->raw_by_inverse_density[inverse])
                        +species
                            ->legacy_theory_filter_complement[inverse];
                    const Comparison closure=compare(
                        reconstructed,
                        species->legacy_theory_retained[inverse]);
                    local.maximum_response_fixed_abs=std::max(
                        local.maximum_response_fixed_abs,closure.absolute);
                    local.maximum_response_fixed_rel=std::max(
                        local.maximum_response_fixed_rel,
                        closure.absolute/std::max({
                            role_scale(
                                species->raw_by_inverse_density[inverse]),
                            polynomial_scale(
                                species
                                    ->legacy_theory_filter_complement[inverse]),
                            polynomial_scale(
                                species->legacy_theory_retained[inverse]),
                            1.0e-300}));
                    require_close(
                        reconstructed,
                        species->legacy_theory_retained[inverse],
                        "node fixed Poisson PNG raw plus legacy filter");
                }
                for (std::size_t role=0;role<kRoleCount;++role) {
                    for (const auto& term:
                         continuum.random_poisson
                             .raw_by_inverse_density[inverse]
                             .value[role].terms()) {
                        local.maximum_continuum_random=std::max(
                            local.maximum_continuum_random,
                            std::fabs(term.second));
                    }
                }
            }
            add_weighted(
                local.accumulated.Ashot_residual,
                finite.Ashot_residual,node.weight);
            add_weighted(
                local.accumulated.Bshot_residual,
                finite.Bshot_residual,node.weight);
            add_weighted(
                local.accumulated.halo_poisson,
                finite.halo_poisson,node.weight);
            add_weighted(
                local.accumulated.random_poisson,
                finite.random_poisson,node.weight);
            add_weighted(
                local.accumulated_response.Ashot_residual,
                response.Ashot_residual,node.weight);
            add_weighted(
                local.accumulated_response.Bshot_residual,
                response.Bshot_residual,node.weight);
            add_weighted(
                local.accumulated_response.halo_poisson,
                response.halo_poisson,node.weight);
            add_weighted(
                local.accumulated_response.random_poisson,
                response.random_poisson,node.weight);
        };

        const int maximum_threads=std::max(1,omp_get_max_threads());
        std::vector<ThreadAccumulation> thread_accumulations(
            static_cast<std::size_t>(maximum_threads));
        std::atomic<bool> parallel_failed{false};
        std::string parallel_error;
        int threads_used=1;
#pragma omp parallel shared(threads_used,parallel_error,parallel_failed)
        {
            const int thread=omp_get_thread_num();
            const int count=omp_get_num_threads();
#pragma omp single
            threads_used=count;
            const std::size_t begin=
                rule.nodes.size()*static_cast<std::size_t>(thread)
                /static_cast<std::size_t>(count);
            const std::size_t end=
                rule.nodes.size()*static_cast<std::size_t>(thread+1)
                /static_cast<std::size_t>(count);
            ThreadAccumulation& local=thread_accumulations.at(
                static_cast<std::size_t>(thread));
            for (std::size_t index=begin;index<end;++index) {
                if (parallel_failed.load(std::memory_order_relaxed)) break;
                try {
                    process_node(rule.nodes[index],local);
                } catch (const std::exception& error) {
#pragma omp critical(marisa_b_dr_stochastic_parallel_error)
                    {
                        if (!parallel_failed.load(
                                std::memory_order_relaxed)) {
                            parallel_error=error.what();
                            parallel_failed.store(
                                true,std::memory_order_relaxed);
                        }
                    }
                } catch (...) {
#pragma omp critical(marisa_b_dr_stochastic_parallel_error)
                    {
                        if (!parallel_failed.load(
                                std::memory_order_relaxed)) {
                            parallel_error="unknown per-node exception";
                            parallel_failed.store(
                                true,std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
        if (parallel_failed.load(std::memory_order_relaxed)) {
            throw std::runtime_error(
                "parallel shell evaluation failed: "+parallel_error);
        }

        ThreadAccumulation merged;
        merged.accumulated.random_catalog=finite_config;
        for (int thread=0;thread<threads_used;++thread) {
            const ThreadAccumulation& local=thread_accumulations.at(
                static_cast<std::size_t>(thread));
            add_weighted(
                merged.accumulated.Ashot_residual,
                local.accumulated.Ashot_residual,1.0);
            add_weighted(
                merged.accumulated.Bshot_residual,
                local.accumulated.Bshot_residual,1.0);
            add_weighted(
                merged.accumulated.halo_poisson,
                local.accumulated.halo_poisson,1.0);
            add_weighted(
                merged.accumulated.random_poisson,
                local.accumulated.random_poisson,1.0);
            add_weighted(
                merged.accumulated_response.Ashot_residual,
                local.accumulated_response.Ashot_residual,1.0);
            add_weighted(
                merged.accumulated_response.Bshot_residual,
                local.accumulated_response.Bshot_residual,1.0);
            add_weighted(
                merged.accumulated_response.halo_poisson,
                local.accumulated_response.halo_poisson,1.0);
            add_weighted(
                merged.accumulated_response.random_poisson,
                local.accumulated_response.random_poisson,1.0);
            merged.maximum_A_role_abs=std::max(
                merged.maximum_A_role_abs,local.maximum_A_role_abs);
            merged.maximum_A_role_rel=std::max(
                merged.maximum_A_role_rel,local.maximum_A_role_rel);
            merged.maximum_B_role_abs=std::max(
                merged.maximum_B_role_abs,local.maximum_B_role_abs);
            merged.maximum_B_role_rel=std::max(
                merged.maximum_B_role_rel,local.maximum_B_role_rel);
            merged.maximum_fixed_abs=std::max(
                merged.maximum_fixed_abs,local.maximum_fixed_abs);
            merged.maximum_fixed_rel=std::max(
                merged.maximum_fixed_rel,local.maximum_fixed_rel);
            merged.maximum_continuum_random=std::max(
                merged.maximum_continuum_random,
                local.maximum_continuum_random);
            merged.maximum_response_A_abs=std::max(
                merged.maximum_response_A_abs,
                local.maximum_response_A_abs);
            merged.maximum_response_B_abs=std::max(
                merged.maximum_response_B_abs,
                local.maximum_response_B_abs);
            merged.maximum_response_B_rel=std::max(
                merged.maximum_response_B_rel,
                local.maximum_response_B_rel);
            merged.maximum_response_fixed_abs=std::max(
                merged.maximum_response_fixed_abs,
                local.maximum_response_fixed_abs);
            merged.maximum_response_fixed_rel=std::max(
                merged.maximum_response_fixed_rel,
                local.maximum_response_fixed_rel);
        }
        const auto& accumulated=merged.accumulated;
        const auto& accumulated_response=merged.accumulated_response;
        const double maximum_A_role_abs=merged.maximum_A_role_abs;
        const double maximum_A_role_rel=merged.maximum_A_role_rel;
        const double maximum_B_role_abs=merged.maximum_B_role_abs;
        const double maximum_B_role_rel=merged.maximum_B_role_rel;
        const double maximum_fixed_abs=merged.maximum_fixed_abs;
        const double maximum_fixed_rel=merged.maximum_fixed_rel;
        const double maximum_continuum_random=
            merged.maximum_continuum_random;
        const double maximum_response_A_abs=
            merged.maximum_response_A_abs;
        const double maximum_response_B_abs=
            merged.maximum_response_B_abs;
        const double maximum_response_B_rel=
            merged.maximum_response_B_rel;
        const double maximum_response_fixed_abs=
            merged.maximum_response_fixed_abs;
        const double maximum_response_fixed_rel=
            merged.maximum_response_fixed_rel;

        require_close(
            eft::signed_dr_role_sum(accumulated.Ashot_residual),
            accumulated.Ashot_residual[eft::SignedDrRole::TiedF],
            "shell Ashot signed role sum");
        require_close(
            eft::signed_dr_role_sum(accumulated.Bshot_residual),
            accumulated.Bshot_residual[eft::SignedDrRole::TiedF],
            "shell Bshot signed role sum");
        require_close(
            eft::signed_dr_role_sum(
                accumulated_response.Ashot_residual),
            accumulated_response.Ashot_residual[
                eft::SignedDrRole::TiedF],
            "shell Ashot PNG response role sum");
        require_close(
            eft::signed_dr_role_sum(
                accumulated_response.Bshot_residual),
            accumulated_response.Bshot_residual[
                eft::SignedDrRole::TiedF],
            "shell Bshot PNG response role sum");
        const auto effective=eft::fixed_poisson_in_halo_density_units(
            accumulated.halo_poisson,
            accumulated.random_poisson,
            static_cast<double>(nran));
        std::array<eft::SignedDrRoleTemplates,3> effective_raw;
        std::array<eft::SparsePolynomial,3> effective_filter_complement;
        std::array<eft::SignedDrRoleTemplates,3> response_effective_raw;
        std::array<eft::SparsePolynomial,3>
            response_effective_filter_complement;
        std::array<eft::SparsePolynomial,3> response_effective_tied;
        double maximum_effective_abs=0.0;
        double maximum_effective_rel=0.0;
        double inverse_random_power=1.0;
        for (std::size_t inverse=1;inverse<3;++inverse) {
            inverse_random_power/=static_cast<double>(nran);
            for (std::size_t role=0;role<kRoleCount;++role) {
                effective_raw[inverse].value[role]=
                    accumulated.halo_poisson
                        .raw_by_inverse_density[inverse].value[role]
                    +inverse_random_power
                        *accumulated.random_poisson
                            .raw_by_inverse_density[inverse].value[role];
            }
            effective_filter_complement[inverse]=
                accumulated.halo_poisson
                    .legacy_theory_filter_complement[inverse]
                +inverse_random_power
                    *accumulated.random_poisson
                        .legacy_theory_filter_complement[inverse];
            for (std::size_t role=0;role<kRoleCount;++role) {
                response_effective_raw[inverse].value[role]=
                    accumulated_response.halo_poisson
                        .raw_by_inverse_density[inverse].value[role]
                    +inverse_random_power
                        *accumulated_response.random_poisson
                            .raw_by_inverse_density[inverse].value[role];
            }
            response_effective_filter_complement[inverse]=
                accumulated_response.halo_poisson
                    .legacy_theory_filter_complement[inverse]
                +inverse_random_power
                    *accumulated_response.random_poisson
                        .legacy_theory_filter_complement[inverse];
            response_effective_tied[inverse]=
                accumulated_response.halo_poisson
                    .legacy_theory_retained[inverse]
                +inverse_random_power
                    *accumulated_response.random_poisson
                        .legacy_theory_retained[inverse];
            const Comparison effective_closure=compare(
                eft::signed_dr_role_sum(effective_raw[inverse])
                    +effective_filter_complement[inverse],
                effective[inverse]);
            maximum_effective_abs=std::max(
                maximum_effective_abs,effective_closure.absolute);
            maximum_effective_rel=std::max(
                maximum_effective_rel,
                effective_closure.absolute/std::max({
                    role_scale(effective_raw[inverse]),
                    polynomial_scale(
                        effective_filter_complement[inverse]),
                    polynomial_scale(effective[inverse]),
                    1.0e-300}));
            require_close(
                eft::signed_dr_role_sum(effective_raw[inverse])
                    +effective_filter_complement[inverse],
                effective[inverse],
                "effective finite-random raw plus legacy-filter closure");
        }

        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":\"marisa-b-post-r1-dr-leading-stochastic-poisson-v1\","
                 <<"\"tree_only\":true,\"loops\":false,"
                 <<"\"maximum_threads\":"<<maximum_threads<<','
                 <<"\"threads_used\":"<<threads_used<<','
                 <<"\"parallel_reduction\":{"
                   "\"partition\":\"contiguous_node_ranges\","
                   "\"local_accumulators\":true,"
                   "\"merge_order\":\"ascending_thread_id\"},"
                 <<"\"global_nuisance_parameters\":["
                   "\"Ashot_residual\",\"Bshot_residual\"],"
                 <<"\"nuisance_normalization\":{"
                   "\"Ashot_residual\":\"template times Ashot_residual/n_halo^2\","
                   "\"Bshot_residual\":\"template times Bshot_residual/n_halo\"},"
                 <<"\"number_density_contract\":{"
                   "\"driver_output\":\"coefficients of inverse_n_halo powers; no n_halo value applied\","
                   "\"assembler_source\":\"strict-N94 fiducial halo count mean divided by 1e9 (Mpc/h)^3\","
                   "\"nonzero_png_counts_forbidden\":true,"
                   "\"dn_halo_dfNL_included\":false,"
                   "\"diagnostic_relative_dn_over_n_dfNL\":2.53e-6},"
                 <<"\"linear_png_stochastic_response\":\"frozen legacy_accepted_pre single-alpha3 branch; one half strict degree-6 generating derivative\","
                 <<"\"role_amplitudes\":\"forbidden\","
                 <<"\"empirical_gain\":\"forbidden\","
                 <<"\"signed_roles\":[\"DDD\",\"DDR\",\"DRR\",\"RRR\"],"
                 <<"\"random_catalog\":{"
                 <<"\"measurement_convention\":\"finite_independent_poisson\","
                 <<"\"continuum_exported\":true,"
                 <<"\"Nran\":"<<nran<<','
                 <<"\"n_random_over_n_halo\":"<<nran<<','
                 <<"\"halo_random_cross_cumulants\":\"zero\"},"
                 <<"\"fixed_poisson\":{"
                 <<"\"is_nuisance\":false,"
                 <<"\"frozen_fully_tree_active\":false,"
                 <<"\"policy\":\"diagnostic_not_active_in_frozen_gaussian_or_linear_model\","
                 <<"\"measurement_shot_mapping\":\"diagnostic_model_not_measurement_closed\","
                 <<"\"measurement_validation\":\"failed_low_order_N94_bin1_sign_and_role_mean_diagnostic\","
                 <<"\"halo_intensity\":\"UnitK0 leading tree; ConditionalTracer is identical at this order\","
                 <<"\"random_intensity\":\"independent uniform UnitK0\","
                 <<"\"raw_role_shot_policy\":\"zero shot per ordered D/R role\","
                 <<"\"legacy_filter_identity\":\"raw signed roles plus legacy_theory_filter_complement equals legacy_theory_retained\","
                 <<"\"legacy_filter_complement_is_measurement_fifth_channel\":false,"
                 <<"\"safe_consumption\":\"do not add fixed-Poisson raw, filter complement, or PNG direction to frozen model arrays\"},"
                 <<"\"source_hashes\":{\"linear_power\":";
        write_json_string(file_sha256(power_path));
        std::cout<<",\"png_table\":";
        write_json_string(file_sha256(png_path));
        std::cout<<",\"edges\":";
        write_json_string(file_sha256(edge_path));
        std::cout<<",\"executable\":";
        write_json_string(file_sha256(argv[0]));
        std::cout<<"},\"shell_projection\":{"
                 <<"\"radial_order\":"<<shell_config.exact.radial_order<<','
                 <<"\"angular_order\":"<<shell_config.exact.angular_order<<','
                 <<"\"orientation_orders\":["
                 <<shell_config.n_alpha<<','
                 <<shell_config.n_cos_beta<<','
                 <<shell_config.n_gamma<<"]},"
                 <<"\"reconstruction\":{\"R\":"
                 <<reconstruction.smoothing_radius
                 <<",\"b_rec\":"<<reconstruction.bias_recon
                 <<",\"cell_size\":"<<reconstruction.cell_size
                 <<",\"cic_power\":"<<cic_power<<"},"
                 <<"\"response_policy\":{\"p_bphi\":"<<p_bphi
                 <<",\"p_bphidelta\":1,"
                 <<"\"sector\":\"ExternalBphi K1 product rule\"},"
                 <<"\"parameter_registry\":";
        write_json_string(eft::registry_sha256());
        std::cout<<"}\n";

        std::cout<<"{\"record\":\"bin\",\"index\":"<<flat
                 <<",\"radial_pair\":["<<pair.first<<','<<pair.second<<']'
                 <<",\"edges\":["<<bin.k1_lower<<','<<bin.k1_upper
                 <<','<<bin.k2_lower<<','<<bin.k2_upper<<']'
                 <<",\"shell_nodes\":"<<rule.nodes.size()
                 <<",\"nuisance_tangents\":{"
                 <<"\"Ashot_residual\":";
        write_roles(accumulated.Ashot_residual);
        std::cout<<",\"Bshot_residual\":";
        write_roles(accumulated.Bshot_residual);
        std::cout<<"},\"linear_png_response_tangents\":{"
                 <<"\"Ashot_residual\":";
        write_roles(accumulated_response.Ashot_residual);
        std::cout<<",\"Bshot_residual\":";
        write_roles(accumulated_response.Bshot_residual);
        const eft::DrFixedPoissonSpeciesTemplates empty_random;
        std::cout<<"},\"fixed_poisson\":{"
                 <<"\"continuum_random\":{\"halo_species\":";
        write_fixed_species(accumulated.halo_poisson);
        std::cout<<",\"random_species\":";
        write_fixed_species(empty_random);
        std::cout<<",\"effective_raw_roles_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_roles(
                accumulated.halo_poisson
                    .raw_by_inverse_density[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_filter_complement_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                accumulated.halo_poisson
                    .legacy_theory_filter_complement[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_retained_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                accumulated.halo_poisson
                    .legacy_theory_retained[inverse]);
        }
        std::cout<<"},\"effective_frozen_legacy_exclusion_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                -1.0*accumulated.halo_poisson
                    .legacy_theory_retained[inverse]);
        }
        std::cout<<"},\"effective_frozen_active_tied_in_halo_density_units\":{"
                 <<"\"inverse_n_halo_1\":{},\"inverse_n_halo_2\":{}";
        std::cout<<"}},\"finite_random_Nran"<<nran
                 <<"\":{\"halo_species\":";
        write_fixed_species(accumulated.halo_poisson);
        std::cout<<",\"random_species\":";
        write_fixed_species(accumulated.random_poisson);
        std::cout<<",\"effective_raw_roles_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_roles(effective_raw[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_filter_complement_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(effective_filter_complement[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_retained_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(effective[inverse]);
        }
        std::cout<<"},\"effective_frozen_legacy_exclusion_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(-1.0*effective[inverse]);
        }
        std::cout<<"},\"effective_frozen_active_tied_in_halo_density_units\":{"
                 <<"\"inverse_n_halo_1\":{},\"inverse_n_halo_2\":{}";
        std::cout<<"}}},\"fixed_poisson_linear_png_diagnostic\":{"
                 <<"\"continuum_random\":{\"halo_species\":";
        write_fixed_species(accumulated_response.halo_poisson);
        std::cout<<",\"random_species\":";
        write_fixed_species(empty_random);
        std::cout<<",\"effective_raw_roles_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_roles(
                accumulated_response.halo_poisson
                    .raw_by_inverse_density[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_filter_complement_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                accumulated_response.halo_poisson
                    .legacy_theory_filter_complement[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_retained_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                accumulated_response.halo_poisson
                    .legacy_theory_retained[inverse]);
        }
        std::cout<<"},\"effective_frozen_legacy_exclusion_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                -1.0*accumulated_response.halo_poisson
                    .legacy_theory_retained[inverse]);
        }
        std::cout<<"},\"effective_frozen_active_tied_in_halo_density_units\":{"
                 <<"\"inverse_n_halo_1\":{},\"inverse_n_halo_2\":{}";
        std::cout<<"}},\"finite_random_Nran"<<nran
                 <<"\":{\"halo_species\":";
        write_fixed_species(accumulated_response.halo_poisson);
        std::cout<<",\"random_species\":";
        write_fixed_species(accumulated_response.random_poisson);
        std::cout<<",\"effective_raw_roles_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_roles(response_effective_raw[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_filter_complement_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(
                response_effective_filter_complement[inverse]);
        }
        std::cout<<"},\"effective_legacy_theory_retained_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(response_effective_tied[inverse]);
        }
        std::cout<<"},\"effective_frozen_legacy_exclusion_in_halo_density_units\":{";
        for (std::size_t inverse=1;inverse<3;++inverse) {
            if (inverse>1) std::cout<<',';
            write_json_string("inverse_n_halo_"+std::to_string(inverse));
            std::cout<<':';
            write_polynomial(-1.0*response_effective_tied[inverse]);
        }
        std::cout<<"},\"effective_frozen_active_tied_in_halo_density_units\":{"
                 <<"\"inverse_n_halo_1\":{},\"inverse_n_halo_2\":{}";
        std::cout<<"}}},\"closures\":{"
                 <<"\"Ashot_role_sum_max_abs\":"<<maximum_A_role_abs<<','
                 <<"\"Ashot_role_sum_max_rel\":"<<maximum_A_role_rel<<','
                 <<"\"Bshot_role_sum_max_abs\":"<<maximum_B_role_abs<<','
                 <<"\"Bshot_role_sum_max_rel\":"<<maximum_B_role_rel<<','
                 <<"\"Ashot_linear_png_max_abs\":"
                 <<maximum_response_A_abs<<','
                 <<"\"Bshot_linear_png_role_sum_max_abs\":"
                 <<maximum_response_B_abs<<','
                 <<"\"Bshot_linear_png_role_sum_max_rel\":"
                 <<maximum_response_B_rel<<','
                 <<"\"fixed_linear_png_raw_plus_legacy_filter_max_abs\":"
                 <<maximum_response_fixed_abs<<','
                 <<"\"fixed_linear_png_raw_plus_legacy_filter_max_rel\":"
                 <<maximum_response_fixed_rel<<','
                 <<"\"fixed_raw_plus_legacy_filter_max_abs\":"<<maximum_fixed_abs<<','
                 <<"\"fixed_raw_plus_legacy_filter_max_rel\":"<<maximum_fixed_rel<<','
                 <<"\"effective_finite_raw_plus_legacy_filter_max_abs\":"
                 <<maximum_effective_abs<<','
                 <<"\"effective_finite_raw_plus_legacy_filter_max_rel\":"
                 <<maximum_effective_rel<<','
                 <<"\"continuum_random_max_abs\":"<<maximum_continuum_random
                 <<"}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"D/R stochastic role driver failed: "
                 <<error.what()<<'\n';
        return 1;
    }
}
