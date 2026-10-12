/*
 * Additive fully-tree D/R estimator-role canary.
 *
 * This driver never calls a loop integrator and never changes the historical
 * tied production path.  It expands F=D-R before the tree bispectrum is
 * assembled, preserves all three external field labels, and propagates the
 * local-PNG halo K1/K2 direction analytically.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "diagram_assembler.h"
#include "field_kernel_provider.h"
#include "lattice_shell_rule.h"
#include "parameter_registry.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace shell=marisa_b_shell_v1;

namespace {

constexpr std::size_t kRoleCount=5;
constexpr std::size_t kResponseComponentCount=4;

using ResponseComponent=eft::LocalPngTreeDirectionSector;

const std::array<const char*,kRoleCount> kRoleNames{{
    "DDD","DDR","DRR","RRR","tied_F"
}};

const std::array<ResponseComponent,kResponseComponentCount>
kResponseComponents{{
    ResponseComponent::ExternalBphi,
    ResponseComponent::LocalIc,
    ResponseComponent::BphiAdvection,
    ResponseComponent::Bphidelta
}};

const char* component_name(ResponseComponent component) {
    switch (component) {
        case ResponseComponent::ExternalBphi:
            return "external_bphi_product_rule";
        case ResponseComponent::LocalIc:
            return "local_ic_primordial";
        case ResponseComponent::BphiAdvection:
            return "bphi_advection";
        case ResponseComponent::Bphidelta:
            return "bphidelta";
        case ResponseComponent::Total:
            return "total";
    }
    throw std::logic_error("unknown response component");
}

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
    LogTable(const std::string& path,int column,int required_columns)
        :column_(column),required_columns_(required_columns) {
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
                throw std::runtime_error("invalid positive table row");
            }
            log_k_.push_back(std::log(values[0]));
            log_value_.push_back(std::log(
                values[static_cast<std::size_t>(column_)]));
        }
        if (log_k_.size()<2) throw std::runtime_error("table is too short");
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
        throw std::runtime_error("D/R role driver requires 16 radial edges");
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

struct PolynomialComparison {
    double maximum_absolute=0.0;
    double maximum_scale=0.0;

    double relative() const {
        return maximum_absolute/std::max(maximum_scale,1.0e-300);
    }
};

PolynomialComparison compare(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected) {
    PolynomialComparison result;
    const eft::SparsePolynomial difference=actual-expected;
    for (const auto& term:difference.terms()) {
        result.maximum_absolute=std::max(
            result.maximum_absolute,std::fabs(term.second));
    }
    for (const auto& term:actual.terms()) {
        result.maximum_scale=std::max(
            result.maximum_scale,std::fabs(term.second));
    }
    for (const auto& term:expected.terms()) {
        result.maximum_scale=std::max(
            result.maximum_scale,std::fabs(term.second));
    }
    return result;
}

void require_close(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    const std::string& label) {
    const PolynomialComparison result=compare(actual,expected);
    if (result.maximum_absolute>2.0e-9
        &&result.relative()>2.0e-12) {
        throw std::runtime_error(
            label+" polynomial closure failed: abs="
            +std::to_string(result.maximum_absolute)
            +", rel="+std::to_string(result.relative()));
    }
}

std::array<eft::FieldKernelVariation,kRoleCount> pack_roles(
    const eft::SignedDrTreeVariations& roles,
    const eft::FieldKernelVariation& tied) {
    return {{roles.DDD,roles.DDR,roles.DRR,roles.RRR,tied}};
}

eft::FieldKernelVariationEvaluator evaluator(
    char field,
    const eft::FieldKernelProvider& base,
    const eft::FieldKernelProvider& direction,
    const hv1::ReconstructionConfig& reconstruction) {
    if (field=='D') {
        return [&](const std::vector<eft::Vec3>& momenta) {
            return eft::displaced_data_field_kernel_variation(
                base,direction,base,direction,
                reconstruction,momenta);
        };
    }
    if (field=='R') {
        return [&](const std::vector<eft::Vec3>& momenta) {
            return eft::shifted_random_field_kernel_variation(
                base,direction,reconstruction,momenta);
        };
    }
    if (field=='F') {
        return [&](const std::vector<eft::Vec3>& momenta) {
            return eft::reconstructed_field_kernel_variation_with_shift(
                base,direction,base,direction,
                reconstruction,momenta);
        };
    }
    throw std::invalid_argument("unknown D/R/F field selector");
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

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=16) {
            std::cerr
                <<"usage: marisa_b_post_r1_dr_tree_roles_driver "
                <<"POWER_TABLE PNG_TABLE EDGE_FILE FLAT_INDEX "
                <<"SHELL_NRAD SHELL_NMU N_ALPHA N_COS_BETA N_GAMMA "
                <<"R_SMOOTH B_REC CELL_SIZE CIC_POWER "
                <<"P_BPHI P_BPHIDELTA\n";
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
        const double p_bphi=parse_double(argv[14],"p bphi");
        const double p_bphidelta=parse_double(argv[15],"p bphidelta");
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
        const eft::LocalPngHaloTreeDirectionProvider total_direction(
            transfer,p_bphi,p_bphidelta,ResponseComponent::Total);
        std::array<
            eft::LocalPngHaloTreeDirectionProvider,
            kResponseComponentCount>
            component_directions{{
                {transfer,p_bphi,p_bphidelta,
                 ResponseComponent::ExternalBphi},
                {transfer,p_bphi,p_bphidelta,
                 ResponseComponent::LocalIc},
                {transfer,p_bphi,p_bphidelta,
                 ResponseComponent::BphiAdvection},
                {transfer,p_bphi,p_bphidelta,
                 ResponseComponent::Bphidelta}
            }};

        std::array<eft::SparsePolynomial,kRoleCount> gaussian;
        std::array<eft::SparsePolynomial,kRoleCount> linear;
        std::array<
            std::array<eft::SparsePolynomial,kRoleCount>,
            kResponseComponentCount> linear_components;
        double maximum_field_gaussian_closure=0.0;
        double maximum_field_response_closure=0.0;

        for (const shell::ShellNode& node:rule.nodes) {
            const auto D=evaluator(
                'D',base,total_direction,reconstruction);
            const auto R=evaluator(
                'R',base,total_direction,reconstruction);
            const auto F=evaluator(
                'F',base,total_direction,reconstruction);
            const eft::SignedDrTreeVariations roles=
                eft::signed_dr_tree_variations(
                    power,node.closed_vectors,D,R);
            const eft::FieldKernelVariation tied=
                eft::ordered_tree_bispectrum_variation(
                    power,node.closed_vectors,
                    std::array<eft::FieldKernelVariationEvaluator,3>{F,F,F});
            require_close(roles.total.value,tied.value,
                          "node Gaussian role sum");
            require_close(roles.total.direction,tied.direction,
                          "node PNG role sum");
            const auto packed=pack_roles(roles,tied);
            for (std::size_t role=0;role<kRoleCount;++role) {
                gaussian[role]+=node.weight*packed[role].value;
                linear[role]+=node.weight*packed[role].direction;
            }

            const std::array<std::vector<eft::Vec3>,5> kernel_probes{{
                {node.closed_vectors[0]},
                {node.closed_vectors[1]},
                {node.closed_vectors[2]},
                {node.closed_vectors[0],node.closed_vectors[1]},
                {node.closed_vectors[1],node.closed_vectors[2]}
            }};
            for (const auto& momenta:kernel_probes) {
                const eft::FieldKernelVariation d=D(momenta);
                const eft::FieldKernelVariation r=R(momenta);
                const eft::FieldKernelVariation f=F(momenta);
                maximum_field_gaussian_closure=std::max(
                    maximum_field_gaussian_closure,
                    compare(d.value-r.value,f.value).maximum_absolute);
                maximum_field_response_closure=std::max(
                    maximum_field_response_closure,
                    compare(d.direction-r.direction,f.direction)
                        .maximum_absolute);
            }

            for (std::size_t component=0;
                 component<kResponseComponentCount;++component) {
                const auto component_D=evaluator(
                    'D',base,component_directions[component],reconstruction);
                const auto component_R=evaluator(
                    'R',base,component_directions[component],reconstruction);
                const auto component_F=evaluator(
                    'F',base,component_directions[component],reconstruction);
                const eft::SignedDrTreeVariations component_roles=
                    eft::signed_dr_tree_variations(
                        power,node.closed_vectors,
                        component_D,component_R);
                const eft::FieldKernelVariation component_tied=
                    eft::ordered_tree_bispectrum_variation(
                        power,node.closed_vectors,
                        std::array<eft::FieldKernelVariationEvaluator,3>{
                            component_F,component_F,component_F});
                require_close(
                    component_roles.total.direction,
                    component_tied.direction,
                    std::string("node ")
                        +component_name(kResponseComponents[component])
                        +" role sum");
                const auto component_packed=
                    pack_roles(component_roles,component_tied);
                for (std::size_t role=0;role<kRoleCount;++role) {
                    linear_components[component][role]+=
                        node.weight*component_packed[role].direction;
                }
            }
        }

        const eft::SparsePolynomial gaussian_role_sum=
            gaussian[0]+gaussian[1]+gaussian[2]+gaussian[3];
        const eft::SparsePolynomial linear_role_sum=
            linear[0]+linear[1]+linear[2]+linear[3];
        require_close(gaussian_role_sum,gaussian[4],
                      "shell Gaussian role sum");
        require_close(linear_role_sum,linear[4],
                      "shell PNG role sum");
        double maximum_component_sum_absolute=0.0;
        double maximum_component_sum_relative=0.0;
        for (std::size_t role=0;role<kRoleCount;++role) {
            eft::SparsePolynomial component_sum;
            for (std::size_t component=0;
                 component<kResponseComponentCount;++component) {
                component_sum+=linear_components[component][role];
            }
            require_close(component_sum,linear[role],
                          std::string("shell PNG component sum ")
                              +kRoleNames[role]);
            const PolynomialComparison comparison=
                compare(component_sum,linear[role]);
            maximum_component_sum_absolute=std::max(
                maximum_component_sum_absolute,
                comparison.maximum_absolute);
            maximum_component_sum_relative=std::max(
                maximum_component_sum_relative,
                comparison.relative());
        }

        const PolynomialComparison gaussian_closure=
            compare(gaussian_role_sum,gaussian[4]);
        const PolynomialComparison linear_closure=
            compare(linear_role_sum,linear[4]);
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":\"marisa-b-post-r1-dr-fully-tree-roles-v1\","
                 <<"\"tree_only\":true,"
                 <<"\"loops\":false,"
                 <<"\"stochastic_dr_contraction\":"
                   "\"not_implemented_next_gate\","
                 <<"\"source_hashes\":{\"linear_power\":";
        write_json_string(file_sha256(power_path));
        std::cout<<",\"png_table\":";
        write_json_string(file_sha256(png_path));
        std::cout<<",\"edges\":";
        write_json_string(file_sha256(edge_path));
        std::cout<<",\"executable\":";
        write_json_string(file_sha256(argv[0]));
        std::cout<<"},\"shell_projection\":{"
                 <<"\"radial_order\":"
                 <<shell_config.exact.radial_order<<','
                 <<"\"angular_order\":"
                 <<shell_config.exact.angular_order<<','
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
                 <<",\"p_bphidelta\":"<<p_bphidelta
                 <<"},\"parameter_registry\":";
        write_json_string(eft::registry_sha256());
        std::cout<<"}\n";

        std::cout<<"{\"record\":\"bin\",\"index\":"<<flat
                 <<",\"radial_pair\":["<<pair.first<<','<<pair.second<<']'
                 <<",\"edges\":["<<bin.k1_lower<<','<<bin.k1_upper
                 <<','<<bin.k2_lower<<','<<bin.k2_upper<<']'
                 <<",\"shell_nodes\":"<<rule.nodes.size()
                 <<",\"roles\":{";
        for (std::size_t role=0;role<kRoleCount;++role) {
            if (role) std::cout<<',';
            write_json_string(kRoleNames[role]);
            std::cout<<":{\"gaussian_tree\":";
            write_polynomial(gaussian[role]);
            std::cout<<",\"linear_png\":";
            write_polynomial(linear[role]);
            std::cout<<",\"linear_components\":{";
            for (std::size_t component=0;
                 component<kResponseComponentCount;++component) {
                if (component) std::cout<<',';
                write_json_string(component_name(
                    kResponseComponents[component]));
                std::cout<<':';
                write_polynomial(
                    linear_components[component][role]);
            }
            std::cout<<"}}";
        }
        std::cout<<"},\"closures\":{"
                 <<"\"field_D_minus_R_equals_F_gaussian_max_abs\":"
                 <<maximum_field_gaussian_closure<<','
                 <<"\"field_D_minus_R_equals_F_linear_max_abs\":"
                 <<maximum_field_response_closure<<','
                 <<"\"signed_role_gaussian_max_abs\":"
                 <<gaussian_closure.maximum_absolute<<','
                 <<"\"signed_role_gaussian_relative\":"
                 <<gaussian_closure.relative()<<','
                 <<"\"signed_role_linear_max_abs\":"
                 <<linear_closure.maximum_absolute<<','
                 <<"\"signed_role_linear_relative\":"
                 <<linear_closure.relative()<<','
                 <<"\"linear_component_sum_max_abs\":"
                 <<maximum_component_sum_absolute<<','
                 <<"\"linear_component_sum_relative\":"
                 <<maximum_component_sum_relative
                 <<"}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"D/R fully-tree role driver failed: "
                 <<error.what()<<'\n';
        return 1;
    }
}
