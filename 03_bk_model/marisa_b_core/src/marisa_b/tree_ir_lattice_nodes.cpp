// 模块大纲：复用原有 exact-lattice measure；输出逐三角形节点；原分母与零模约定不变。
#include "lattice_shell_rule.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if (argc!=3) throw std::invalid_argument("usage: tree_ir_lattice_nodes nrad nmu < bins");
        marisa_b_eft_v2::ExactLatticeShellRuleConfig c;
        c.radial_order=std::stoi(argv[1]);c.angular_order=std::stoi(argv[2]);
        std::cout<<std::setprecision(17);
        int id;
        marisa_b_shell_v1::ShellBin bin;
        while (std::cin>>id>>bin.k1_lower>>bin.k1_upper>>bin.k2_lower>>bin.k2_upper) {
            const auto rule=marisa_b_eft_v2::make_exact_lattice_shell_rule(bin,c);
            for (const auto& n:rule.nodes) {
                std::cout<<id<<' '<<n.weight;
                for (const auto& v:n.closed_vectors) std::cout<<' '<<v.x<<' '<<v.y<<' '<<v.z;
                std::cout<<'\n';
            }
            std::cerr<<"bin "<<id<<" nodes "<<rule.nodes.size()<<" weight "
                     <<rule.diagnostics.total_weight<<" expected "
                     <<rule.diagnostics.valid_pair_fraction<<'\n';
        }
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
