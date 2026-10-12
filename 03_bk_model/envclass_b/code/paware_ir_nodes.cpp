// 模块大纲：原 FFT-lattice marginal + 明确 Haar 方向；不重归一化排除零模后的权重。
#include "lattice_shell_rule.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=6) throw std::invalid_argument("nrad nmu nalpha nbeta ngamma required");
        marisa_b_eft_v2::HaarOrientedLatticeShellRuleConfig c;
        c.exact.radial_order=std::stoi(argv[1]);c.exact.angular_order=std::stoi(argv[2]);
        c.n_alpha=std::stoi(argv[3]);c.n_cos_beta=std::stoi(argv[4]);c.n_gamma=std::stoi(argv[5]);
        int index;marisa_b_shell_v1::ShellBin bin;
        std::cout<<std::setprecision(17);
        while(std::cin>>index>>bin.k1_lower>>bin.k1_upper>>bin.k2_lower>>bin.k2_upper) {
            const auto rule=marisa_b_eft_v2::make_haar_oriented_lattice_shell_rule(bin,c);
            for(const auto& n:rule.nodes) {
                std::cout<<n.weight;
                for(const auto& leg:n.closed_vectors)
                    std::cout<<' '<<leg.x<<' '<<leg.y<<' '<<leg.z;
                std::cout<<'\n';
            }
            std::cerr<<index<<" nodes="<<rule.nodes.size()<<" weight="<<rule.diagnostics.total_weight<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
