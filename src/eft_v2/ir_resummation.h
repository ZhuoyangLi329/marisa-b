#ifndef MARISA_B_EFT_V2_IR_RESUMMATION_H
#define MARISA_B_EFT_V2_IR_RESUMMATION_H

#include <limits>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "direct_evaluator.h"

namespace marisa_b_eft_v2 {

enum class SigmaPowerPrescription {
    FullP11,
    NoWiggle,
};

enum class SmoothSplitPrescription {
    GaussianEisensteinHuRatio,
    GaussianLogK3Power,
};

struct IrResummationConfig {
    double kmin=1.0e-5;
    double kmax=100.0;
    int grid_size=2048;
    double smoothing_lambda=0.25;
    double smoothing_truncate=4.0;
    SmoothSplitPrescription split=SmoothSplitPrescription::GaussianEisensteinHuRatio;
    double hubble=0.6711;
    double scalar_tilt=0.9624;
    double omega_m=0.3175;
    double omega_b=0.049;
    double cmb_temperature=2.7255;
    double lambda_ir=0.10;
    double sound_horizon=110.0;
    SigmaPowerPrescription sigma_power=SigmaPowerPrescription::FullP11;
    double sigma2_override=std::numeric_limits<double>::quiet_NaN();
};

double eisenstein_hu_nowiggle_shape(double k,const IrResummationConfig& config);

class IrResummation {
public:
    class PowerView final:public PowerSpectrum {
    public:
        enum class Kind { NoWiggle, Loop, TreeNlo };
        real Evaluate(real k) const override;
        const Cosmology& GetCosmology() const override;
    private:
        friend class IrResummation;
        PowerView(const IrResummation& owner,Kind kind):owner_(owner),kind_(kind) {}
        const IrResummation& owner_;
        Kind kind_;
    };

    IrResummation(const PowerSpectrum& linear_power,IrResummationConfig config);

    const PowerView& no_wiggle_power() const noexcept { return no_wiggle_; }
    const PowerView& loop_power() const noexcept { return loop_; }
    const PowerView& tree_power() const noexcept { return tree_; }
    const IrResummationConfig& config() const noexcept { return config_; }
    double sigma2() const noexcept { return sigma2_; }
    std::string metadata_json() const;

private:
    double evaluate(typename PowerView::Kind kind,double k) const;
    double interpolate_log_nw(double k) const;
    double compute_sigma2() const;

    const PowerSpectrum& linear_power_;
    IrResummationConfig config_;
    std::vector<double> log_k_;
    std::vector<double> log_p_nw_;
    double sigma2_=0.0;
    PowerView no_wiggle_;
    PowerView loop_;
    PowerView tree_;
};

DiagramTemplates evaluate_ir_resummed(
    const IrResummation& resummation,
    const std::array<Vec3,3>& closed_triangle,
    const FieldKernelProvider& provider,
    const DirectIntegrationConfig& integration);

}  // namespace marisa_b_eft_v2

#endif
