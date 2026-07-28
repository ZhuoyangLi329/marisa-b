#include "fftlog_dr_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "halo_v1.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;

namespace {

int checks=0;

void require(bool condition,const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    double actual,double expected,double relative,double absolute,
    const std::string& message) {
    ++checks;
    const double error=std::fabs(actual-expected);
    if (error>absolute && error>relative*std::max(std::fabs(actual),std::fabs(expected))) {
        std::ostringstream details;
        details.precision(17);
        details<<message<<": actual="<<actual<<", expected="<<expected
               <<", relative_error="<<error/std::max(std::fabs(expected),1.0e-300);
        throw std::runtime_error(details.str());
    }
}

void require_polynomial_close(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    double relative,double absolute,
    const std::string& message) {
    for (const auto& term:actual.terms()) {
        require_close(
            term.second,expected.coefficient(term.first),
            relative,absolute,message+" "+term.first.canonical_string());
    }
    for (const auto& term:expected.terms()) {
        require_close(
            actual.coefficient(term.first),term.second,
            relative,absolute,message+" "+term.first.canonical_string());
    }
}

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table");
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0,p=0.0;
            if (row>>k>>p) {
                log_k_.push_back(std::log(k));
                log_p_.push_back(std::log(p));
            }
        }
        if (log_k_.size()<2) throw std::runtime_error("power table too short");
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) right=log_k_.size()-1;
        else if (x>log_k_.front()) {
            right=std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin();
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(log_p_[left]+fraction*(log_p_[right]-log_p_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no cosmology");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

class PowerLawSpectrum final:public PowerSpectrum {
public:
    PowerLawSpectrum(double amplitude,double exponent)
        : amplitude_(amplitude),exponent_(exponent) {}

    real Evaluate(real k) const override {
        return k>0.0?amplitude_*std::pow(k,exponent_):0.0;
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("PowerLawSpectrum has no cosmology");
    }

private:
    double amplitude_=1.0;
    double exponent_=0.0;
};

class UnitKernelProvider final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>&) const override {
        return eft::SparsePolynomial::constant(1.0);
    }

    std::string_view name() const noexcept override {
        return "unit_fftlog_dr_test_kernel";
    }
};

class SyntheticTadpoleKernelProvider final:public eft::FieldKernelProvider {
public:
    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        if (momenta.size()!=3) {
            return eft::SparsePolynomial::constant(1.0);
        }
        const eft::Vec3 shifted=hv1::add(momenta[0],momenta[1]);
        const double square=hv1::dot(shifted,shifted);
        if (!(square>0.0)) {
            throw std::domain_error(
                "synthetic tadpole kernel lies on its shifted pole");
        }
        return eft::SparsePolynomial::constant(1.0/square);
    }

    std::string_view name() const noexcept override {
        return "synthetic_shifted_tadpole_test_kernel";
    }
};

std::array<double,eft::kParameterCount> selected_bias_point() {
    std::array<double,eft::kParameterCount> values={};
    values[static_cast<std::size_t>(eft::ParameterId::B1)]=2.2;
    values[static_cast<std::size_t>(eft::ParameterId::B2)]=0.3;
    values[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.2;
    values[static_cast<std::size_t>(eft::ParameterId::B3)]=0.1;
    return values;
}

void test_fftlog_power_reconstruction(const TabulatedPower& source) {
    eft::FftlogConfig config;
    config.kmin=1.0e-6;
    config.kmax=1.0e3;
    config.frequency_count=512;
    config.reconstruction_grid_size=8192;
    const eft::FftlogPowerSpectrum fftlog(source,config);
    double max_relative=0.0;
    double squared_relative=0.0;
    constexpr int count=401;
    for (int index=0;index<count;++index) {
        const double fraction=(index+0.5)/static_cast<double>(count);
        const double k=std::exp(
            std::log(1.0e-4)+fraction*std::log(80.0/1.0e-4));
        const double expected=source(k);
        const double relative=std::fabs(fftlog(k)/expected-1.0);
        max_relative=std::max(max_relative,relative);
        squared_relative+=relative*relative;
    }
    const double rms_relative=std::sqrt(squared_relative/count);
    require(max_relative<1.0e-3,
            "FFTLog power maximum reconstruction error="+std::to_string(max_relative));
    require(rms_relative<2.0e-4,
            "FFTLog power RMS reconstruction error="+std::to_string(rms_relative));
    const std::string metadata=fftlog.metadata_json();
    require(metadata.find("\"schema\":\"marisa-b-fftlog-power-v1\"")!=std::string::npos,
            "FFTLog power metadata schema");
    require(metadata.find("\"frequency_count\":512")!=std::string::npos,
            "FFTLog power metadata frequency count");
    require(metadata.find("\"bias_nu\":")!=std::string::npos,
            "FFTLog power metadata bias");
    require(metadata.find("\"mode_convention\":")!=std::string::npos,
            "FFTLog power metadata mode convention");
    const auto modes=fftlog.modes();
    require(modes.size()==513,"FFTLog public mode count including split Nyquist");
    double maximum_mode_relative=0.0;
    double maximum_mode_imaginary_relative=0.0;
    double worst_mode_condition=0.0;
    double worst_mode_expected=0.0;
    double maximum_mode_rounding_units=0.0;
    int worst_mode_sample=-1;
    for (int index=0;index<41;++index) {
        const int sample=(37*index)%config.frequency_count;
        const double k=std::exp(
            std::log(config.kmin)
            +sample*std::log(config.kmax/config.kmin)
                /config.frequency_count);
        std::complex<long double> reconstructed(0.0L,0.0L);
        long double absolute_term_sum=0.0L;
        for (const auto& mode:modes) {
            const std::complex<long double> term=std::complex<long double>(
                mode.coefficient.real(),mode.coefficient.imag())
                *std::exp(std::complex<long double>(
                    mode.exponent.real(),mode.exponent.imag())
                    *static_cast<long double>(std::log(k/mode.pivot_k)));
            reconstructed+=term;
            absolute_term_sum+=std::abs(term);
        }
        const double expected=source(k);
        const double relative=
            std::fabs(static_cast<double>(reconstructed.real())/expected-1.0);
        const double condition=static_cast<double>(absolute_term_sum/expected);
        maximum_mode_rounding_units=std::max(
            maximum_mode_rounding_units,
            relative/(config.frequency_count*std::numeric_limits<double>::epsilon()
                      *condition));
        if (relative>maximum_mode_relative) {
            maximum_mode_relative=relative;
            worst_mode_condition=condition;
            worst_mode_expected=expected;
            worst_mode_sample=sample;
        }
        maximum_mode_imaginary_relative=std::max(
            maximum_mode_imaginary_relative,
            std::fabs(static_cast<double>(reconstructed.imag()))/expected);
    }
    std::ostringstream imaginary_details;
    imaginary_details.precision(17);
    imaginary_details
        <<"FFTLog public modes reconstruct a real spectrum, maximum relative imaginary="
        <<maximum_mode_imaginary_relative;
    require(maximum_mode_imaginary_relative<2.0e-10,imaginary_details.str());
    std::ostringstream mode_details;
    mode_details.precision(17);
    mode_details<<"FFTLog public mode phase convention, maximum relative="
                <<maximum_mode_relative<<", sample="<<worst_mode_sample
                <<", expected="<<worst_mode_expected
                <<", condition="<<worst_mode_condition
                <<", maximum N*epsilon*condition units="
                <<maximum_mode_rounding_units;
    require(maximum_mode_relative<2.0e-6,mode_details.str());
    require(maximum_mode_rounding_units<4.0,
            "FFTLog public mode reconstruction exceeds its conditioning-scaled rounding bound");

    eft::FftlogConfig analytic_only;
    analytic_only.kmin=1.0e-5;
    analytic_only.kmax=10.0;
    analytic_only.frequency_count=16;
    analytic_only.reconstruction_grid_size=128;
    analytic_only.bias_nu=-0.6;
    analytic_only.reconstruct_interpolated_power=false;
    analytic_only.endpoint_inclusive_sampling=true;
    const eft::FftlogPowerSpectrum modes_only(source,analytic_only);
    const std::string modes_only_metadata=modes_only.metadata_json();
    require(
        modes_only_metadata.find(
            "\"interpolated_power_available\":false")
            !=std::string::npos,
        "analytic-only FFTLog metadata disables interpolated power");
    require(
        modes_only_metadata.find(
            "\"endpoint_inclusive_sampling\":true")
            !=std::string::npos,
        "analytic-only FFTLog metadata records public endpoint convention");
    bool interpolation_threw=false;
    try {
        static_cast<void>(modes_only(0.1));
    } catch (const std::logic_error&) {
        interpolation_threw=true;
    }
    require(
        interpolation_threw,
        "analytic-only FFTLog rejects accidental interpolated-power use");
    const auto analytic_modes=modes_only.modes();
    require(
        analytic_modes.size()==17,
        "analytic-only FFTLog exposes N+1 modes after split Nyquist");
    for (int sample=0;sample<analytic_only.frequency_count;++sample) {
        const double k=std::exp(
            std::log(analytic_only.kmin)
            +sample*std::log(analytic_only.kmax/analytic_only.kmin)
                /(analytic_only.frequency_count-1));
        std::complex<long double> reconstructed(0.0L,0.0L);
        long double absolute_term_sum=0.0L;
        for (const auto& mode:analytic_modes) {
            const std::complex<long double> term=
                std::complex<long double>(
                    mode.coefficient.real(),mode.coefficient.imag())
                *std::exp(
                    std::complex<long double>(
                        mode.exponent.real(),mode.exponent.imag())
                    *static_cast<long double>(
                        std::log(k/mode.pivot_k)));
            reconstructed+=term;
            absolute_term_sum+=std::abs(term);
        }
        const double expected=source(k);
        const double relative=std::fabs(
            static_cast<double>(reconstructed.real())/expected-1.0);
        const double condition=static_cast<double>(
            absolute_term_sum/std::max(
                static_cast<long double>(expected),1.0e-300L));
        require(
            relative
                <8.0*analytic_only.frequency_count
                    *std::numeric_limits<double>::epsilon()
                    *std::max(condition,1.0),
            "endpoint-inclusive analytic modes reconstruct sampled source");
    }
}

void test_public_table_operator_masks() {
    constexpr double exponent=-1.73;
    const PowerLawSpectrum source(1.4,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-5;
    fftlog.kmax=10.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    fftlog.reconstruct_interpolated_power=false;
    fftlog.endpoint_inclusive_sampling=true;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider provider;

    const auto full_b222=oracle.evaluate_b222_public_table_analytic(
        triangle,provider,{},1.0e-12,eft::kB222AllContourMask);
    const auto nonconstant_b222=
        oracle.evaluate_b222_public_table_analytic(
            triangle,provider,{},1.0e-12,
            eft::kB222AllContourMask
                &~eft::kB222ConstantContourMask);
    const auto constant_b222=
        oracle.evaluate_b222_public_table_analytic(
            triangle,provider,{},1.0e-12,
            eft::kB222ConstantContourMask);
    require_polynomial_close(
        nonconstant_b222.value+constant_b222.value,
        full_b222.value,3.0e-13,2.0e-13,
        "public B222 operator masks exactly partition the topology");
    require(
        constant_b222.value.terms().size()==1,
        "public B222 constant contour selects only b2 cubed");
    require(
        constant_b222.reduction.basis_terms==1,
        "public B222 b2 cubed uses one active frequency");

    const auto full_b321i=
        oracle.evaluate_b321i_public_table_analytic(
            triangle,provider,{},1.0e-12,1.0e-8,
            eft::kB321IAllContourMask);
    eft::SparsePolynomial partitioned_b321i;
    for (const std::uint64_t mask:{
            eft::kB321ISoftContourMask,
            eft::kB321ILocalQuadraticContourMask,
            eft::kB321ICompositeContourMask}) {
        partitioned_b321i+=
            oracle.evaluate_b321i_public_table_analytic(
                triangle,provider,{},1.0e-12,1.0e-8,mask).value;
    }
    require_polynomial_close(
        partitioned_b321i,full_b321i.value,4.0e-13,2.0e-13,
        "public B321I contour masks exactly partition the topology");
    const auto constant_b321i=
        oracle.evaluate_b321i_public_table_analytic(
            triangle,provider,{},1.0e-12,1.0e-8,
            eft::kB321ICompositeContourMask);
    require(
        constant_b321i.value.terms().size()==15,
        "public B321I composite contour selects fifteen bias monomials");
    require(
        (eft::kB321ISoftContourMask
         |eft::kB321ILocalQuadraticContourMask
         |eft::kB321ICompositeContourMask)
            ==eft::kB321IAllContourMask,
        "public B321I contour masks cover every operator");
    require(
        (eft::kB321ISoftContourMask
         &eft::kB321ILocalQuadraticContourMask)==0
        &&(eft::kB321ISoftContourMask
           &eft::kB321ICompositeContourMask)==0
        &&(eft::kB321ILocalQuadraticContourMask
           &eft::kB321ICompositeContourMask)==0,
        "public B321I contour masks are pairwise disjoint");

    bool b222_zero_mask_threw=false;
    try {
        static_cast<void>(
            oracle.evaluate_b222_public_table_analytic(
                triangle,provider,{},1.0e-12,0));
    } catch (const std::invalid_argument&) {
        b222_zero_mask_threw=true;
    }
    require(
        b222_zero_mask_threw,
        "public B222 rejects an empty operator mask");
    bool b321i_zero_mask_threw=false;
    try {
        static_cast<void>(
            oracle.evaluate_b321i_public_table_analytic(
                triangle,provider,{},1.0e-12,1.0e-8,0));
    } catch (const std::invalid_argument&) {
        b321i_zero_mask_threw=true;
    }
    require(
        b321i_zero_mask_threw,
        "public B321I rejects an empty operator mask");
}

void test_analytic_three_propagator_master() {
    const auto two_point=eft::fftlog_dr_two_propagator_integral(
        {0.9,0.0},{0.85,0.0});
    require_close(two_point.real(),0.15576180657109628,
                  2.0e-13,2.0e-15,
                  "analytic FFTLog/DR I real reference");
    require_close(two_point.imag(),0.0,0.0,2.0e-14,
                  "analytic FFTLog/DR I real imaginary part");
    const auto two_point_zero=eft::fftlog_dr_two_propagator_integral(
        {0.0,0.0},{0.85,0.0});
    require_close(std::abs(two_point_zero),0.0,0.0,1.0e-15,
                  "analytic FFTLog/DR I scaleless zero");

    // Independent reference values were obtained from the Feynman-parameter
    // representation in Simonovic et al. eq. (3.31), using endpoint-free
    // 2D Gauss--Legendre rules and an N=128,256,512,768 convergence sequence.
    const auto real_value=eft::fftlog_dr_master_integral(
        {0.8,0.0},{0.75,0.0},{0.7,0.0},0.49,0.81);
    require_close(real_value.value.real(),0.10811084396108896,
                  2.0e-11,2.0e-13,"analytic FFTLog/DR J real reference");
    require_close(real_value.value.imag(),0.0,0.0,2.0e-13,
                  "analytic FFTLog/DR J real imaginary part");
    require(real_value.outer_terms<32,"analytic FFTLog/DR J outer convergence");
    require(real_value.relative_tail<1.0e-10,
            "analytic FFTLog/DR J reported tail");

    const std::complex<double> nu1(0.67,0.21);
    const std::complex<double> nu2(0.59,-0.13);
    const std::complex<double> nu3(0.62,0.08);
    const auto complex_value=eft::fftlog_dr_master_integral(
        nu1,nu2,nu3,0.36,0.72);
    // Independent complex Feynman-parameter quadrature at N=1024.  Its
    // N=768 -> 1024 change is 3.0e-9 (real) and 3.4e-10 (imaginary).
    require_close(complex_value.value.real(),0.11050665019585088,
                  2.0e-7,2.0e-9,"analytic FFTLog/DR J complex real reference");
    require_close(complex_value.value.imag(),-0.01663765472708497,
                  2.0e-7,2.0e-9,"analytic FFTLog/DR J complex imaginary reference");
    const auto conjugate_value=eft::fftlog_dr_master_integral(
        std::conj(nu1),std::conj(nu2),std::conj(nu3),0.36,0.72);
    require_close(conjugate_value.value.real(),complex_value.value.real(),
                  2.0e-11,2.0e-13,"analytic FFTLog/DR J conjugate real");
    require_close(conjugate_value.value.imag(),-complex_value.value.imag(),
                  2.0e-11,2.0e-13,"analytic FFTLog/DR J conjugate imaginary");
}

void test_b222_laurent_reduction() {
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider provider;
    eft::FftlogLaurentConfig config;
    config.minimum_power=-2;
    config.maximum_power=2;
    config.oversampling_factor=3;
    config.validation_samples=64;
    const auto reduction=eft::fftlog_dr_reduce_b222_kernel(
        triangle,provider,config);
    require(reduction.basis_terms==125,"B222 Laurent basis size");
    require(!reduction.terms.empty() && reduction.terms.size()<=125,
            "B222 Laurent retained-term count");
    require(reduction.numerical_rank==125,
            "B222 Laurent collocation full numerical rank");
    require(reduction.maximum_validation_relative_error<2.0e-8,
            "B222 Laurent independent holdout reconstruction, error="
            +std::to_string(reduction.maximum_validation_relative_error));
}

void test_b321i_route_laurent_reduction() {
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider provider;
    eft::FftlogLaurentConfig config;
    config.minimum_power=-3;
    config.maximum_power=3;
    config.oversampling_factor=3;
    config.validation_samples=16;
    config.singular_value_tolerance=1.0e-12;
    config.coefficient_prune_relative=1.0e-10;
    for (int shifted=0;shifted<3;++shifted) {
        for (int external=0;external<3;++external) {
            if (external==shifted) continue;
            const auto reduction=eft::fftlog_dr_reduce_b321i_route_kernel(
                triangle,provider,external,shifted,config);
            const std::string route=std::to_string(external)
                                   +"->"+std::to_string(shifted);
            require(reduction.basis_terms==490,
                    "B321I Laurent route "+route+" two-chart basis size");
            require(reduction.numerical_rank==490,
                    "B321I Laurent route "+route
                    +" full candidate rank="
                    +std::to_string(reduction.numerical_rank));
            require(reduction.refit_rank
                        ==static_cast<int>(reduction.terms.size()),
                    "B321I Laurent route "+route
                    +" active-support refit rank");
            std::ostringstream details;
            details.precision(17);
            details<<"B321I Laurent route "<<route
                   <<" holdout relative error="
                   <<reduction.maximum_validation_relative_error
                   <<", absolute error="
                   <<reduction.maximum_validation_absolute_error
                   <<", rank="<<reduction.numerical_rank
                   <<", condition="<<reduction.condition_number
                   <<", retained="<<reduction.terms.size()
                   <<", refit rank="<<reduction.refit_rank
                   <<", refit condition="
                   <<reduction.refit_condition_number;
            require(reduction.maximum_validation_relative_error<3.0e-8,
                    details.str());
        }
    }
}

void test_b321ii_route_laurent_reduction() {
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider provider;
    eft::FftlogLaurentConfig config;
    config.minimum_power=-2;
    config.maximum_power=2;
    config.oversampling_factor=3;
    config.validation_samples=16;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            for (const int tadpole:{left,right}) {
                const auto reduction=
                    eft::fftlog_dr_reduce_b321ii_route_kernel(
                        triangle,provider,left,right,tadpole,config);
                const std::string route=std::to_string(left)+","
                    +std::to_string(right)+";"
                    +std::to_string(tadpole);
                require(reduction.basis_terms==125,
                        "B321II Laurent route "+route+" basis size");
                require(!reduction.terms.empty(),
                        "B321II Laurent route "+route+" retained terms");
                std::ostringstream details;
                details.precision(17);
                details<<"B321II Laurent route "<<route
                       <<" holdout relative error="
                       <<reduction.maximum_validation_relative_error
                       <<", absolute error="
                       <<reduction.maximum_validation_absolute_error
                       <<", candidate rank="<<reduction.numerical_rank
                       <<", retained="<<reduction.terms.size()
                       <<", refit rank="<<reduction.refit_rank;
                require(
                    reduction.maximum_validation_relative_error<3.0e-8,
                    details.str());
            }
        }
    }
}

void test_analytic_b411_generated_table() {
    constexpr double amplitude=1.3;
    constexpr double exponent=-1.25;
    const PowerLawSpectrum source(amplitude,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-4;
    fftlog.kmax=100.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::MatterKernelProvider matter_provider;
    const auto matter=oracle.evaluate_b411_analytic(
        triangle,matter_provider,{}, {},1.0e-12,1.0e-8);
    const eft::EftBiasKernelProvider bias_provider;
    const auto bias=oracle.evaluate_b411_analytic(
        triangle,bias_provider,{}, {},1.0e-12,1.0e-8);
    require(matter.fftlog_modes_used==1,
            "analytic B411 pure power retains one FFTLog mode");
    require(matter.reduction.basis_terms==112,
            "analytic B411 generated frequency-table size");
    require(matter.master_integral_evaluations==336,
            "analytic B411 evaluates 112 frequencies for three routes");
    require(bias.value.terms().size()==10,
            "analytic B411 has ten nonzero real-space operator rows");
    std::array<double,eft::kParameterCount> matter_point={};
    matter_point[static_cast<std::size_t>(eft::ParameterId::B1)]=1.0;
    require_close(
        bias.value.evaluate(matter_point),matter.value.evaluate({}),
        2.0e-12,1.0e-12,
        "analytic B411 matter projection of generated bias table");
    static constexpr std::array<eft::ParameterId,11> operator_ids={{
        eft::ParameterId::B1,
        eft::ParameterId::B2,
        eft::ParameterId::B3,
        eft::ParameterId::Gamma2,
        eft::ParameterId::Gamma21,
        eft::ParameterId::Gamma211,
        eft::ParameterId::Gamma21x,
        eft::ParameterId::Gamma22,
        eft::ParameterId::Gamma2x,
        eft::ParameterId::Gamma3,
        eft::ParameterId::Gamma31}};
    // Independent 70-digit mpmath evaluation of the decoded upstream WDX
    // tables and Simonovic master series at epsilon=1e-8.
    static constexpr std::array<double,11> reference={{
        0.92635204650812882,
        0.25098255266096487,
        0.0,
        -23.780986910061000,
        111.07075068183171,
        -17.780489283607690,
        26.470708781755396,
        -36.121990384267365,
        -22.689178955790339,
        22.860629078924173,
        -4.4611023722334836}};
    for (std::size_t index=0;index<operator_ids.size();++index) {
        const eft::MonomialKey key=operator_ids[index]==eft::ParameterId::B1
            ?eft::MonomialKey({{
                eft::ParameterId::B1,static_cast<std::uint8_t>(3)}})
            :eft::MonomialKey({
                {eft::ParameterId::B1,static_cast<std::uint8_t>(2)},
                {operator_ids[index],static_cast<std::uint8_t>(1)}});
        require_close(
            bias.value.coefficient(key),reference[index],
            3.0e-7,2.0e-9,
            "analytic B411 generated-table high-precision operator "
            +std::to_string(index));
    }
    require(matter.maximum_imaginary_to_real<1.0e-13,
            "analytic B411 pure-power result is real");
}

void test_analytic_b222_power_law() {
    constexpr double amplitude=1.7;
    constexpr double exponent=-1.2;
    const PowerLawSpectrum source(amplitude,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-4;
    fftlog.kmax=100.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const UnitKernelProvider provider;
    eft::FftlogLaurentConfig laurent;
    laurent.validation_samples=32;
    const auto analytic=oracle.evaluate_b222_analytic(
        triangle,provider,laurent,{},1.0e-12);

    const double x=std::pow(canonical.sides.k1/canonical.sides.k3,2);
    const double y=std::pow(canonical.sides.k2/canonical.sides.k3,2);
    const std::complex<double> nu(-0.5*exponent,0.0);
    const auto master=eft::fftlog_dr_master_integral(nu,nu,nu,x,y);
    const double expected=8.0*std::pow(amplitude,3)
        *std::pow(canonical.sides.k3,3.0+3.0*exponent)
        *master.value.real();
    require(analytic.fftlog_modes_used==1,
            "analytic B222 pure power law retains one FFTLog mode");
    require(analytic.master_integral_evaluations
                ==analytic.reduction.terms.size(),
            "analytic B222 evaluates one master per retained Laurent term");
    require(analytic.reduction.terms.size()==1,
            "unit-kernel B222 Laurent reduction retains only its constant");
    require_close(
        analytic.value.evaluate({}),expected,2.0e-9,1.0e-11,
        "analytic B222 pure-power master mapping and scale");
    require(analytic.maximum_imaginary_to_real<1.0e-12,
            "analytic B222 pure-power imaginary cancellation");
}

void test_analytic_b321i_power_law() {
    constexpr double amplitude=1.3;
    constexpr double exponent=-1.8;
    const PowerLawSpectrum source(amplitude,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-4;
    fftlog.kmax=100.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const UnitKernelProvider provider;
    eft::FftlogLaurentConfig laurent;
    laurent.validation_samples=24;
    const auto analytic=oracle.evaluate_b321i_analytic(
        triangle,provider,laurent,{},1.0e-12);

    const std::array<double,3> sides={{
        canonical.sides.k1,canonical.sides.k2,canonical.sides.k3}};
    const std::complex<double> nu(-0.5*exponent,0.0);
    const double master=eft::fftlog_dr_two_propagator_integral(
        nu,nu).real();
    double expected=0.0;
    for (int shifted=0;shifted<3;++shifted) {
        for (int external=0;external<3;++external) {
            if (external==shifted) continue;
            expected+=6.0*std::pow(amplitude,3)
                *std::pow(sides[static_cast<std::size_t>(external)],exponent)
                *std::pow(
                    sides[static_cast<std::size_t>(shifted)],
                    3.0+2.0*exponent)
                *master;
        }
    }
    require(analytic.fftlog_modes_used==1,
            "analytic B321I pure power law retains one FFTLog mode");
    require(analytic.route_reductions.size()==6,
            "analytic B321I has six route reductions");
    require(analytic.master_integral_evaluations==6,
            "analytic B321I unit kernel uses six two-propagator masters");
    require_close(
        analytic.value.evaluate({}),expected,3.0e-8,1.0e-9,
        "analytic B321I pure-power route sum and scale");
    require(analytic.maximum_imaginary_to_real<1.0e-12,
            "analytic B321I pure-power imaginary cancellation");
}

void test_analytic_b321ii_power_law() {
    constexpr double amplitude=1.4;
    constexpr double exponent=-1.2;
    const PowerLawSpectrum source(amplitude,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-4;
    fftlog.kmax=100.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const SyntheticTadpoleKernelProvider provider;
    eft::FftlogLaurentConfig laurent;
    laurent.validation_samples=24;
    laurent.coefficient_prune_relative=1.0e-8;
    const auto analytic=oracle.evaluate_b321ii_analytic(
        triangle,provider,laurent,{},1.0e-12,1.0e-9);

    const std::array<double,3> sides={{
        canonical.sides.k1,canonical.sides.k2,canonical.sides.k3}};
    const double master=eft::fftlog_dr_two_propagator_integral(
        {-0.5*exponent,0.0},{1.0,0.0}).real();
    double expected=0.0;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            for (const int tadpole:{left,right}) {
                expected+=6.0*std::pow(amplitude,3)
                    *std::pow(sides[static_cast<std::size_t>(left)],exponent)
                    *std::pow(sides[static_cast<std::size_t>(right)],exponent)
                    *std::pow(
                        sides[static_cast<std::size_t>(tadpole)],
                        1.0+exponent)
                    *master;
            }
        }
    }
    require(analytic.fftlog_modes_used==1,
            "analytic B321II pure power law retains one FFTLog mode");
    require(analytic.route_reductions.size()==6,
            "analytic B321II has six route reductions");
    require(analytic.master_integral_evaluations>=6,
            "analytic B321II uses non-scaleless master integrals");
    double maximum_holdout=0.0;
    double maximum_refit_condition=0.0;
    std::size_t minimum_terms=std::numeric_limits<std::size_t>::max();
    std::size_t maximum_terms=0;
    for (const auto& reduction:analytic.route_reductions) {
        maximum_holdout=std::max(
            maximum_holdout,reduction.maximum_validation_relative_error);
        maximum_refit_condition=std::max(
            maximum_refit_condition,reduction.refit_condition_number);
        minimum_terms=std::min(minimum_terms,reduction.terms.size());
        maximum_terms=std::max(maximum_terms,reduction.terms.size());
    }
    const double actual=analytic.value.evaluate({});
    const double relative=std::fabs(actual/expected-1.0);
    std::ostringstream analytic_details;
    analytic_details.precision(17);
    analytic_details
        <<"analytic B321II pure-power route sum and scale: actual="<<actual
        <<", expected="<<expected<<", relative error="<<relative
        <<", maximum holdout="<<maximum_holdout
        <<", retained terms=["<<minimum_terms<<","<<maximum_terms<<"]"
        <<", maximum refit condition="<<maximum_refit_condition
        <<", master evaluations="<<analytic.master_integral_evaluations;
    require(relative<8.0e-8,analytic_details.str());
    require(analytic.maximum_imaginary_to_real<1.0e-10,
            "analytic B321II pure-power imaginary cancellation");
}

void test_exact_finite_uv_restorations() {
    const PowerLawSpectrum unit_power(1.0,0.0);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const std::array<eft::Vec3,3> external={{
        canonical.k1,canonical.k2,canonical.k3}};
    const std::array<double,3> sides={{
        canonical.sides.k1,canonical.sides.k2,canonical.sides.k3}};

    const eft::MatterKernelProvider matter;
    const eft::SparsePolynomial matter_b321ii=
        eft::fftlog_b321ii_exact_uv_restoration(
            unit_power,triangle,matter,1.0);
    double expected_matter_b321ii=0.0;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            const double F2=eft::spt_F({
                external[static_cast<std::size_t>(left)],
                external[static_cast<std::size_t>(right)]});
            expected_matter_b321ii+=
                (-61.0/315.0)*F2
                *(sides[static_cast<std::size_t>(left)]
                    *sides[static_cast<std::size_t>(left)]
                  +sides[static_cast<std::size_t>(right)]
                    *sides[static_cast<std::size_t>(right)]);
        }
    }
    require_close(
        matter_b321ii.evaluate({}),expected_matter_b321ii,
        3.0e-14,2.0e-16,
        "exact B321II P13 UV restoration matter normalization");

    const eft::EftBiasKernelProvider biased;
    std::array<double,eft::kParameterCount> values={};
    values[static_cast<std::size_t>(eft::ParameterId::B1)]=2.2;
    values[static_cast<std::size_t>(eft::ParameterId::B2)]=0.3;
    values[static_cast<std::size_t>(eft::ParameterId::Gamma2)]=-0.2;
    values[static_cast<std::size_t>(eft::ParameterId::Gamma21)]=0.13;
    const double alpha=(
        -61.0*values[static_cast<std::size_t>(eft::ParameterId::B1)]
        -576.0*values[
            static_cast<std::size_t>(eft::ParameterId::Gamma2)]
        +672.0*values[
            static_cast<std::size_t>(eft::ParameterId::Gamma21)])/315.0;
    double expected_biased_b321ii=0.0;
    for (int left=0;left<3;++left) {
        for (int right=left+1;right<3;++right) {
            const double K2=biased.deterministic({
                external[static_cast<std::size_t>(left)],
                external[static_cast<std::size_t>(right)]}).evaluate(values);
            expected_biased_b321ii+=
                K2*values[static_cast<std::size_t>(eft::ParameterId::B1)]
                *alpha
                *(sides[static_cast<std::size_t>(left)]
                    *sides[static_cast<std::size_t>(left)]
                  +sides[static_cast<std::size_t>(right)]
                    *sides[static_cast<std::size_t>(right)]);
        }
    }
    const eft::SparsePolynomial biased_b321ii=
        eft::fftlog_b321ii_exact_uv_restoration(
            unit_power,triangle,biased,1.0);
    require_close(
        biased_b321ii.evaluate(values),expected_biased_b321ii,
        3.0e-14,2.0e-15,
        "exact B321II P13 UV restoration bias polynomial");

    // Independent exact-rational evaluations of the f=0 FullForm stored in
    // upstream b411uv-flat.wdx, summed over its 123, 231 and 132 routes for
    // unit external powers and unit normalized linear-power moment.
    static constexpr std::array<double,eft::kBiasParameterCount>
        expected_b411={{
            -0.0015248031590731416243,
            -0.0025114590359901546014,
             0.0,
            -0.015453705495136658876,
             0.098726631314718463279,
            -0.016646893884135488120,
             0.027040469877896623315,
            -0.033183961593941514097,
            -0.023177545609625677127,
             0.021403149279602770439,
            -0.0040537756706370560658}};
    static constexpr std::array<eft::ParameterId,eft::kBiasParameterCount>
        operator_ids={{
            eft::ParameterId::B1,
            eft::ParameterId::B2,
            eft::ParameterId::B3,
            eft::ParameterId::Gamma2,
            eft::ParameterId::Gamma21,
            eft::ParameterId::Gamma211,
            eft::ParameterId::Gamma21x,
            eft::ParameterId::Gamma22,
            eft::ParameterId::Gamma2x,
            eft::ParameterId::Gamma3,
            eft::ParameterId::Gamma31}};
    const eft::SparsePolynomial biased_b411=
        eft::fftlog_b411_exact_uv_restoration(
            unit_power,triangle,biased,1.0);
    for (std::size_t index=0;index<operator_ids.size();++index) {
        std::vector<eft::MonomialFactor> factors;
        if (index==0) {
            factors.push_back({eft::ParameterId::B1,3});
        } else {
            factors.push_back({eft::ParameterId::B1,2});
            factors.push_back({operator_ids[index],1});
        }
        const double actual=biased_b411.coefficient(
            eft::MonomialKey(std::move(factors)));
        require_close(
            actual,expected_b411[index],
            8.0e-13,3.0e-15,
            "exact generated B411 UV operator "+std::to_string(index));
    }
    require(
        biased_b411.terms().size()==10,
        "renormalized B411 UV table has ten nonzero real-space operators");
    require_close(
        eft::fftlog_b411_exact_uv_restoration(
            unit_power,triangle,matter,1.0).evaluate({}),
        expected_b411[0],8.0e-13,3.0e-15,
        "exact generated B411 UV matter projection");
}

void test_exact_k4_uv_asymptotic_match() {
    const PowerLawSpectrum unit_power(1.0,0.0);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::MatterKernelProvider matter;
    const eft::DiagramAssembler assembler(
        unit_power,triangle,matter);
    eft::UvSubtractionConfig config;
    config.mu_ren=0.30;
    config.asymptotic_factor=64.0;
    const eft::UvSubtraction uv(assembler,config);
    const eft::UvSubtractionComponents local=
        uv.angular_averaged_subleading_coefficients(16,16);
    const double exact_b321ii=
        eft::fftlog_b321ii_exact_uv_restoration(
            unit_power,triangle,matter,1.0).evaluate({});
    const double exact_b411=
        eft::fftlog_b411_exact_uv_restoration(
            unit_power,triangle,matter,1.0).evaluate({});
    require_close(
        local.B321II.evaluate({}),exact_b321ii,
        3.0e-7,2.0e-13,
        "local large-q P13 coefficient matches exact public restoration");
    require_close(
        local.B411.evaluate({}),exact_b411,
        3.0e-5,2.0e-12,
        "exact opposite K4 large-q coefficient matches public B411 UV table");
}

void test_analytic_one_loop_assembly() {
    constexpr double amplitude=1.3;
    constexpr double exponent=-1.25;
    const PowerLawSpectrum source(amplitude,exponent);
    eft::FftlogConfig fftlog;
    fftlog.kmin=1.0e-4;
    fftlog.kmax=100.0;
    fftlog.frequency_count=16;
    fftlog.reconstruction_grid_size=128;
    fftlog.bias_nu=exponent;
    const eft::FftlogDrOracle oracle(source,fftlog);
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle(
        {0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::MatterKernelProvider provider;
    eft::FftlogAnalyticOneLoopConfig config;
    config.B222_laurent.validation_samples=16;
    config.B321I_laurent.validation_samples=16;
    config.B321II_laurent.validation_samples=16;
    config.uv_restoration.enabled=false;
    const auto analytic=oracle.evaluate_analytic(
        triangle,provider,config);
    const auto convolutions=
        oracle.evaluate_convolutions_analytic(
            triangle,provider,config);

    const auto& templates=analytic.templates;
    require(
        convolutions.B222.value.serialize()
            ==analytic.B222.value.serialize(),
        "convolution-only analytic path preserves B222");
    require(
        convolutions.B321I.extrapolated.value.serialize()
            ==analytic.B321I.extrapolated.value.serialize(),
        "convolution-only analytic path preserves B321I");
    require(
        convolutions.B411.extrapolated.value.serialize()
            ==analytic.B411.extrapolated.value.serialize(),
        "convolution-only analytic path preserves B411");
    require(
        convolutions.master_integral_evaluations
            ==convolutions.B222.master_integral_evaluations
                +convolutions.B321I.extrapolated
                    .master_integral_evaluations
                +convolutions.B411.extrapolated
                    .master_integral_evaluations
        &&analytic.master_integral_evaluations
            ==convolutions.master_integral_evaluations
                +analytic.B321II.extrapolated
                    .master_integral_evaluations,
        "convolution-only analytic path accounts for exactly its master calls");
    require(
        templates.B222.serialize()==analytic.B222.value.serialize(),
        "analytic one-loop assembly preserves B222");
    require(
        templates.B321I.serialize()
            ==analytic.B321I.extrapolated.value.serialize(),
        "analytic one-loop assembly preserves extrapolated B321I");
    require(
        templates.B321II.serialize()
            ==analytic.B321II_factorized.value.serialize(),
        "analytic one-loop assembly uses factorized renormalized P13 B321II");
    require(
        templates.B321II.serialize()
            !=analytic.B321II.extrapolated.value.serialize(),
        "flattened-master B321II remains diagnostic-only");
    require(
        templates.B411.serialize()
            ==analytic.B411.extrapolated.value.serialize(),
        "analytic one-loop assembly preserves extrapolated B411");
    const double topology_sum=
        templates.B222.evaluate({})+templates.B321I.evaluate({})
        +templates.B321II.evaluate({})+templates.B411.evaluate({});
    require_close(
        templates.one_loop.evaluate({}),topology_sum,
        2.0e-14,1.0e-12,
        "analytic one-loop assembly topology sum");
    require_close(
        templates.total.evaluate({}),
        templates.tree.evaluate({})+templates.one_loop.evaluate({}),
        2.0e-14,1.0e-12,
        "analytic one-loop assembly tree plus loop");
    require(
        templates.B321II_bare.serialize()==templates.B321II.serialize()
        && templates.B411_bare.serialize()==templates.B411.serialize()
        && templates.uv_subtraction_total.empty(),
        "analytic DR assembly records scaleless projectors as zero");
    require(!templates.ir_safe,
            "analytic DR assembly does not overclaim pointwise IR remapping");
    require(
        templates.integration_nodes
            ==analytic.master_integral_evaluations
                +analytic.B321II_factorized.integration_nodes
        && analytic.master_integral_evaluations>0
        && analytic.B321II_factorized.integration_nodes>0,
        "analytic one-loop assembly accounts for master calls and factorized P13 nodes");

    const std::array<const eft::FftlogRegulatorDiagnostic*,3> diagnostics={{
        &analytic.B321I.regulator,
        &analytic.B321II.regulator,
        &analytic.B411.regulator}};
    for (std::size_t index=0;index<diagnostics.size();++index) {
        const auto& diagnostic=*diagnostics[index];
        require_close(
            diagnostic.fine_epsilon,
            diagnostic.coarse_epsilon/diagnostic.refinement_ratio,
            2.0e-15,0.0,
            "analytic regulator fine epsilon topology "
            +std::to_string(index));
        require_close(
            diagnostic.maximum_absolute_extrapolation_correction,
            diagnostic.maximum_absolute_coarse_to_fine
                /(diagnostic.refinement_ratio-1.0),
            1.0e-8,2.0e-15,
            "analytic regulator Richardson identity topology "
            +std::to_string(index));
        require(
            std::isfinite(diagnostic.maximum_relative_coarse_to_fine)
            && std::isfinite(
                diagnostic.maximum_relative_extrapolation_correction),
            "analytic regulator diagnostics are finite topology "
            +std::to_string(index));
    }

    const std::string metadata=oracle.analytic_metadata_json(config);
    require(
        metadata.find("\"schema\":\"marisa-b-fftlog-dr-analytic-v2\"")
            !=std::string::npos,
        "analytic one-loop metadata schema");
    require(
        metadata.find(
            "\"B321II\":\"factorized renormalized source-power P13\"")
            !=std::string::npos,
        "analytic one-loop metadata records factorized B321II production path");
    require(
        metadata.find(
            "\"B321II_flattened_master\":\"diagnostic only\"")
            !=std::string::npos,
        "analytic one-loop metadata demotes flattened B321II to diagnostic");
    require(
        metadata.find("\"analytic_master_integrals\":true")
            !=std::string::npos,
        "analytic one-loop metadata truthfully records master integrals");
    require(
        metadata.find("\"B321I\":[-3,3]")
            !=std::string::npos,
        "analytic one-loop metadata records topology-specific Laurent range");
    require(
        metadata.find("\"extrapolation_order\":1")
            !=std::string::npos,
        "analytic one-loop metadata records regulator extrapolation");

    eft::FftlogAnalyticOneLoopConfig sector_config=config;
    sector_config.contour_sectors.enabled=true;
    sector_config.contour_sectors.B222_default_bias=exponent;
    sector_config.contour_sectors.B222_constant_bias=exponent;
    sector_config.contour_sectors.B321I_soft_bias=exponent;
    sector_config.contour_sectors.B321I_local_quadratic_bias=exponent;
    sector_config.contour_sectors.B321I_composite_bias=exponent;
    sector_config.contour_sectors.B411_bias=exponent;
    const auto sector_analytic=oracle.evaluate_analytic(
        triangle,provider,sector_config);
    require_polynomial_close(
        sector_analytic.B222.value,analytic.B222.value,
        3.0e-12,2.0e-11,
        "matter B222 contour-sector assembly equals single-contour result");
    require_polynomial_close(
        sector_analytic.B321I.extrapolated.value,
        analytic.B321I.extrapolated.value,
        3.0e-12,2.0e-11,
        "matter B321I contour-sector assembly equals single-contour result");
    require_polynomial_close(
        sector_analytic.B411.extrapolated.value,
        analytic.B411.extrapolated.value,
        3.0e-12,2.0e-11,
        "matter B411 contour-sector assembly equals single-contour result");
    const std::string sector_metadata=
        oracle.analytic_metadata_json(sector_config);
    require(
        sector_metadata.find(
            "\"contour_sectors\":{\"enabled\":true")
            !=std::string::npos,
        "analytic metadata records enabled contour sectors");
    require(
        sector_metadata.find(
            "\"soft_operator_indices\":[0,5]")
            !=std::string::npos
        &&sector_metadata.find(
            "\"local_quadratic_operator_indices\":[1]")
            !=std::string::npos
        &&sector_metadata.find(
            "\"composite_operator_indices\":[2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]")
            !=std::string::npos,
        "analytic metadata records the exact B321I sector partition");
}

void test_selected_triangle_scheme_conversion(const TabulatedPower& source) {
    const hv1::CanonicalTriangle canonical=hv1::canonicalize_triangle({0.045,0.067,0.083});
    const std::array<eft::Vec3,3> triangle={{canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider provider;
    eft::DirectIntegrationConfig integration;
    integration.qmin=1.0e-3;
    integration.qmax=0.50;
    integration.n_radial=10;
    integration.n_mu=14;
    integration.n_phi=10;
    integration.ir_safe=true;
    integration.uv_subtract=true;

    const eft::DiagramTemplates direct=eft::evaluate_direct(
        source,triangle,provider,integration);
    eft::FftlogConfig fftlog_config;
    fftlog_config.kmin=1.0e-6;
    fftlog_config.kmax=1.0e3;
    fftlog_config.frequency_count=512;
    fftlog_config.reconstruction_grid_size=8192;
    const eft::FftlogDrOracle oracle(source,fftlog_config);
    const eft::DiagramTemplates transformed=oracle.evaluate(
        triangle,provider,integration);
    const auto bias=selected_bias_point();
    require_close(
        transformed.tree.evaluate(bias),direct.tree.evaluate(bias),
        1.0e-3,1.0e-9,"selected-triangle FFTLog/DR tree");
    require_close(
        transformed.total.evaluate(bias),direct.total.evaluate(bias),
        1.0e-3,1.0e-6,"selected-triangle FFTLog/DR renormalized total");
    require_close(
        transformed.one_loop.evaluate(bias),direct.one_loop.evaluate(bias),
        1.0e-2,1.0e-6,"selected-triangle FFTLog/DR renormalized loop");
    require(transformed.integration_nodes!=direct.integration_nodes,
            "selected-triangle oracle uses an independent node set");
    const std::string metadata=oracle.metadata_json(integration);
    require(metadata.find("\"schema\":\"marisa-b-fftlog-dr-oracle-v1\"")!=std::string::npos,
            "FFTLog/DR oracle metadata schema");
    require(metadata.find("\"renormalization\":")!=std::string::npos,
            "FFTLog/DR oracle metadata scheme");
    require(metadata.find("\"analytic_master_integrals\":false")!=std::string::npos,
            "FFTLog oracle does not overclaim analytic DR master integrals");
    require(metadata.find("\"analytic_three_propagator_master_available\":true")
                !=std::string::npos,
            "FFTLog oracle records analytic three-propagator master availability");
    require(metadata.find("\"analytic_diagram_laurent_reduction\":false")
                !=std::string::npos,
            "FFTLog oracle records missing diagram Laurent reduction");
    require(metadata.find("antipodal Fibonacci sphere")!=std::string::npos,
            "FFTLog/DR oracle records independent angular rule");
}

void test_reconstructed_direct_uv_tail(const TabulatedPower& source) {
    const hv1::CanonicalTriangle canonical=
        hv1::canonicalize_triangle({0.055,0.073,0.091});
    const std::array<eft::Vec3,3> triangle={{
        canonical.k1,canonical.k2,canonical.k3}};
    const eft::EftBiasKernelProvider base;
    hv1::ReconstructionConfig reconstruction;
    reconstruction.enabled=true;
    reconstruction.smoothing_radius=15.0;
    reconstruction.bias_recon=2.7340475186190334;
    reconstruction.cell_size=8.0;
    const eft::ReconstructedFieldKernelProvider provider(
        base,reconstruction);

    eft::DirectIntegrationConfig integration;
    integration.qmin=1.0e-4;
    integration.qmax=0.8;
    integration.n_radial=6;
    integration.n_mu=10;
    integration.n_phi=12;
    integration.ir_safe=true;
    integration.uv_subtract=true;
    integration.restore_uv_tail=true;
    integration.uv_tail_kmax=30.0;
    integration.uv_tail_quadrature_order=128;
    integration.uv_tail_n_mu=16;
    integration.uv_tail_n_phi=20;
    integration.exploit_phi_reflection=false;
    integration.diagram_mask=
        eft::kDirectB321II|eft::kDirectB411;

    const eft::DiagramTemplates restored=
        eft::evaluate_direct(
            source,triangle,provider,integration);
    require(
        !restored.B321II_uv_restoration.empty(),
        "reconstructed direct B321II UV tail is present");
    require(
        !restored.B411_uv_restoration.empty(),
        "reconstructed direct B411 UV tail is present");
    require_polynomial_close(
        restored.B321II,
        restored.B321II_bare
            -restored.B321II_uv_subtraction
            +restored.B321II_uv_restoration,
        2.0e-5,1.0e-6,
        "reconstructed direct B321II tail recomposition");
    require_polynomial_close(
        restored.B411,
        restored.B411_bare
            -restored.B411_uv_subtraction
            +restored.B411_uv_restoration,
        2.0e-5,1.0e-6,
        "reconstructed direct B411 tail recomposition");

    eft::DirectIntegrationConfig bare=integration;
    bare.restore_uv_tail=false;
    const eft::DiagramTemplates truncated=
        eft::evaluate_direct(
            source,triangle,provider,bare);
    require_polynomial_close(
        restored.B321II-truncated.B321II,
        restored.B321II_uv_restoration,
        2.0e-12,1.0e-8,
        "reconstructed B321II restored-minus-truncated");
    require_polynomial_close(
        restored.B411-truncated.B411,
        restored.B411_uv_restoration,
        2.0e-12,1.0e-8,
        "reconstructed B411 restored-minus-truncated");

    eft::DirectIntegrationConfig all_topologies_config=
        integration;
    all_topologies_config.diagram_mask=
        eft::kDirectAllDiagrams;
    // B321I is a UV-finite two-power convolution rather than a tadpole
    // topology.  Its physical q^2 P(q)^2 tail is still material between
    // qmax=0.8 and 1.6, so that pair is not a renormalization test.  Test
    // the registered production plateau instead, with a quadrature that
    // independently resolves the small remaining tail.
    all_topologies_config.qmax=6.4;
    all_topologies_config.n_radial=12;
    all_topologies_config.n_mu=24;
    all_topologies_config.n_phi=24;
    const eft::DiagramTemplates restored_all=
        eft::evaluate_direct(
            source,triangle,provider,
            all_topologies_config);
    eft::DirectIntegrationConfig
        all_topologies_extended=
        all_topologies_config;
    all_topologies_extended.qmax=12.8;
    const eft::DiagramTemplates restored_all_extended=
        eft::evaluate_direct(
            source,triangle,provider,
            all_topologies_extended);
    const std::array<eft::SparsePolynomial,4> first_topologies={{
        restored_all.B222,restored_all.B321I,
        restored_all.B321II,restored_all.B411}};
    const std::array<eft::SparsePolynomial,4> extended_topologies={{
        restored_all_extended.B222,
        restored_all_extended.B321I,
        restored_all_extended.B321II,
        restored_all_extended.B411}};
    const std::array<std::string,4> topology_names={{
        "B222","B321I","B321II","B411"}};
    for (std::size_t topology=0;
         topology<first_topologies.size();
         ++topology) {
        require_polynomial_close(
            first_topologies[topology],
            extended_topologies[topology],
            3.0e-2,3.0e-3,
            "reconstructed "+topology_names[topology]
                +" qmax monomial stability");
    }
    std::array<double,eft::kParameterCount> second_bias=
        selected_bias_point();
    second_bias[
        static_cast<std::size_t>(
            eft::ParameterId::B1)]=1.45;
    second_bias[
        static_cast<std::size_t>(
            eft::ParameterId::B2)]=-0.55;
    second_bias[
        static_cast<std::size_t>(
            eft::ParameterId::Gamma2)]=0.28;
    second_bias[
        static_cast<std::size_t>(
            eft::ParameterId::B3)]=-0.18;
    for (const auto& parameters:
         {selected_bias_point(),second_bias}) {
        for (std::size_t topology=0;
             topology<first_topologies.size();
             ++topology) {
            require_close(
                first_topologies[topology].evaluate(
                    parameters),
                extended_topologies[topology].evaluate(
                    parameters),
                3.0e-2,2.0e-5,
                "reconstructed "
                    +topology_names[topology]
                    +" qmax bias-point stability");
        }
    }

    eft::DirectIntegrationConfig shifted_reference=
        integration;
    shifted_reference.uv.mu_ren=0.18;
    shifted_reference.uv.asymptotic_factor=96.0;
    const eft::DiagramTemplates reference_variant=
        eft::evaluate_direct(
            source,triangle,provider,
            shifted_reference);
    require_polynomial_close(
        restored.B321II_uv_restoration,
        reference_variant.B321II_uv_restoration,
        5.0e-3,3.0e-3,
        "reconstructed B321II reference-q stability");
    require_polynomial_close(
        restored.B411_uv_restoration,
        reference_variant.B411_uv_restoration,
        5.0e-3,3.0e-3,
        "reconstructed B411 reference-q stability");

    hv1::ReconstructionConfig no_shift=
        reconstruction;
    no_shift.bias_recon=1.0e300;
    const eft::ReconstructedFieldKernelProvider
        no_shift_provider(base,no_shift);
    const eft::DiagramTemplates no_shift_value=
        eft::evaluate_direct(
            source,triangle,no_shift_provider,
            integration);
    const eft::DiagramTemplates pre_value=
        eft::evaluate_direct(
            source,triangle,base,integration);
    require_polynomial_close(
        no_shift_value.B321II_uv_restoration,
        pre_value.B321II_uv_restoration,
        5.0e-12,2.0e-7,
        "R-infinity B321II tail equals pre tail");
    require_polynomial_close(
        no_shift_value.B411_uv_restoration,
        pre_value.B411_uv_restoration,
        5.0e-12,2.0e-7,
        "R-infinity B411 tail equals pre tail");

    bool reflection_rejected=false;
    try {
        eft::DirectIntegrationConfig invalid=
            integration;
        invalid.exploit_phi_reflection=true;
        (void)eft::evaluate_direct(
            source,triangle,provider,invalid);
    } catch (const std::invalid_argument&) {
        reflection_rejected=true;
    }
    require(
        reflection_rejected,
        "Cartesian reconstructed provider rejects phi reflection");
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("usage: test_uv_renormalization POWER_TABLE");
        const TabulatedPower source(argv[1]);
        test_fftlog_power_reconstruction(source);
        test_public_table_operator_masks();
        test_analytic_three_propagator_master();
        test_b222_laurent_reduction();
        test_b321i_route_laurent_reduction();
        test_b321ii_route_laurent_reduction();
        test_analytic_b411_generated_table();
        test_analytic_b222_power_law();
        test_analytic_b321i_power_law();
        test_analytic_b321ii_power_law();
        test_exact_finite_uv_restorations();
        test_exact_k4_uv_asymptotic_match();
        test_analytic_one_loop_assembly();
        test_selected_triangle_scheme_conversion(source);
        test_reconstructed_direct_uv_tail(source);
        std::cout<<"EFT-v2 UV/FFTLog checks passed: "<<checks<<"\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 UV/FFTLog test failed: "<<error.what()<<"\n";
        return 1;
    }
}
