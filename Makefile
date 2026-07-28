CXX ?= g++
PYTHON ?= python3

FORMAL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DM_DIR ?= $(FORMAL_ROOT)/src/marisa_b
REACT_DIR ?= $(FORMAL_ROOT)/external/ACTio-ReACTio
POWER_TABLE ?= $(FORMAL_ROOT)/tests/data/quijote_fiducial_plin_z1.dat
BUILD_DIR ?= build/halo_v1
BUILD_DEFINES ?=
GSL_PREFIX ?=

ifneq ($(strip $(GSL_PREFIX)),)
GSL_LIBDIR ?= $(firstword $(wildcard $(GSL_PREFIX)/lib/*-linux-gnu) $(GSL_PREFIX)/lib)
GSL_CFLAGS := -I$(GSL_PREFIX)/include
GSL_LIBS := -L$(GSL_LIBDIR) -Wl,--disable-new-dtags,-rpath,$(GSL_LIBDIR) -lgsl -lgslcblas -lm
else
GSL_CFLAGS := $(shell pkg-config --cflags gsl 2>/dev/null)
GSL_LIBS := $(shell pkg-config --libs gsl 2>/dev/null || printf '%s' '-lgsl -lgslcblas -lm')
endif

DM_CPP := $(DM_DIR)/marisa_b_native.cpp
DM_HEADER := $(DM_DIR)/marisa_b_native.h
REACT_SRC := $(REACT_DIR)/reactions/src
INCLUDES := -I src/halo_v1 -I src/eft_v2 -I $(DM_DIR) -I $(REACT_DIR)/reactions -I $(REACT_SRC) $(GSL_CFLAGS)
BASE_FLAGS := -std=gnu++17 -O2 -g -fopenmp -ffunction-sections -fdata-sections -MMD -MP -DHAVE_CONFIG_H $(BUILD_DEFINES) $(INCLUDES)
PROJECT_FLAGS := $(BASE_FLAGS) -Wall -Wextra -Wpedantic
LINK_FLAGS := -fopenmp -Wl,--gc-sections $(GSL_LIBS)

TEST_BINARY := $(BUILD_DIR)/test_halo_v1
EFT_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_template_algebra
EFT_MODEL_CONFIG_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_model_config
EFT_BIAS_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_bias_operators
EFT_DIAGRAM_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_diagrams
EFT_UV_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_uv_renormalization
EFT_COUNTERTERM_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_counterterms
EFT_STOCHASTIC_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_stochastic
EFT_IR_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_ir_resummation
EFT_SHELL_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_shell_projector
EFT_POWER_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_tracer_power
EFT_POST_R1_PNG_TEST_BINARY := $(BUILD_DIR)/test_eft_v2_post_r1_png_native
SHELL_BINARY := $(BUILD_DIR)/shell_checkpoint
EFT_R0_DRIVER_BINARY := $(BUILD_DIR)/eft_v2_r0_template_driver
EFT_R0_POWER_DRIVER_BINARY := $(BUILD_DIR)/eft_v2_r0_power_driver
EFT_POST_R1_DRIVER_BINARY := $(BUILD_DIR)/eft_v2_post_r1_template_driver
EFT_POST_R1_PNG_DRIVER_BINARY := $(BUILD_DIR)/eft_v2_post_r1_png_template_driver
MARISA_B_TRIANGLE_BINARY := $(BUILD_DIR)/marisa_b_triangle
MARISA_B_TRIANGLE_ADAPTER_DIR := $(BUILD_DIR)/marisa_b_triangle_adapter
MARISA_B_TRIANGLE_HELPER := scripts/build_marisa_b_triangle.py
MARISA_B_TRIANGLE_REACT_INPUTS := \
	$(REACT_SRC)/BSPT.cpp \
	$(REACT_SRC)/BSPT.h \
	$(REACT_SRC)/Common.cpp \
	$(REACT_SRC)/Cosmology.cpp \
	$(REACT_SRC)/InterpolatedPS.cpp \
	$(REACT_SRC)/PowerSpectrum.cpp \
	$(REACT_SRC)/Quadrature.cpp \
	$(REACT_SRC)/SPT.cpp \
	$(REACT_SRC)/SpecialFunctions.cpp \
	$(REACT_SRC)/Spline.cpp \
	$(REACT_SRC)/array.cpp
EFT_OBJECTS := \
	$(BUILD_DIR)/eft_v2_parameter_registry.o \
	$(BUILD_DIR)/eft_v2_model_config.o \
	$(BUILD_DIR)/eft_v2_template_algebra.o \
	$(BUILD_DIR)/eft_v2_kernel_primitives.o \
	$(BUILD_DIR)/eft_v2_bias_operators.o \
	$(BUILD_DIR)/eft_v2_field_kernel_provider.o \
	$(BUILD_DIR)/eft_v2_diagram_assembler.o \
	$(BUILD_DIR)/eft_v2_ir_safe_integrands.o \
	$(BUILD_DIR)/eft_v2_uv_subtraction.o \
	$(BUILD_DIR)/eft_v2_direct_evaluator.o \
	$(BUILD_DIR)/eft_v2_fftlog_dr_oracle.o \
	$(BUILD_DIR)/eft_v2_counterterms.o \
	$(BUILD_DIR)/eft_v2_stochastic.o \
	$(BUILD_DIR)/eft_v2_poisson_reconstruction.o \
	$(BUILD_DIR)/eft_v2_ir_resummation.o \
	$(BUILD_DIR)/eft_v2_lattice_shell_rule.o \
	$(BUILD_DIR)/eft_v2_shell_projector.o \
	$(BUILD_DIR)/eft_v2_tracer_power.o
CORE_OBJECTS := \
	$(BUILD_DIR)/halo_v1.o \
	$(BUILD_DIR)/shell_average.o \
	$(BUILD_DIR)/marisa_b_native.o \
	$(BUILD_DIR)/Common.o \
	$(BUILD_DIR)/PowerSpectrum.o \
	$(BUILD_DIR)/array.o \
	$(BUILD_DIR)/Quadrature.o
POST_R1_PNG_OBJECTS := \
	$(BUILD_DIR)/eft_v2_post_r1_png_template_driver.o \
	$(EFT_OBJECTS) \
	$(BUILD_DIR)/halo_v1.o \
	$(BUILD_DIR)/shell_average.o \
	$(BUILD_DIR)/marisa_b_native.o \
	$(BUILD_DIR)/Common.o \
	$(BUILD_DIR)/PowerSpectrum.o \
	$(BUILD_DIR)/array.o \
	$(BUILD_DIR)/Quadrature.o
TEST_OBJECTS := $(BUILD_DIR)/test_halo_v1.o $(CORE_OBJECTS)
SHELL_OBJECTS := $(BUILD_DIR)/shell_checkpoint.o $(CORE_OBJECTS)
OBJECTS := $(TEST_OBJECTS) $(BUILD_DIR)/shell_checkpoint.o $(EFT_OBJECTS) \
	$(BUILD_DIR)/test_eft_v2_template_algebra.o \
	$(BUILD_DIR)/test_eft_v2_model_config.o \
	$(BUILD_DIR)/test_eft_v2_bias_operators.o \
	$(BUILD_DIR)/test_eft_v2_diagrams.o \
	$(BUILD_DIR)/test_eft_v2_uv_renormalization.o \
	$(BUILD_DIR)/test_eft_v2_counterterms.o \
	$(BUILD_DIR)/test_eft_v2_stochastic.o \
	$(BUILD_DIR)/test_eft_v2_ir_resummation.o \
	$(BUILD_DIR)/test_eft_v2_shell_projector.o \
	$(BUILD_DIR)/test_eft_v2_tracer_power.o \
	$(BUILD_DIR)/eft_v2_r0_template_driver.o \
	$(BUILD_DIR)/eft_v2_r0_power_driver.o \
	$(BUILD_DIR)/eft_v2_post_r1_template_driver.o
DEPFILES := $(OBJECTS:.o=.d) $(POST_R1_PNG_OBJECTS:.o=.d)

.PHONY: binary shell-binary marisa-b-triangle marisa-b-triangle-test eft-v2-r0-driver eft-v2-r0-power-driver eft-v2-post-r1-driver eft-v2-post-r1-png-driver eft-v2-post-r1-png-test eft-v2-model-config-test eft-v2-test legacy-r0-mean-statistics-test python-test test release-test external-inference-regression verify-inputs verify-release verify-canonical-native

binary: verify-inputs $(TEST_BINARY)

shell-binary: verify-inputs $(SHELL_BINARY)
	@! nm -C --defined-only $(SHELL_BINARY) | rg -q ' [TW] marisa_b_native::compute_(pre|post)_recon_halo_bias'

marisa-b-triangle: verify-inputs $(MARISA_B_TRIANGLE_BINARY)

marisa-b-triangle-test: marisa-b-triangle
	$(PYTHON) tests/test_build_marisa_b_triangle.py \
		--build-dir $(MARISA_B_TRIANGLE_ADAPTER_DIR) \
		--binary $(MARISA_B_TRIANGLE_BINARY)

eft-v2-r0-driver: verify-inputs $(EFT_R0_DRIVER_BINARY)

eft-v2-r0-power-driver: verify-inputs $(EFT_R0_POWER_DRIVER_BINARY)

eft-v2-post-r1-driver: verify-inputs $(EFT_POST_R1_DRIVER_BINARY)

eft-v2-post-r1-png-driver: verify-inputs $(EFT_POST_R1_PNG_DRIVER_BINARY)

eft-v2-post-r1-png-test: verify-inputs $(EFT_POST_R1_PNG_TEST_BINARY)
	$(EFT_POST_R1_PNG_TEST_BINARY)

eft-v2-model-config-test: verify-inputs $(EFT_MODEL_CONFIG_TEST_BINARY)
	$(EFT_MODEL_CONFIG_TEST_BINARY)

legacy-r0-mean-statistics-test:
	$(PYTHON) tests/legacy/test_fit_eft_v2_r0_mean.py

python-test:
	$(PYTHON) -m pytest -q tests/python

eft-v2-test: verify-inputs $(EFT_TEST_BINARY) $(EFT_MODEL_CONFIG_TEST_BINARY) $(EFT_BIAS_TEST_BINARY) $(EFT_DIAGRAM_TEST_BINARY) $(EFT_UV_TEST_BINARY) $(EFT_COUNTERTERM_TEST_BINARY) $(EFT_STOCHASTIC_TEST_BINARY) $(EFT_IR_TEST_BINARY) $(EFT_SHELL_TEST_BINARY) $(EFT_POWER_TEST_BINARY) $(EFT_POST_R1_PNG_TEST_BINARY)
	$(EFT_TEST_BINARY)
	$(EFT_MODEL_CONFIG_TEST_BINARY)
	$(EFT_BIAS_TEST_BINARY)
	$(EFT_DIAGRAM_TEST_BINARY) $(POWER_TABLE)
	$(EFT_UV_TEST_BINARY) $(POWER_TABLE)
	$(EFT_COUNTERTERM_TEST_BINARY)
	$(EFT_STOCHASTIC_TEST_BINARY)
	$(EFT_IR_TEST_BINARY)
	$(EFT_SHELL_TEST_BINARY)
	$(EFT_POWER_TEST_BINARY)
	$(EFT_POST_R1_PNG_TEST_BINARY)
	$(PYTHON) tests/legacy/test_fit_eft_v2_r0_mean.py
	$(PYTHON) -m pytest -q tests/python

test: verify-inputs $(TEST_BINARY) $(EFT_TEST_BINARY) $(EFT_MODEL_CONFIG_TEST_BINARY) $(EFT_BIAS_TEST_BINARY) $(EFT_DIAGRAM_TEST_BINARY) $(EFT_UV_TEST_BINARY) $(EFT_COUNTERTERM_TEST_BINARY) $(EFT_STOCHASTIC_TEST_BINARY) $(EFT_IR_TEST_BINARY) $(EFT_SHELL_TEST_BINARY) $(EFT_POWER_TEST_BINARY) $(EFT_POST_R1_PNG_TEST_BINARY)
	$(TEST_BINARY) $(POWER_TABLE)
	$(EFT_TEST_BINARY)
	$(EFT_MODEL_CONFIG_TEST_BINARY)
	$(EFT_BIAS_TEST_BINARY)
	$(EFT_DIAGRAM_TEST_BINARY) $(POWER_TABLE)
	$(EFT_UV_TEST_BINARY) $(POWER_TABLE)
	$(EFT_COUNTERTERM_TEST_BINARY)
	$(EFT_STOCHASTIC_TEST_BINARY)
	$(EFT_IR_TEST_BINARY)
	$(EFT_SHELL_TEST_BINARY)
	$(EFT_POWER_TEST_BINARY)
	$(EFT_POST_R1_PNG_TEST_BINARY)
	$(PYTHON) tests/legacy/test_fit_eft_v2_r0_mean.py
	$(PYTHON) -m pytest -q tests/python
	$(MAKE) verify-canonical-native

release-test:
	+$(MAKE) test
	+$(MAKE) marisa-b-triangle-test
	+$(MAKE) verify-release

external-inference-regression: marisa-b-triangle
	@test -n "$(MARISA_B_DATA_ROOT)" || \
		{ echo "MARISA_B_DATA_ROOT is required"; exit 2; }
	$(PYTHON) scripts/verify_inference_regression.py \
		--data-root "$(MARISA_B_DATA_ROOT)" \
		--triangle-binary "$(MARISA_B_TRIANGLE_BINARY)"

verify-inputs:
	@$(PYTHON) scripts/verify_repository_boundary.py --allow-dirty >/dev/null
	@sha256sum $(POWER_TABLE) | rg -q '^974ffe272cb6d15bb279c63ee64a6183540485cc5a1b6d0d780c4418176679aa '
	@sha256sum $(DM_CPP) | rg -q '^d3ddf662ca5bb8a36c8126dd72dfb4c19b6b61c87ec6b65047c4c1361bc112b2 '
	@sha256sum $(DM_HEADER) | rg -q '^28743d88b165f74edf2708d995a066336841c3c6a0ea72364b4df985f80eb184 '
	@test "$$(git -C $(REACT_DIR) rev-parse HEAD)" = a36dda02db6b537a4ddcc7bf044c446c10a58373
	@test "$$(git -C $(REACT_DIR) archive HEAD | sha256sum | awk '{print $$1}')" = d2f3b86ac85016aa1c386dd91299284ea72a83ae0480652be02f60c15655d48d
	@sha256sum $(REACT_DIR)/LICENSE | rg -q '^b8b9c247a9d453f0669a8c6104b08293b31851a6cdf0f0dcbe08b5bf6dcaf5a1 '
	@sha256sum $(REACT_DIR)/reactions/COPYING | rg -q '^8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903 '

verify-release: verify-inputs
	@$(PYTHON) scripts/verify_repository_boundary.py >/dev/null

verify-canonical-native:
	@test "$$(git ls-files '*marisa_b_native.cpp' | wc -l)" -eq 1
	@test "$$(git ls-files '*marisa_b_native.h' | wc -l)" -eq 1
	@! find $(BUILD_DIR) -type f -name 'portable_marisa_b_native*' -print -quit | rg -q .
	@test "$$(nm -C --defined-only $(BUILD_DIR)/marisa_b_native.o | rg -c ' [TW] marisa_b_native::compute_pre_recon_halo_bias_v1_gaussian\(')" -eq 1
	@test "$$(nm -C --defined-only $(BUILD_DIR)/marisa_b_native.o | rg -c ' [TW] marisa_b_native::compute_post_recon_halo_bias_v1_gaussian\(')" -eq 1

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $(TEST_OBJECTS) $(LINK_FLAGS) -o $@

$(SHELL_BINARY): $(SHELL_OBJECTS)
	$(CXX) $(SHELL_OBJECTS) $(LINK_FLAGS) -o $@

$(MARISA_B_TRIANGLE_BINARY): $(MARISA_B_TRIANGLE_HELPER) src/marisa_b/marisa_b_triangle.cpp $(DM_HEADER) $(BUILD_DIR)/marisa_b_native.o $(MARISA_B_TRIANGLE_REACT_INPUTS)
	$(PYTHON) $(MARISA_B_TRIANGLE_HELPER) \
		--build-dir $(MARISA_B_TRIANGLE_ADAPTER_DIR) \
		--binary $@ \
		--native-object $(BUILD_DIR)/marisa_b_native.o \
		--gsl-prefix "$(GSL_PREFIX)" \
		--cxx "$(CXX)" \
		--max-threads 28 \
		--smoke-help

$(EFT_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_template_algebra.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_MODEL_CONFIG_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_model_config.o $(BUILD_DIR)/eft_v2_model_config.o
	$(CXX) $^ -o $@

$(EFT_BIAS_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_bias_operators.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_DIAGRAM_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_diagrams.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_UV_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_uv_renormalization.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_COUNTERTERM_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_counterterms.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_STOCHASTIC_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_stochastic.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_IR_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_ir_resummation.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_SHELL_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_shell_projector.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_POWER_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_tracer_power.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_R0_DRIVER_BINARY): $(BUILD_DIR)/eft_v2_r0_template_driver.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_R0_POWER_DRIVER_BINARY): $(BUILD_DIR)/eft_v2_r0_power_driver.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_POST_R1_DRIVER_BINARY): $(BUILD_DIR)/eft_v2_post_r1_template_driver.o $(EFT_OBJECTS) $(CORE_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_POST_R1_PNG_DRIVER_BINARY): $(POST_R1_PNG_OBJECTS)
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(EFT_POST_R1_PNG_TEST_BINARY): $(BUILD_DIR)/test_eft_v2_post_r1_png_native.o $(BUILD_DIR)/marisa_b_native.o $(BUILD_DIR)/Common.o $(BUILD_DIR)/PowerSpectrum.o $(BUILD_DIR)/array.o $(BUILD_DIR)/Quadrature.o
	$(CXX) $^ $(LINK_FLAGS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/test_halo_v1.o: tests/test_halo_v1.cpp src/halo_v1/halo_v1.h src/halo_v1/shell_average.h $(DM_HEADER) | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/halo_v1.o: src/halo_v1/halo_v1.cpp src/halo_v1/halo_v1.h $(DM_HEADER) | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/shell_average.o: src/halo_v1/shell_average.cpp src/halo_v1/shell_average.h src/halo_v1/halo_v1.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/shell_checkpoint.o: src/halo_v1/shell_checkpoint.cpp src/halo_v1/shell_average.h src/halo_v1/halo_v1.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_parameter_registry.o: src/eft_v2/parameter_registry.cpp src/eft_v2/parameter_registry.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_model_config.o: src/eft_v2/model_config.cpp src/eft_v2/model_config.h src/eft_v2/bias_operators.h src/eft_v2/parameter_registry.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_template_algebra.o: src/eft_v2/template_algebra.cpp src/eft_v2/template_algebra.h src/eft_v2/parameter_registry.h src/halo_v1/halo_v1.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_kernel_primitives.o: src/eft_v2/kernel_primitives.cpp src/eft_v2/kernel_primitives.h src/halo_v1/halo_v1.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_bias_operators.o: src/eft_v2/bias_operators.cpp src/eft_v2/bias_operators.h src/eft_v2/kernel_primitives.h src/eft_v2/template_algebra.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_field_kernel_provider.o: src/eft_v2/field_kernel_provider.cpp src/eft_v2/field_kernel_provider.h src/eft_v2/bias_operators.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_diagram_assembler.o: src/eft_v2/diagram_assembler.cpp src/eft_v2/diagram_assembler.h src/eft_v2/field_kernel_provider.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_ir_safe_integrands.o: src/eft_v2/ir_safe_integrands.cpp src/eft_v2/ir_safe_integrands.h src/eft_v2/diagram_assembler.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_direct_evaluator.o: src/eft_v2/direct_evaluator.cpp src/eft_v2/direct_evaluator.h src/eft_v2/ir_safe_integrands.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_uv_subtraction.o: src/eft_v2/uv_subtraction.cpp src/eft_v2/uv_subtraction.h src/eft_v2/diagram_assembler.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_fftlog_dr_oracle.o: src/eft_v2/fftlog_dr_oracle.cpp src/eft_v2/fftlog_dr_oracle.h src/eft_v2/direct_evaluator.h src/eft_v2/generated_b222_b321i_tables.h src/eft_v2/generated_b411_table.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_counterterms.o: src/eft_v2/counterterms.cpp src/eft_v2/counterterms.h src/eft_v2/bias_operators.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_stochastic.o: src/eft_v2/stochastic.cpp src/eft_v2/stochastic.h src/eft_v2/counterterms.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_poisson_reconstruction.o: src/eft_v2/poisson_reconstruction.cpp src/eft_v2/poisson_reconstruction.h src/eft_v2/field_kernel_provider.h src/eft_v2/stochastic.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_ir_resummation.o: src/eft_v2/ir_resummation.cpp src/eft_v2/ir_resummation.h src/eft_v2/direct_evaluator.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_lattice_shell_rule.o: src/eft_v2/lattice_shell_rule.cpp src/eft_v2/lattice_shell_rule.h src/halo_v1/shell_average.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_shell_projector.o: src/eft_v2/shell_projector.cpp src/eft_v2/shell_projector.h src/eft_v2/lattice_shell_rule.h src/eft_v2/stochastic.h src/halo_v1/shell_average.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_tracer_power.o: src/eft_v2/tracer_power.cpp src/eft_v2/tracer_power.h src/eft_v2/field_kernel_provider.h src/halo_v1/shell_average.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_template_algebra.o: tests/eft_v2/test_template_algebra.cpp src/eft_v2/template_algebra.h src/eft_v2/parameter_registry.h src/halo_v1/halo_v1.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_model_config.o: tests/eft_v2/test_model_config.cpp src/eft_v2/model_config.h src/eft_v2/bias_operators.h src/eft_v2/parameter_registry.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_bias_operators.o: tests/eft_v2/test_bias_operators.cpp src/eft_v2/bias_operators.h src/eft_v2/kernel_primitives.h src/eft_v2/template_algebra.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_diagrams.o: tests/eft_v2/test_diagrams.cpp src/eft_v2/direct_evaluator.h src/eft_v2/diagram_assembler.h src/eft_v2/field_kernel_provider.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_uv_renormalization.o: tests/eft_v2/test_uv_renormalization.cpp src/eft_v2/fftlog_dr_oracle.h src/eft_v2/direct_evaluator.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_counterterms.o: tests/eft_v2/test_counterterms.cpp src/eft_v2/counterterms.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_stochastic.o: tests/eft_v2/test_stochastic.cpp src/eft_v2/stochastic.h src/eft_v2/poisson_reconstruction.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_ir_resummation.o: tests/eft_v2/test_ir_resummation.cpp src/eft_v2/ir_resummation.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_shell_projector.o: tests/eft_v2/test_shell_projector.cpp src/eft_v2/shell_projector.h src/eft_v2/lattice_shell_rule.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_tracer_power.o: tests/eft_v2/test_tracer_power.cpp src/eft_v2/tracer_power.h src/halo_v1/shell_average.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_r0_template_driver.o: src/eft_v2/r0_template_driver.cpp src/eft_v2/shell_projector.h src/eft_v2/lattice_shell_rule.h src/eft_v2/ir_resummation.h src/eft_v2/fftlog_dr_oracle.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_r0_power_driver.o: src/eft_v2/r0_power_driver.cpp src/eft_v2/tracer_power.h src/eft_v2/ir_resummation.h src/halo_v1/shell_average.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_post_r1_template_driver.o: src/eft_v2/post_r1_template_driver.cpp src/eft_v2/shell_projector.h src/eft_v2/lattice_shell_rule.h | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/eft_v2_post_r1_png_template_driver.o: src/eft_v2/post_r1_png_template_driver.cpp src/eft_v2/lattice_shell_rule.h src/eft_v2/field_kernel_provider.h src/eft_v2/poisson_reconstruction.h $(DM_HEADER) | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_eft_v2_post_r1_png_native.o: tests/eft_v2/test_post_r1_png_native.cpp $(DM_HEADER) | $(BUILD_DIR)
	$(CXX) $(PROJECT_FLAGS) -c $< -o $@

$(BUILD_DIR)/marisa_b_native.o: $(DM_CPP) $(DM_HEADER) | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/Common.o: $(REACT_SRC)/Common.cpp | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/PowerSpectrum.o: $(REACT_SRC)/PowerSpectrum.cpp | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/array.o: $(REACT_SRC)/array.cpp | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/Quadrature.o: $(REACT_SRC)/Quadrature.cpp | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) -c $< -o $@

-include $(DEPFILES)
