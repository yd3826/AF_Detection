#!/bin/bash
af_detection_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/af_detection/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/af_detection/main_functions.cc 

af_detection_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/af_detection/main_functions.h 

af_detection_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/af_detect.tflite 

af_detection_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/af_detect_model_data.cc

af_detection_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/af_detect_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/af_detection/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.



# Builds a standalone object recognition binary.
$(eval $(call microlite_test,af_detection,\
$(af_detection_SRCS),$(af_detection_HDRS),$(af_detection_GENERATOR_INPUTS)))

# Add sources and headers generated from $(af_detection_GENERATOR_INPUTS).
af_detection_SRCS += $(af_detection_GENERATED_SRCS)
af_detection_HDRS += $(af_detection_GENERATED_HDRS)

list_af_detection_example_sources:
	@echo $(af_detection_SRCS)

list_af_detection_example_headers:
	@echo $(af_detection_HDRS)
