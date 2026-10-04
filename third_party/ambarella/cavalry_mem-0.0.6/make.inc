##
## make.inc
##
## History:
##    2018/09/18 - [Tao Wu] Created file
##
## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.	  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
##		 http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

ifeq ($(BUILD_AMBARELLA_CAVALRY_MEM_PACKAGE), y)

LOCAL_PATH      := $(call my-dir)

###
include $(CLEAR_VARS)

LOCAL_SRCS      := $(wildcard $(LOCAL_PATH)/src/*.c)
LOCAL_CFLAGS    := -fvisibility=hidden \
	-I$(LOCAL_PATH)/inc \
	-I$(LOCAL_PATH)/src \
	-I$(AMB_TOPDIR)/include/cavalry

LOCAL_LDFLAGS   :=

CAVALRY_MEM_MODULE = libcavalry_mem.so
CAVALRY_MEM_VER_PREFIX = MEM
CAVALRY_MEM_VERSION_FILE = $(LOCAL_PATH)/src/mem_ver.h
CAVALRY_MEM_SO_VER_MAJOR  := $(shell awk '/define $(CAVALRY_MEM_VER_PREFIX)_LIB_MAJOR/{print $$3}' $(CAVALRY_MEM_VERSION_FILE))
CAVALRY_MEM_SO_VER_MINOR  := $(shell awk '/define $(CAVALRY_MEM_VER_PREFIX)_LIB_MINOR/{print $$3}' $(CAVALRY_MEM_VERSION_FILE))
CAVALRY_MEM_SO_VER_PATCH  := $(shell awk '/define $(CAVALRY_MEM_VER_PREFIX)_LIB_PATCH/{print $$3}' $(CAVALRY_MEM_VERSION_FILE))
CAVALRY_MEM_SO_VER_STRING := $(CAVALRY_MEM_SO_VER_MAJOR).$(CAVALRY_MEM_SO_VER_MINOR).$(CAVALRY_MEM_SO_VER_PATCH)
CAVALRY_MEM_SO_NAME   = $(patsubst %.so,%.so.$(CAVALRY_MEM_SO_VER_MAJOR), $(CAVALRY_MEM_MODULE))
CAVALRY_MEM_SO_TARGET = $(patsubst %.so,%.so.$(CAVALRY_MEM_SO_VER_STRING), $(CAVALRY_MEM_MODULE))
LOCAL_SO_NAME := $(CAVALRY_MEM_SO_NAME)

LOCAL_TARGET    := $(CAVALRY_MEM_MODULE)

include $(BUILD_APP)

.PHONY: $(LOCAL_TARGET)

$(LOCAL_TARGET): $(LOCAL_MODULE)
	@mkdir -p $(FAKEROOT_DIR)/usr/lib/
	@cp -dpRf $< $(FAKEROOT_DIR)/usr/lib/$(CAVALRY_MEM_SO_TARGET)
	@ln -sf $(CAVALRY_MEM_SO_TARGET) $(FAKEROOT_DIR)/usr/lib/$(CAVALRY_MEM_SO_NAME)
	@ln -sf $(CAVALRY_MEM_SO_NAME) $(FAKEROOT_DIR)/usr/lib/$(CAVALRY_MEM_MODULE)
	@echo "Build $@ Done."

$(call add-target-into-build, $(LOCAL_TARGET))

endif
