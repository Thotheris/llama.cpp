//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <map>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <optional>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
// beginning in oneDNN 3.12.3, "Enabled matmul and inner product primitives compatibility with SYCL Graph native recording mode on Intel GPUs."
#if GGML_SYCL_DNNL
#    include "dnnl_version.h"
#    if (DNNL_VERSION_MAJOR > 3 || (DNNL_VERSION_MAJOR == 3 && (DNNL_VERSION_MINOR > 12 || (DNNL_VERSION_MINOR == 12 && DNNL_VERSION_PATCH >= 3))))
#        define GGML_SYCL_GRAPH_DNNL_SUPPORTED
#    endif
#endif
#include <sycl/half_type.hpp>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/dense-dnn.hpp"
#include "ggml-sycl/gate-mix-mm.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/fwht.hpp"
#include "ggml-sycl/fused-gemm.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/mem.hpp"
#include "ggml-sycl/mmvq-id-multi.hpp"
#include "ggml-sycl/mmv-f32.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/dsv4-hc.hpp"
#include "ggml-sycl/lightning-indexer.hpp"
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/block-mix.hpp"
#include "ggml-sycl/rel-bias.hpp"
#include "ggml-sycl/fattn-dpas.hpp"
#include "ggml-sycl/ssm-conv-hist.hpp"
#include "ggml-sycl/route-fuse.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/opt-step.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_dev_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_mkl_fa = 1;
int g_ggml_sycl_fa_dpas = 1;
int g_ggml_sycl_memtrace = 0;
int g_ggml_sycl_memtrace_step = 64;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_fuse_conv_window = 1;
int g_ggml_sycl_enable_esimd = 1;
int g_ggml_sycl_mmid_esimd = 1;
int g_ggml_sycl_mmvq_wide = 1;
int g_ggml_sycl_enable_xmx_esimd = 1;
int g_ggml_sycl_xmx_max_cols = GGML_SYCL_XMX_MAX_COLS_DEFAULT;
int g_ggml_sycl_xmx_glu_max_cols = GGML_SYCL_XMX_GLU_MAX_COLS;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
int g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
int g_ggml_sycl_mmid_sched = GGML_SYCL_MMID_SCHED_DEVICE;
// f32 weights times up to GGML_SYCL_MMV_F32_MAX_COLS columns take the small f32 GEMV (mmv-f32.cpp) instead of oneMKL
int g_ggml_sycl_mmv_f32 = 1;

// MUL_MAT_ID of 2..g_ggml_sycl_mmid_mc_max tokens takes the multi-column MoE GEMV (mmvq-id-multi.cpp), and of up to
// g_ggml_sycl_mmid_mc_xmx_max tokens when its XMX variant takes the expert type; 0 or 1 disables either
int g_ggml_sycl_mmid_mc_max     = 8;
int g_ggml_sycl_mmid_mc_xmx_max = 16;
// the multi-column MoE GEMV takes the XMX kernel for the types and shapes it covers
int g_ggml_sycl_mmid_mc_xmx = 1;
int g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
int g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
static const char * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
static int ggml_sycl_get_env_precision(const char * name, int def) {
    const char * env = getenv(name);
    if (!env) {
        return def;
    }
    std::string mode(env);
    for (char & c : mode) {
        c = (char) std::toupper((unsigned char) c);
    }
    for (int i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        if (mode == ggml_sycl_dynamic_precision_names[i]) {
            return i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    return def;
}
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;
int g_ggml_sycl_bmg_vram_check = 1;
int g_ggml_sycl_enable_host_pinned_mem = 1;
int g_ggml_sycl_host_pinned_mem_2g = 0;
int g_ggml_sycl_upload_staging_slots = 4;
int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
int g_ggml_sycl_enable_sparse_fa = 0;
int g_ggml_sycl_debug_sparse_fa = 0;
int g_ggml_sycl_sparse_fa_margin = 256;
int g_ggml_sycl_graph_eviction_timeout = 10;

#if GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
static bool ggml_sycl_dnnl_detect_optimized_gemm(int device) {
    using dt = dnnl::memory::data_type;

    try {
        const dnnl::memory::dims dims    = { 1, 64, 64 };
        const dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        const auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        const auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        const auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        const std::string impl = pd.impl_info_str();
        if (impl.find("ref") == std::string::npos) {
            return true;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    return false;
}
#endif

static ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    try {
        info.device_count = dpct::dev_mgr::instance().device_count();
    } catch (sycl::exception const &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_count = 0;
    }
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(GGML_SYCL_SUPPORT_XMX) && !defined(GGML_SYCL_FORCE_MMQ)
//     GGML_LOG_INFO("%s: GGML_SYCL_SUPPORT_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: GGML_SYCL_SUPPORT_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#if !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#else
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#if GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#endif
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

#ifdef GGML_SYCL_GRAPH
        info.devices[i].graph_support        = device.has(sycl::aspect::ext_oneapi_limited_graph);
        info.devices[i].graph_update_support = device.has(sycl::aspect::ext_oneapi_graph);
#endif

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            if (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = true;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    return info;
}

ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else if (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        return "Host Forward";
    } else {
        return "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    if (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (int) res);
    }
    return res;
}

static ze_result_t get_zes_init_res() {
    static ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (int) zes_init_res);
    return zes_init_res;
}
#endif

static void initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

static void ggml_check_sycl() try {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    static bool initialized = false;

    if (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
#if GGML_SYCL_DNNL
        ggml_sycl_dense_dnn_init_env();
#endif
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_fa_dpas = ggml_sycl_get_env("GGML_SYCL_FA_DPAS", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_step = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_fuse_conv_window = ggml_sycl_get_env("GGML_SYCL_FUSE_CONV_WINDOW", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_mmid_esimd = ggml_sycl_get_env("GGML_SYCL_MMID_ESIMD", 1);
        g_ggml_sycl_mmvq_wide = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE", 1);
        g_ggml_sycl_enable_xmx_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_XMX_ESIMD", 1);
        g_ggml_sycl_xmx_max_cols = std::max(0, ggml_sycl_get_env("GGML_SYCL_XMX_MAX_COLS", GGML_SYCL_XMX_MAX_COLS_DEFAULT));
        g_ggml_sycl_xmx_glu_max_cols =
            std::clamp(ggml_sycl_get_env("GGML_SYCL_XMX_GLU_MAX_COLS", GGML_SYCL_XMX_GLU_MAX_COLS), 0, GGML_SYCL_XMX_GLU_MAX_COLS);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_graph_eviction_timeout = ggml_sycl_get_env("GGML_SYCL_GRAPH_EVICTION_TIMEOUT", 10);
        g_ggml_sycl_xmx_gather_types = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_TYPES", GGML_SYCL_XMX_GATHER_TYPES_DEFAULT);
        g_ggml_sycl_mmid_sched = ggml_sycl_get_env("GGML_SYCL_MMID_SCHED", GGML_SYCL_MMID_SCHED_DEVICE) & 3;
        g_ggml_sycl_mmid_mc_max = ggml_sycl_get_env("GGML_SYCL_MMID_MC_MAX", 8);
        g_ggml_sycl_mmid_mc_xmx_max = ggml_sycl_get_env("GGML_SYCL_MMID_MC_XMX_MAX", 16);
        g_ggml_sycl_mmid_mc_xmx = ggml_sycl_get_env("GGML_SYCL_MMID_MC_XMX", 1);
        g_ggml_sycl_mmv_f32 = ggml_sycl_get_env("GGML_SYCL_MMV_F32", 1);
        g_ggml_sycl_xmx_gather_shapes = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_SHAPES", GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT);
        g_ggml_sycl_dynamic_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_DEFAULT);
        g_ggml_sycl_dynamic_required_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_REQUIRED_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_F32);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_bmg_vram_check = ggml_sycl_get_env("GGML_SYCL_BMG_VRAM_CHECK", 1);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        g_ggml_sycl_host_pinned_mem_2g =
            ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;
        g_ggml_sycl_upload_staging_slots = std::max(0, ggml_sycl_get_env("GGML_SYCL_UPLOAD_STAGING_SLOTS", 4));

        g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
        g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
        g_ggml_sycl_sparse_fa_margin = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MARGIN", 256);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_XMX)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_XMX: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_XMX: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_DENSE_DNN: %d (from %d columns, %s, budget %d MB)\n", g_ggml_sycl_dense_dnn,
                      g_ggml_sycl_dense_dnn_min_cols, g_ggml_sycl_dense_dnn_bf16 ? "bf16" : "f16", g_ggml_sycl_dense_dnn_max_mb);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_FA_DPAS: %d\n", g_ggml_sycl_fa_dpas);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
        GGML_LOG_INFO("  GGML_SYCL_GRAPH_EVICTION_TIMEOUT: %d\n", g_ggml_sycl_graph_eviction_timeout);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_MMID_SCHED: %d\n", g_ggml_sycl_mmid_sched);
        GGML_LOG_INFO("  GGML_SYCL_MMID_MC_MAX: %d\n", g_ggml_sycl_mmid_mc_max);
        GGML_LOG_INFO("  GGML_SYCL_MMID_MC_XMX_MAX: %d\n", g_ggml_sycl_mmid_mc_xmx_max);
        GGML_LOG_INFO("  GGML_SYCL_MMID_MC_XMX: %d\n", g_ggml_sycl_mmid_mc_xmx);
        GGML_LOG_INFO("  GGML_SYCL_MMV_F32: %d\n", g_ggml_sycl_mmv_f32);
        GGML_LOG_INFO("  GGML_SYCL_MMID_ESIMD: %d\n", g_ggml_sycl_mmid_esimd);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_FUSE_CONV_WINDOW: %d\n", g_ggml_sycl_fuse_conv_window);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#if defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif

        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_XMX_ESIMD: %d\n", g_ggml_sycl_enable_xmx_esimd);
        GGML_LOG_INFO("  GGML_SYCL_XMX_MAX_COLS: %d\n", g_ggml_sycl_xmx_max_cols);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GLU_MAX_COLS: %d\n", g_ggml_sycl_xmx_glu_max_cols);

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_BMG_VRAM_CHECK: %d\n", g_ggml_sycl_bmg_vram_check);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
        GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);

        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA: %d\n", g_ggml_sycl_enable_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_DEBUG: %d\n", g_ggml_sycl_debug_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_MARGIN: %d\n", g_ggml_sycl_sparse_fa_margin);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(GGML_SYCL_SUPPORT_XMX) && !defined(GGML_SYCL_FORCE_MMQ)
        fprintf(stderr, "%s: GGML_SYCL_SUPPORT_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_SYCL_SUPPORT_XMX: no\n", __func__);
#endif
*/
#ifdef GGML_SYCL_GRAPH
        if (g_ggml_sycl_enable_graph) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_limited_graph)) {
                    GGML_LOG_WARN("SYCL device %u does not support SYCL graphs\n", i);
                }
            }
        }

        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    static constexpr size_t staging_slot_size = 8*1024*1024;

    struct host_staging {
        void * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<bool> submitted;
        int next = 0;
    };

    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    void drop_host_staging() {
        for (size_t i = 0; i < staging.submitted.size(); ++i) {
            if (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = false;
            }
        }
        if (staging.data != nullptr) {
            sycl::free(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
#if GGML_SYCL_DNNL
    ggml_sycl_dense_dnn_drop_buffer(buffer);
#endif
    ggml_sycl_gate_mix_mm_drop_buffer(buffer);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    // a compute buffer copy of a host weight (e.g. -ncmoe experts offloaded for prompt processing) is
    // refilled in the normal layout before every run, so reordering it is wasted work
    if (g_ggml_sycl_enable_optimize && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// allocates the pinned upload staging on first use; returns false if staging is disabled or the allocation failed
static bool ggml_sycl_buffer_init_staging(ggml_backend_sycl_buffer_context * ctx) {
    const int n_slots = g_ggml_sycl_upload_staging_slots;
    if (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        if (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, false);
        }
    }
    return ctx->staging.data != nullptr;
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    if (ggml_sycl_buffer_init_staging(ctx)) {
        queue_ptr    stream    = ctx->stream;
        char *       dst       = (char *) tensor->data + offset;
        const char * src       = (const char *) data;
        size_t       remaining = size;
        while (remaining > 0) {
            const size_t chunk = std::min(remaining, ctx->staging_slot_size);
            const int    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (int) ctx->staging.submitted.size();
            if (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            void * stage = (char *) ctx->staging.data + slot * ctx->staging_slot_size;
            memcpy(stage, src, chunk);
            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
            ctx->staging.submitted[slot] = true;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        return;
    }

    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Copy n_copies rows in one call. -sm tensor uploads split weights this way; the generic fallback does one set_tensor per row.
static void ggml_backend_sycl_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                   const void * data, size_t offset, size_t size,
                                                   size_t n_copies, size_t stride_tensor,
                                                   size_t stride_data) try {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    const char * src = (const char *) data;
    ggml_sycl_set_device(ctx->device);

    if (stride_tensor != size || size > ctx->staging_slot_size || !ggml_sycl_buffer_init_staging(ctx)) {
        // dst rows not contiguous, row larger than a staging slot, or no staging: same as the generic fallback
        for (size_t i = 0; i < n_copies; ++i) {
            ggml_backend_sycl_buffer_set_tensor(buffer, tensor, src + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }

    // dst rows are contiguous: pack as many rows as fit into a staging slot and copy each slot with one memcpy
    // a slot is only waited on when it is reused, so packing the next slot overlaps with the previous copies
    queue_ptr    stream        = ctx->stream;
    char *       dst           = (char *) tensor->data + offset;
    const size_t rows_per_slot = ctx->staging_slot_size / size;
    for (size_t i0 = 0; i0 < n_copies; i0 += rows_per_slot) {
        const size_t n    = std::min(rows_per_slot, n_copies - i0);
        const int    slot = ctx->staging.next;
        ctx->staging.next = (ctx->staging.next + 1) % (int) ctx->staging.submitted.size();
        if (ctx->staging.submitted[slot]) {
            ctx->staging.events[slot].wait_and_throw();
        }
        char * stage = (char *) ctx->staging.data + slot * ctx->staging_slot_size;
        for (size_t i = 0; i < n; ++i) {
            memcpy(stage + i * size, src + (i0 + i) * stride_data, size);
        }
        ctx->staging.events[slot]    = stream->memcpy(dst + i0 * size, stage, n * size);
        ctx->staging.submitted[slot] = true;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

static void memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                         const void *ptr_src, size_t size) {
    char *host_buf = (char *)malloc(size);
    if (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    if(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } else {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_sycl_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool is_bmg_g31_arch(int device) {
    return ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    int device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    if (g_ggml_sycl_host_pinned_mem_2g) {
        return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } else {
        if(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        return max_alloc_size;

    }
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    if (ctx->streams.empty()) {
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

static int ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    return static_cast<const ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
static void * ggml_backend_sycl_host_malloc(int device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    void * ptr = nullptr;
    try {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } catch (...) {
        ptr = nullptr;
    }
    if (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    return ptr;
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context == nullptr) {
        return;
    }
    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(buffer->context, q)));
    } else {
        free_aligned_mem_host((void *) buffer->context);
    }
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        if (g_ggml_sycl_host_pinned_mem_2g) {
            return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } else {
            return max_alloc_size;
        }
    } else {
        return SIZE_MAX;
    }
}

static ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    static std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_count());
        for (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        return bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (int) buffer_types_host.size());

    return &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    return ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_SUPPORT_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            if (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_size);

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;
    int                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            void * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        if (b.ptr != nullptr) {
            if (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                void * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                return ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        void * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    if (ncols <= WARP_SIZE && nrows >= 64) {
        // short rows (the 4-tap windows of a convolution, the 8 expert weights of a token): one work-item per row
        // instead of a sub-group, the sum the tree of the sub-group reduction below (lane i holds x[i], the
        // lanes past the row 0, halves added from WARP_SIZE/2 down), so the bits are the same
        constexpr int wg = 256;
        stream->parallel_for(sycl::nd_range<1>(GGML_PAD((size_t) nrows, wg), wg), [=](sycl::nd_item<1> it) {
            const int row = (int) it.get_global_id(0);
            if (row >= nrows) {
                return;
            }
            const float * xr = x + (int64_t) row*ncols;
            float v[WARP_SIZE];
#pragma unroll
            for (int i = 0; i < WARP_SIZE; ++i) {
                v[i] = i < ncols ? xr[i] : 0.0f;
            }
#pragma unroll
            for (int s = WARP_SIZE/2; s > 0; s >>= 1) {
#pragma unroll
                for (int i = 0; i < s; ++i) {
                    v[i] = v[i] + v[i + s];
                }
            }
            dst[row] = v[0];
        });
        return;
    }
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

// the bitonic network of k_argsort_f32_i32 run by one sub-group per row: position p = 32 e + lane is element e of
// the lane, partners at distance j < 32 are exchanged by shuffles, the others are registers of the same lane. The
// compare-and-swap of every pair is the one of k_argsort_f32_i32, so the result is the same, ties included.
// The steps (k, j) are template arguments, so the element indices are constants and the arrays stay in registers.
template <ggml_sort_order order>
static inline bool argsort_sg_swap(float va, int ia, float vb, int ib, bool up, int ncols) {
    // true when the element at the lower position (a) and the one at the higher (b) swap
    if (up) {
        return ia >= ncols || (ib < ncols && (order == GGML_SORT_ORDER_ASC ? va > vb : va < vb));
    }
    return ib >= ncols || (ia < ncols && (order == GGML_SORT_ORDER_ASC ? va < vb : va > vb));
}

// one element per work-item, NP work-items in sub-groups of 32: partners within the sub-group by shuffles, the
// others (J >= 32) through SLM, double-buffered so that each such step needs one barrier
template <int NP, ggml_sort_order order, int K, int J>
static inline void argsort_wg_step(float & v, int & id, int p, int ncols, const sycl::nd_item<1> & it,
                                   float * sv, int * si, int buf) {
    const bool is_lo = (p & J) == 0;
    float      ov;
    int        oi;
    if constexpr (J >= 32) {
        sv[buf * NP + p] = v;
        si[buf * NP + p] = id;
        it.barrier(sycl::access::fence_space::local_space);
        ov = sv[buf * NP + (p ^ J)];
        oi = si[buf * NP + (p ^ J)];
        buf ^= 1;
    } else {
        ov = sycl::permute_group_by_xor(it.get_sub_group(), v, J);
        oi = sycl::permute_group_by_xor(it.get_sub_group(), id, J);
    }
    const bool up = ((p & ~J) & K) == 0;
    const bool sw = is_lo ? argsort_sg_swap<order>(v, id, ov, oi, up, ncols) : argsort_sg_swap<order>(ov, oi, v, id, up, ncols);
    v  = sw ? ov : v;
    id = sw ? oi : id;
    if constexpr (J > 1) {
        argsort_wg_step<NP, order, K, J / 2>(v, id, p, ncols, it, sv, si, buf);
    } else if constexpr (K < NP) {
        argsort_wg_step<NP, order, 2 * K, K>(v, id, p, ncols, it, sv, si, buf);
    }
}

template <int NP>
static void argsort_f32_i32_sg_sycl(const float * x, int * dst, const int ncols, const int nrows,
                                    ggml_sort_order order, queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(2 * NP), cgh);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(2 * NP), cgh);
        const sycl::nd_range<1>        ndr(sycl::range<1>((size_t) nrows * NP), sycl::range<1>(NP));
        auto kern = [=](sycl::nd_item<1> it, auto ord) {
            constexpr ggml_sort_order o = decltype(ord)::value;
            const int     p   = it.get_local_id(0);
            const int64_t row = it.get_group(0);
            float         v   = p < ncols ? x[row * ncols + p] : 0.0f;
            int           id  = p;
            argsort_wg_step<NP, o, 2, 1>(v, id, p, ncols, it, &sv[0], &si[0], 0);
            if (p < ncols) {
                dst[row * ncols + p] = id;
            }
        };
        if (order == GGML_SORT_ORDER_ASC) {
            cgh.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                kern(it, std::integral_constant<ggml_sort_order, GGML_SORT_ORDER_ASC>{});
            });
        } else {
            cgh.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                kern(it, std::integral_constant<ggml_sort_order, GGML_SORT_ORDER_DESC>{});
            });
        }
    });
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // rows of 33 .. 256 columns (the MoE routers): the same network in one sub-group per row, no barriers
    if ((order == GGML_SORT_ORDER_ASC || order == GGML_SORT_ORDER_DESC) && ncols > 32 && ncols <= 256) {
        switch (next_power_of_2(ncols)) {
            case 64:  argsort_f32_i32_sg_sycl<64>(x, dst, ncols, nrows, order, stream); return;
            case 128: argsort_f32_i32_sg_sycl<128>(x, dst, ncols, nrows, order, stream); return;
            case 256: argsort_f32_i32_sg_sycl<256>(x, dst, ncols, nrows, order, stream); return;
            default: break;
        }
    }
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

// Scan and block merge, shared by every launch shape below so a partitioned row uses the
// same insertion order as an unpartitioned one.
//
// src_map  != nullptr: report src_map[col] instead of col, so a merge pass can carry the
//                      original column index through.
// out_vals != nullptr: also emit the k winning values, for a later merge pass.
// swap01:              emit in the output order the single-pass path uses.
static void top_k_scan_merge_f32(
    const float *   src_vals,
    const int32_t * src_map,
    const int       begin,
    const int       end,
    const int       k,
    const int       block_size,
    float *         shared_vals,
    int *           shared_idx,
    float *         out_vals,
    int32_t *       out_idx,
    const bool      swap01,
    const sycl::nd_item<1> & item_ct1
) {
    const int tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    float * lv = shared_vals + tid;
    int * li = shared_idx + tid;

    for (int i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    float kth = -FLT_MAX;

    for (int col = begin + tid; col < end; col += block_size) {
        float val = src_vals[col];

        if (val > kth) {
            int pos = k - 1;
            while (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            for (int i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tid != 0) {
        return;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    float * fv = shared_vals + (size_t) k * block_size;
    int * fi = shared_idx + (size_t) k * block_size;

    for (int i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    float fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    for (int t = 0; t < block_size; t++) {
        for (int i = 0; i < k; i++) {
            float val = shared_vals[i * block_size + t];

            if (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                break;
            }

            int idx = shared_idx[i * block_size + t];

            int pos = k - 1;
            while (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            for (int j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    if (out_vals) {
        for (int i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    for (int i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    if (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

static void top_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    // A row is scanned by exactly one work-group, so a vocabulary-sized row leaves the
    // rest of the device idle. What the scan is short of is memory requests in flight,
    // not bandwidth or per-request latency, so lanes in flight is the lever: split the
    // row across independent work-groups, have each emit its partition's top-k, and
    // merge those nsplit*k candidates in a second launch.
    //
    // split_block trades parallelism against SLM residency. Its cost is
    // (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128
    // lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit
    // if the supported k ever grows.
    constexpr int split_block = 128;
    constexpr int max_splits  = 128;
    constexpr int min_cols    = 8192;

    int nsplit = 1;
    if (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        const int64_t want = ncols / split_block;
        nsplit = (int) (want > max_splits ? max_splits : want);
    }

    if (nsplit > 1) {
        const int nchunk = (int) ((ncols + nsplit - 1) / nsplit);
        const size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<float>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        float *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        const sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int grp = item_ct1.get_group(0);
                    const int row = grp / nsplit;
                    const int part = grp % nsplit;

                    const int begin = part * nchunk;
                    int end = begin + nchunk;
                    if (end > (int) ncols) {
                        end = (int) ncols;
                    }

                    top_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, false, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int row = item_ct1.get_group(0);
                    const size_t off = (size_t) row * nsplit * k;

                    top_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
                });
        });

        return;
    }

    // Short rows: one work-item per column and k work-group argmax rounds.
    // Ties take the lower column.
    const int wg = std::max(WARP_SIZE, next_power_of_2((int) ncols));
    if (wg <= ggml_sycl_info().max_work_group_sizes[ctx.device]) {
        main_stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nrows * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item_ct1) {
                const int64_t row = item_ct1.get_group(0);
                const int     tid = item_ct1.get_local_id(0);
                const auto    grp = item_ct1.get_group();

                // NaN is never preferred, and every round still has a winner since k <= ncols
                const float x     = tid < ncols ? src[row * ncols + tid] : -INFINITY;
                const float val   = sycl::isnan(x) ? -INFINITY : x;
                bool        taken = tid >= ncols;
                for (int i = 0; i < k; i++) {
                    const float cand   = taken ? -INFINITY : val;
                    const float best   = sycl::reduce_over_group(grp, cand, sycl::maximum<float>());
                    const int   winner = sycl::reduce_over_group(grp, !taken && cand == best ? tid : INT_MAX,
                                                                 sycl::minimum<int>());
                    if (tid == winner) {
                        dst_indices[row * k + i] = tid;
                        taken = true;
                    }
                }
            });
        return;
    }

    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);

                if (row >= nrows) return;

                top_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (int) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
            });
    });
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                if (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#if GGML_SYCL_DNNL
// F32 GEMMs below this size (row_diff * src1_ncols * ne10) call oneMKL directly instead of oneDNN
static constexpr int64_t SYCL_F32_DNNL_MIN_GEMM_FLOPS = 256 * 256 * 256;
#endif

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    bool src0_reordered = false;
    if (const auto * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        return;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && ggml_prec_get_acc(dst) == GGML_PREC_UNDEFINED) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#if GGML_SYCL_DNNL
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < SYCL_F32_DNNL_MIN_GEMM_FLOPS;
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    if (k <= SYCL_TOP_K_SCAN_MERGE_MAX_K) {
        top_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } else {
        ggml_sycl_top_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, [[maybe_unused]] int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (char *) src0_extra->data_device[i] : (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = (float *) ((char *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            const char * src1_ddf_i_source = (const char *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
#if GGML_SYCL_DNNL
    const size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#if GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            if (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_reorder_esimd(enum ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        // async_malloc throws on out-of-memory; return nullptr like the sync path so callers can fall back.
        void * ptr = nullptr;
        try {
            ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("%s: async_malloc of %zu bytes failed: %s\n", __func__, size, e.what());
            ggml_sycl_memtrace_fail(GGML_SYCL_MEM_ASYNC, size);
            return nullptr;
        }
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        return ptr;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

// Reorder a 2D tensor of quantized blocks in place into [head of every block][tail of every block] without a
// full-size temporary copy. The head is the contiguous `head_bytes` at `head_offset` inside each block. Heads only
// move towards the front, so they are packed in chunks that never overwrite unread blocks. The tail overlaps
// blocks that are not packed yet, so `write_tail(x, ib, tail, nblocks)` first writes every block's remaining fields
// (in their final layout, relative to the start of the tail) into a side buffer that is copied to the end afterwards.
template <typename block_t, typename tail_fn_t>
static bool reorder_qw_planar(const char * name, uint8_t * data_device, size_t size, size_t head_offset,
                              size_t head_bytes, tail_fn_t write_tail, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_t) == 0);
    GGML_ASSERT(head_offset + head_bytes <= sizeof(block_t));

    const int64_t nblocks    = size / sizeof(block_t);
    const size_t  tail_bytes = (sizeof(block_t) - head_bytes) * nblocks;

    sycl_reorder_temp_buffer tail_tmp(stream, tail_bytes);
    if (!tail_tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", name, tail_bytes);
        return false;
    }
    uint8_t * tail_buf = static_cast<uint8_t *>(tail_tmp.ptr);

    const size_t  max_chunk_bytes = 32u << 20;
    const int64_t chunk_blocks    = std::max<int64_t>(1, std::min<int64_t>(nblocks, max_chunk_bytes / sizeof(block_t)));
    sycl_reorder_temp_buffer tmp(stream, chunk_blocks * sizeof(block_t));
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", name,
                      (size_t) chunk_blocks * sizeof(block_t));
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event event = stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        write_tail(((const block_t *) data_device)[ib], ib, tail_buf, nblocks);
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        event.wait_and_throw();
    }

    for (int64_t b0 = 0; b0 < nblocks; b0 += chunk_blocks) {
        const int64_t n_chunk = std::min(chunk_blocks, nblocks - b0);

        SYCL_CHECK(CHECK_TRY_ERROR(event = stream->memcpy(tmp_buf, data_device + b0 * sizeof(block_t),
                                                          n_chunk * sizeof(block_t))));
        if (!g_ggml_sycl_use_async_mem_op) {
            event.wait();
        }

        uint8_t * head_ptr = data_device + b0 * head_bytes;
        event = stream->parallel_for(n_chunk, [=](auto i) {
            const int64_t   ib  = i;
            const uint8_t * src = tmp_buf + ib * sizeof(block_t) + head_offset;
            for (size_t j = 0; j < head_bytes; ++j) {
                head_ptr[ib * head_bytes + j] = src[j];
            }
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            event.wait_and_throw();
        }
    }

    SYCL_CHECK(CHECK_TRY_ERROR(event = stream->memcpy(data_device + head_bytes * nblocks, tail_buf, tail_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        event.wait();
    }
    return true;
}

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT((size_t) ncols * nrows / 2 == size / sizeof(block_q4_0) * (QK4_0 / 2));
    return reorder_qw_planar<block_q4_0>(__func__, data_device, size, offsetof(block_q4_0, qs), QK4_0 / 2,
        [](const block_q4_0 & x, int64_t ib, uint8_t * tail, int64_t) {
            ((sycl::half *) tail)[ib] = x.d;
        }, stream);
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT((size_t) ncols * nrows == size / sizeof(block_q8_0) * QK8_0);
    return reorder_qw_planar<block_q8_0>(__func__, data_device, size, offsetof(block_q8_0, qs), QK8_0,
        [](const block_q8_0 & x, int64_t ib, uint8_t * tail, int64_t) {
            ((sycl::half *) tail)[ib] = x.d;
        }, stream);
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    return reorder_qw_planar<block_q4_K>(__func__, data_device, size, offsetof(block_q4_K, qs), QK_K / 2,
        [](const block_q4_K & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * scales = tail;
            for (int j = 0; j < K_SCALE_SIZE; ++j) {
                scales[ib * K_SCALE_SIZE + j] = x.scales[j];
            }
            ((sycl::half2 *) (scales + K_SCALE_SIZE * nblocks))[ib] = x.dm;
        }, stream);
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout, a few experts at a time.
static bool reorder_qw_q8_0_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q8_0) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_q8_0));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int          gb   = gb_;
            const int          e    = gb / blocks_per_expert;
            const int          ib   = gb % blocks_per_expert;
            const block_q8_0 * x    = (const block_q8_0 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *          base = chunk + (size_t) e * expert_bytes;

            auto * qs_ptr = base;
            auto * d_ptr  = (sycl::half *) (qs_ptr + (size_t) blocks_per_expert * QK8_0);

            for (int j = 0; j < QK8_0; ++j) {
                *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            d_ptr[ib] = x[ib].d;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
// Like the MXFP4 version below, this goes a few experts at a time through a small temporary:
// a whole-tensor temporary (GBs for large MoE models) can exceed the VRAM left on a nearly full card.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int          gb   = gb_;
            const int          e    = gb / blocks_per_expert;
            const int          ib   = gb % blocks_per_expert;
            const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *          base = chunk + (size_t) e * expert_bytes;

            auto * qs_ptr     = base;
            auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
            auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

            for (int j = 0; j < QK_K / 2; ++j) {
                qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
            }
            for (int j = 0; j < K_SCALE_SIZE; ++j) {
                scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
            }
            dm_ptr[ib] = x[ib].dm;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm], a few experts at a time (see Q4_K above).
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int          gb   = gb_;
            const int          e    = gb / blocks_per_expert;
            const int          ib   = gb % blocks_per_expert;
            const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *          base = chunk + (size_t) e * expert_bytes;

            auto * qs_ptr     = base;
            auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
            auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
            auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

            for (int j = 0; j < QK_K / 2; ++j) {
                qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
            }
            for (int j = 0; j < QK_K / 8; ++j) {
                qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
            }
            for (int j = 0; j < K_SCALE_SIZE; ++j) {
                scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
            }
            dm_ptr[ib] = x[ib].dm;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d], a few experts at a time (see Q4_K above).
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int          gb   = gb_;
            const int          e    = gb / blocks_per_expert;
            const int          ib   = gb % blocks_per_expert;
            const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *          base = chunk + (size_t) e * expert_bytes;

            auto * ql_ptr     = base;
            auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
            auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
            auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

            for (int j = 0; j < QK_K / 2; ++j) {
                ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
            }
            for (int j = 0; j < QK_K / 4; ++j) {
                qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
            }
            d_ptr[ib] = x[ib].d;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    return true;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
static bool reorder_qw_mxfp4_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_mxfp4) == 0);
    const int     blocks_per_expert = (int) (expert_bytes / sizeof(block_mxfp4));
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        if (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        const int total_blocks = blocks_per_expert * (int) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            const int           gb   = gb_;
            const int           e    = gb / blocks_per_expert;
            const int           ib   = gb % blocks_per_expert;
            const block_mxfp4 * x    = (const block_mxfp4 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *           base = chunk + (size_t) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) blocks_per_expert;

            for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        if (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    return true;
}

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    return reorder_qw_planar<block_q2_K>(__func__, data_device, size, offsetof(block_q2_K, qs), QK_K / 4,
        [](const block_q2_K & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * scales = tail;
            for (int j = 0; j < QK_K / 16; ++j) {
                scales[ib * (QK_K / 16) + j] = x.scales[j];
            }
            ((sycl::half2 *) (scales + (QK_K / 16) * nblocks))[ib] = x.dm;
        }, stream);
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    return reorder_qw_planar<block_q3_K>(__func__, data_device, size, offsetof(block_q3_K, qs), QK_K / 4,
        [](const block_q3_K & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * hmask  = tail;
            uint8_t * scales = hmask + (QK_K / 8) * nblocks;
            for (int j = 0; j < QK_K / 8; ++j) {
                hmask[ib * (QK_K / 8) + j] = x.hmask[j];
            }
            for (int j = 0; j < 12; ++j) {
                scales[ib * 12 + j] = x.scales[j];
            }
            ((sycl::half *) (scales + 12 * nblocks))[ib] = x.d;
        }, stream);
}

// Reorder each Q3_K expert slice on its own.
static bool reorder_qw_q3_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    for (int64_t e = 0; e < n_expert; ++e) {
        if (!reorder_qw_q3_k(data_device + e * expert_bytes, expert_bytes, 0, stream)) {
            return false;
        }
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    return reorder_qw_planar<block_q5_K>(__func__, data_device, size, offsetof(block_q5_K, qs), QK_K / 2,
        [](const block_q5_K & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * qh     = tail;
            uint8_t * scales = qh + (QK_K / 8) * nblocks;
            for (int j = 0; j < QK_K / 8; ++j) {
                qh[ib * (QK_K / 8) + j] = x.qh[j];
            }
            for (int j = 0; j < K_SCALE_SIZE; ++j) {
                scales[ib * K_SCALE_SIZE + j] = x.scales[j];
            }
            ((sycl::half2 *) (scales + K_SCALE_SIZE * nblocks))[ib] = x.dm;
        }, stream);
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(offset == 0);
    return reorder_qw_planar<block_q6_K>(__func__, data_device, size, offsetof(block_q6_K, ql), QK_K / 2,
        [](const block_q6_K & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * qh     = tail;
            uint8_t * scales = qh + (QK_K / 4) * nblocks;
            for (int j = 0; j < QK_K / 4; ++j) {
                qh[ib * (QK_K / 4) + j] = x.qh[j];
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                scales[ib * (QK_K / 16) + j] = x.scales[j];
            }
            ((sycl::half *) (scales + (QK_K / 16) * nblocks))[ib] = x.d;
        }, stream);
}

static bool reorder_qw_iq3_s(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    return reorder_qw_planar<block_iq3_s>(__func__, data_device, size, offsetof(block_iq3_s, qs), QK_K / 4,
        [](const block_iq3_s & x, int64_t ib, uint8_t * tail, int64_t nblocks) {
            uint8_t * qh       = tail;
            uint8_t * signs    = qh + (QK_K / 32) * nblocks;
            uint8_t * metadata = signs + (QK_K / 8) * nblocks + ib * (sizeof(ggml_half) + IQ3S_N_SCALE);
            for (int j = 0; j < QK_K / 32; ++j) {
                qh[ib * (QK_K / 32) + j] = x.qh[j];
            }
            for (int j = 0; j < QK_K / 8; ++j) {
                signs[ib * (QK_K / 8) + j] = x.signs[j];
            }
            *reinterpret_cast<ggml_half *>(metadata) = x.d;
            for (int j = 0; j < IQ3S_N_SCALE; ++j) {
                metadata[sizeof(ggml_half) + j] = x.scales[j];
            }
        }, stream);
}

static bool reorder_qw_iq3_xxs(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    return reorder_qw_planar<block_iq3_xxs>(__func__, data_device, size, offsetof(block_iq3_xxs, qs), 3 * QK_K / 8,
        [](const block_iq3_xxs & x, int64_t ib, uint8_t * tail, int64_t) {
            reinterpret_cast<ggml_half *>(tail)[ib] = x.d;
        }, stream);
}

// IQ4_NL / IQ4_XS: the qs go to the front, everything before them (d, plus the IQ4_XS scales) is the block's
// record in the tail
template <typename block_t>
static bool reorder_qw_iq4(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    constexpr size_t record_size = offsetof(block_t, qs);
    return reorder_qw_planar<block_t>(__func__, data_device, size, record_size, sizeof(block_t) - record_size,
        [](const block_t & x, int64_t ib, uint8_t * tail, int64_t) {
            const uint8_t * record = reinterpret_cast<const uint8_t *>(&x);
            for (size_t j = 0; j < record_size; ++j) {
                tail[ib * record_size + j] = record[j];
            }
        }, stream);
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                return reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q3_K:
                return reorder_qw_q3_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_MXFP4:
                return reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        case GGML_TYPE_IQ3_XXS:
            return reorder_qw_iq3_xxs(data_device, size, stream);
        case GGML_TYPE_IQ3_S:
            return reorder_qw_iq3_s(data_device, size, stream);
        case GGML_TYPE_IQ4_NL:
            return reorder_qw_iq4<block_iq4_nl>(data_device, size, stream);
        case GGML_TYPE_IQ4_XS:
            return reorder_qw_iq4<block_iq4_xs>(data_device, size, stream);
        default:
            return false;
    }
}

// A view of a 2D weight that has the reorder (SoA) layout: the view has no extra of its own, so every kernel would
// read the planar bytes at its offset as the standard block layout (the Inkling2 attn_in row views past 128 tokens
// after a decode step reordered the stacked weight: NaN logits). reorder_qw_planar packs the head field of every
// block of the whole tensor first and the remaining fields after it, so a row range is not contiguous in that
// layout. The rows of such a view are rebuilt here in the standard layout, in a pool buffer.
template <typename block_t, typename tail_fn_t>
static void unreorder_blocks_planar(const uint8_t * parent, int64_t nblocks, int64_t b0, int64_t n, block_t * out,
                                    size_t head_offset, size_t head_bytes, tail_fn_t read_tail, dpct::queue_ptr stream) {
    const uint8_t * tail = parent + head_bytes * nblocks;
    stream->parallel_for(n, [=](auto i) {
        const int64_t ib = b0 + (int64_t) i;
        block_t & x = out[(int64_t) i];
        uint8_t * dst_head = (uint8_t *) &x + head_offset;
        for (size_t j = 0; j < head_bytes; ++j) {
            dst_head[j] = parent[ib * head_bytes + j];
        }
        read_tail(x, ib, tail, nblocks);
    });
}

static bool unreorder_blocks(ggml_type type, const uint8_t * parent, size_t parent_bytes, int64_t b0, int64_t n,
                             void * out, dpct::queue_ptr stream) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            unreorder_blocks_planar<block_q4_0>(parent, parent_bytes / sizeof(block_q4_0), b0, n, (block_q4_0 *) out,
                offsetof(block_q4_0, qs), QK4_0 / 2,
                [](block_q4_0 & x, int64_t ib, const uint8_t * tail, int64_t) { x.d = ((const sycl::half *) tail)[ib]; },
                stream);
            return true;
        case GGML_TYPE_Q8_0:
            unreorder_blocks_planar<block_q8_0>(parent, parent_bytes / sizeof(block_q8_0), b0, n, (block_q8_0 *) out,
                offsetof(block_q8_0, qs), QK8_0,
                [](block_q8_0 & x, int64_t ib, const uint8_t * tail, int64_t) { x.d = ((const sycl::half *) tail)[ib]; },
                stream);
            return true;
        case GGML_TYPE_Q2_K:
            unreorder_blocks_planar<block_q2_K>(parent, parent_bytes / sizeof(block_q2_K), b0, n, (block_q2_K *) out,
                offsetof(block_q2_K, qs), QK_K / 4,
                [](block_q2_K & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * scales = tail;
                    for (int j = 0; j < QK_K / 16; ++j) {
                        x.scales[j] = scales[ib * (QK_K / 16) + j];
                    }
                    x.dm = ((const sycl::half2 *) (scales + (QK_K / 16) * nblocks))[ib];
                }, stream);
            return true;
        case GGML_TYPE_Q3_K:
            unreorder_blocks_planar<block_q3_K>(parent, parent_bytes / sizeof(block_q3_K), b0, n, (block_q3_K *) out,
                offsetof(block_q3_K, qs), QK_K / 4,
                [](block_q3_K & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * hmask  = tail;
                    const uint8_t * scales = hmask + (QK_K / 8) * nblocks;
                    for (int j = 0; j < QK_K / 8; ++j) {
                        x.hmask[j] = hmask[ib * (QK_K / 8) + j];
                    }
                    for (int j = 0; j < 12; ++j) {
                        x.scales[j] = scales[ib * 12 + j];
                    }
                    x.d = ((const sycl::half *) (scales + 12 * nblocks))[ib];
                }, stream);
            return true;
        case GGML_TYPE_Q4_K:
            unreorder_blocks_planar<block_q4_K>(parent, parent_bytes / sizeof(block_q4_K), b0, n, (block_q4_K *) out,
                offsetof(block_q4_K, qs), QK_K / 2,
                [](block_q4_K & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * scales = tail;
                    for (int j = 0; j < K_SCALE_SIZE; ++j) {
                        x.scales[j] = scales[ib * K_SCALE_SIZE + j];
                    }
                    x.dm = ((const sycl::half2 *) (scales + K_SCALE_SIZE * nblocks))[ib];
                }, stream);
            return true;
        case GGML_TYPE_Q5_K:
            unreorder_blocks_planar<block_q5_K>(parent, parent_bytes / sizeof(block_q5_K), b0, n, (block_q5_K *) out,
                offsetof(block_q5_K, qs), QK_K / 2,
                [](block_q5_K & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * qh     = tail;
                    const uint8_t * scales = qh + (QK_K / 8) * nblocks;
                    for (int j = 0; j < QK_K / 8; ++j) {
                        x.qh[j] = qh[ib * (QK_K / 8) + j];
                    }
                    for (int j = 0; j < K_SCALE_SIZE; ++j) {
                        x.scales[j] = scales[ib * K_SCALE_SIZE + j];
                    }
                    x.dm = ((const sycl::half2 *) (scales + K_SCALE_SIZE * nblocks))[ib];
                }, stream);
            return true;
        case GGML_TYPE_Q6_K:
            unreorder_blocks_planar<block_q6_K>(parent, parent_bytes / sizeof(block_q6_K), b0, n, (block_q6_K *) out,
                offsetof(block_q6_K, ql), QK_K / 2,
                [](block_q6_K & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * qh     = tail;
                    const uint8_t * scales = qh + (QK_K / 4) * nblocks;
                    for (int j = 0; j < QK_K / 4; ++j) {
                        x.qh[j] = qh[ib * (QK_K / 4) + j];
                    }
                    for (int j = 0; j < QK_K / 16; ++j) {
                        x.scales[j] = scales[ib * (QK_K / 16) + j];
                    }
                    x.d = ((const sycl::half *) (scales + (QK_K / 16) * nblocks))[ib];
                }, stream);
            return true;
        case GGML_TYPE_IQ3_S:
            unreorder_blocks_planar<block_iq3_s>(parent, parent_bytes / sizeof(block_iq3_s), b0, n, (block_iq3_s *) out,
                offsetof(block_iq3_s, qs), QK_K / 4,
                [](block_iq3_s & x, int64_t ib, const uint8_t * tail, int64_t nblocks) {
                    const uint8_t * qh       = tail;
                    const uint8_t * signs    = qh + (QK_K / 32) * nblocks;
                    const uint8_t * metadata = signs + (QK_K / 8) * nblocks + ib * (sizeof(ggml_half) + IQ3S_N_SCALE);
                    for (int j = 0; j < QK_K / 32; ++j) {
                        x.qh[j] = qh[ib * (QK_K / 32) + j];
                    }
                    for (int j = 0; j < QK_K / 8; ++j) {
                        x.signs[j] = signs[ib * (QK_K / 8) + j];
                    }
                    x.d = *reinterpret_cast<const ggml_half *>(metadata);
                    for (int j = 0; j < IQ3S_N_SCALE; ++j) {
                        x.scales[j] = metadata[sizeof(ggml_half) + j];
                    }
                }, stream);
            return true;
        case GGML_TYPE_IQ3_XXS:
            unreorder_blocks_planar<block_iq3_xxs>(parent, parent_bytes / sizeof(block_iq3_xxs), b0, n,
                (block_iq3_xxs *) out, offsetof(block_iq3_xxs, qs), 3 * QK_K / 8,
                [](block_iq3_xxs & x, int64_t ib, const uint8_t * tail, int64_t) {
                    x.d = reinterpret_cast<const ggml_half *>(tail)[ib];
                }, stream);
            return true;
        default:
            return false;
    }
}

// the reordered 2D parent of a row-range view src0, else nullptr
static const ggml_tensor * ggml_sycl_reordered_view_parent(const ggml_tensor * src0) {
    const ggml_tensor * p = src0->view_src;
    if (!p || (src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder)) {
        return nullptr;
    }
    const auto * pex = (const ggml_tensor_extra_gpu *) p->extra;
    if (!pex || !pex->optimized_feature.reorder) {
        return nullptr;
    }
    // reorder_qw() reorders 3D tensors per expert, so only 2D parents have the whole-tensor planar layout
    GGML_ASSERT(p->ne[2] == 1 && p->ne[3] == 1 && "view of a reordered 3D weight");
    GGML_ASSERT(src0->type == p->type && src0->ne[0] == p->ne[0] && src0->nb[1] == p->nb[1] &&
                ggml_is_contiguous(src0) && src0->view_offs % p->nb[1] == 0 && "non row-range view of a reordered weight");
    return p;
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        return;
    }
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q3_K && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        return;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    if (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        return;
    }
    // The grouped GEMM loads reordered Q3_K qs and hmask as 16-byte vectors.
    if (src0->type == GGML_TYPE_Q3_K && ((src0->nb[2] / sizeof(block_q3_K)) % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

// reordered weights of the types in ggml_sycl_xmx_supports_type() take the XMX kernel for wider batches
static bool can_use_xmx_batch(int device, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return ggml_sycl_xmx_enabled(device) && ggml_sycl_xmx_supports_type(src0->type) && extra &&
           extra->optimized_feature.reorder && src1->ne[1] <= g_ggml_sycl_xmx_max_cols && src1->ne[2] == 1 &&
           src1->ne[3] == 1;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    return false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
}

static bool can_use_mul_mat_vec_q(int device, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE || can_use_xmx_batch(device, src0, src1));
}

static bool can_use_mul_mat_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    if (!ggml_sycl_supports_mmq(src0->type) || src0->type == GGML_TYPE_IQ2_XXS) {
        return false;
    }
#if defined(GGML_SYCL_SUPPORT_XMX) && !defined(GGML_SYCL_FORCE_MMQ)
    if (src1->ne[1] > MMQ_MAX_BATCH_SIZE) {
        return false;
    }
#endif // GGML_SYCL_SUPPORT_XMX && !GGML_SYCL_FORCE_MMQ
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

#if GGML_SYCL_DNNL
    // a weight times a wide batch: oneDNN against a persistent f16 copy of the weight (dense-dnn.hpp). Taken before
    // the row-view rebuild below: the weight copy is made from the whole reordered base, so a view needs no rebuild.
    if (!ggml_backend_buffer_is_sycl_split(src0->buffer) && dst->src[0] == src0 && dst->src[1] == src1 &&
        !(dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) &&
        ggml_sycl_dense_dnn_eligible(ctx.device, dst) && ggml_sycl_dense_dnn_mul_mat(ctx, src0, src1, dst)) {
        return;
    }
#endif

    // a row view of a reordered weight: multiply a standard-layout copy of its rows (see unreorder_blocks)
    ggml_sycl_pool_alloc<char> src0_std_alloc(ctx.pool());
    ggml_tensor                src0_std;
    if (const ggml_tensor * parent = ggml_sycl_reordered_view_parent(src0)) {
        const size_t  bs = ggml_type_size(src0->type);
        const int64_t b0 = (int64_t) (src0->view_offs / bs);
        const int64_t n  = (int64_t) (ggml_nbytes(src0) / bs);
        char * buf = src0_std_alloc.alloc(ggml_nbytes(src0));
        if (!unreorder_blocks(src0->type, (const uint8_t *) parent->data, ggml_nbytes(parent), b0, n, buf, ctx.stream())) {
            GGML_ABORT("%s: view %s of reordered weight %s: type %s not handled\n", __func__, src0->name, parent->name,
                       ggml_type_name(src0->type));
        }
        src0_std           = *src0;
        src0_std.data      = buf;
        src0_std.view_src  = nullptr;
        src0_std.view_offs = 0;
        src0_std.extra     = nullptr;
        src0               = &src0_std;
    }

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
    if (src1->ne[1] == 1 && ggml_sycl_xmx_min_cols(src0->type) == 1 && can_use_xmx_batch(ctx.device, src0, src1)) {
        use_dequantize_mul_mat_vec = false;
    }
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(ctx.device, src0, src1, dst);

    bool use_mul_mat_q = can_use_mul_mat_q(src0, src1, dst);

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    if (g_ggml_sycl_mmv_f32 && !split && ggml_sycl_mmv_f32_eligible(src0, src1, dst)) {
        ggml_sycl_mmv_f32(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
static bool ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, const ggml_tensor * wu,
                                             const ggml_tensor * wg, const ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        return false;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    const auto plain_type = [](ggml_type t) { return t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    if (!plain_type(wu->type) || !plain_type(wg->type)) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (float *) glu->data, (int) ne00, (int) wu->ne[1],
                                             (int) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(int) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

// Gate and up weights of one type on the XMX engines: one launch reads the activations once and applies the GLU.
// Both weights must already be in the reorder layout. Returns false if the kernel declined.
static bool ggml_sycl_mul_mat_glu_xmx(ggml_backend_sycl_context & ctx,
                                      ggml_tensor *               glu,
                                      ggml_tensor *               up,
                                      const ggml_tensor *         wu,
                                      const ggml_tensor *         wg,
                                      const ggml_tensor *         act) {
    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);
    const int       stride_y_bytes   = src1_padded_cols * (int) sizeof(block_q8_1) / QK8_1;

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (XMX)");

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), (size_t) ne11 * stride_y_bytes);
    char *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_xmx(ctx.device, wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                           (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11, stride_y_bytes,
                                           /*stride_col_dst=*/(int) glu->ne[0], stream);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
static bool ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, const ggml_tensor * wu,
                                             const ggml_tensor * wg, const ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    const char * act_begin = (const char *) act->data;
    const char * glu_begin = (const char *) glu->data;
    if (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        return false;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    return ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (const float *) act->data, (float *) glu->data,
                                                              (int) wu->ne[0], (int) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
static bool ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return false;
    }

    ggml_tensor *       glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    if (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        return false;
    }
    // the fused kernels read the weights in place: a row view of a reordered weight takes ggml_sycl_mul_mat()
    if (ggml_sycl_reordered_view_parent(wu) || ggml_sycl_reordered_view_parent(wg)) {
        return false;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    if (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        return true;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }

    // gate and up of one type the XMX kernel handles, it needs the reorder layout like the unfused mmvq path
    if (wg->type == wu->type && ggml_sycl_xmx_supports_type(wu->type) && gate->src[1] == act &&
        act->ne[1] >= ggml_sycl_xmx_min_cols(wu->type) && act->ne[1] <= g_ggml_sycl_xmx_glu_max_cols) {
        opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
        opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);
        if (can_use_xmx_batch(ctx.device, wu, act) && can_use_xmx_batch(ctx.device, wg, act) &&
            ggml_sycl_mul_mat_glu_xmx(ctx, glu, up, wu, wg, act)) {
            return true;
        }
    }

    // the kernels below take up to MMVQ_MAX_BATCH_SIZE columns, a wider batch got here for the XMX kernel
    if (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    const bool reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    if (!reorder_pair) {
        const auto is_plain_type = [](ggml_type type) {
            return type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS;
        };
        if (!is_plain_type(wu->type) || !is_plain_type(wg->type)) {
            return false;
        }
        return ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    if (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        return false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (int) sizeof(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(int) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
static int ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_sycl_info().device_count != 1 || node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 || node->src[0]->ne[0] >= 1024) {
        return 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    int           count = 0;
    int           last  = node_idx;
    float         eps0;
    memcpy(&eps0, node->op_params, sizeof(float));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * ab = (const char *) a->data;
        const char * bb = (const char *) b->data;
        return ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    for (int j = node_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        if (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;  // not a launch; cannot break a run of adjacent norms
        }
        if (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, node) ||
            !ggml_are_same_shape(nj->src[0], node->src[0])) {
            break;  // any other launch ends the run
        }
        bool same_nb = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (nj->nb[d] != node->nb[d] || nj->src[0]->nb[d] != node->src[0]->nb[d]) {
                same_nb = false;
                break;
            }
        }
        if (!same_nb) {
            break;  // one nb[] stride set is shared by the whole batch
        }
        float epsj;
        memcpy(&epsj, nj->op_params, sizeof(float));
        if (epsj != eps0) {
            break;  // eps mismatch ends the run
        }
        bool indep = true;
        for (int k = 0; k < count; ++k) {
            if (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = false;
                break;
            }
        }
        if (!indep) {
            break;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    if (count < 2) {
        return 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    return last - node_idx;
}


__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

__dpct_inline__ static void k_zero_dst_rows(
    char *__restrict__ dst_original,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = 0.0f;
    }
}

// layout checks for the fused MMVQ path; the type check is ggml_sycl_mul_mat_vec_q_id_supports_type
static bool ggml_sycl_mul_mat_id_mmvq_eligible(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    if (ne12 != 1) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;

    const int64_t n_ids_per_group = ids->ne[0];
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t)) return false;
    if (ids->ne[1] != 1) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    return true;
}

// returns true when ggml_sycl_mul_mat_id takes the fused MMVQ path
static bool ggml_sycl_mul_mat_id_should_use_mmvq(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    return ggml_sycl_mul_mat_id_mmvq_eligible(src0, dst->src[1], dst->src[2], dst) &&
           ggml_sycl_mul_mat_vec_q_id_supports_type(src0->type);
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    if (!ggml_sycl_mul_mat_id_should_use_mmvq(dst)) return false;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t n_ids_per_group = ids->ne[0];
    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    }

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    if (use_reorder) {
        return ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
            (float *) dst->data, (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    return ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
        (float *) dst->data, (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// the XMX variant of the multi-column MoE GEMV would take src0 in the layout opt_for_reorder_id() leaves it in
static bool ggml_sycl_mmid_mc_xmx_expected(const ggml_tensor * src0) {
    if (!g_ggml_sycl_mmid_mc_xmx) {
        return false;
    }
    const bool reordered = src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
    const bool reorders  = g_ggml_sycl_enable_optimize &&
                          (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_K ||
                           src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K);
    return ggml_sycl_mmid_mc_xmx_supported(src0->type, reordered || reorders, (int) src0->ne[0], src0->nb[2],
                                           src0->data);
}

// true when ggml_sycl_mul_mat_id takes the multi-column MoE GEMV: 2..g_ggml_sycl_mmid_mc_max tokens (up to
// g_ggml_sycl_mmid_mc_xmx_max for the XMX variant), a type mmvq-id-multi.cpp covers in either layout, and the
// f32/i32 layout it reads
static bool ggml_sycl_mul_mat_id_mc_eligible(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const int64_t       ne12 = src1->ne[2];
    if (ne12 < 2 || ne12 > std::max(g_ggml_sycl_mmid_mc_max, g_ggml_sycl_mmid_mc_xmx ? g_ggml_sycl_mmid_mc_xmx_max : 0)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || ggml_backend_buffer_is_sycl_split(src0->buffer) || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || src1->ne[0] % QK8_1 != 0 || dst->ne[0] != src0->ne[1] ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t n_ids = ids->ne[0];
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[1] != ne12 || ids->ne[2] != 1) {
        return false;
    }
    if (src1->ne[1] != 1 && src1->ne[1] != n_ids) {
        return false;
    }
    if (src0->ne[2] > GGML_SYCL_MMID_MC_MAX_EXPERTS || n_ids * ne12 > GGML_SYCL_MMID_MC_MAX_ROUTES) {
        return false;
    }
    // past g_ggml_sycl_mmid_mc_max tokens only the XMX variant is faster than the grouped GEMM
    if (ne12 > g_ggml_sycl_mmid_mc_max && !ggml_sycl_mmid_mc_xmx_expected(src0)) {
        return false;
    }
    // the reorderable types have a kernel in both layouts, the others are never reordered
    return ggml_sycl_mmid_mc_supports_type(src0->type, false);
}

// the multi-column MoE GEMV of a MUL_MAT_ID that ggml_sycl_mul_mat_id_mc_eligible() accepts; false, with nothing
// queued, when the layout src0 ends up in has no kernel. sched_ext: a schedule buffer of
// ggml_sycl_mmid_mc_sched_ints() to use instead of a pool one, sched_ready: it holds the schedule of these ids.
static bool ggml_sycl_mul_mat_id_mc(ggml_backend_sycl_context & ctx, ggml_tensor * dst, int32_t * sched_ext = nullptr,
                                    bool sched_ready = false) {
    if (!ggml_sycl_mul_mat_id_mc_eligible(dst)) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    opt_for_reorder_id(&ctx, src0);
    const bool reordered = src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
    if (!ggml_sycl_mmid_mc_supports_type(src0->type, reordered)) {
        return false;
    }
    const queue_ptr stream   = ctx.stream();
    const int       ne10     = (int) src1->ne[0];
    const int       ne11     = (int) src1->ne[1];
    const int       ne12     = (int) src1->ne[2];
    const int       n_ids    = (int) ids->ne[0];
    const int       n_as     = (int) src0->ne[2];
    const int       padded   = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t    row_q8   = (size_t) padded * sizeof(block_q8_1) / QK8_1;
    const int       nrows_q8 = ne11 * ne12;

    const int  n_routes = n_ids * ne12;
    ggml_sycl_pool_alloc<int32_t> sched_pool(ctx.pool());
    int32_t * sched = sched_ext ? sched_ext : sched_pool.alloc(ggml_sycl_mmid_mc_sched_ints(n_as, n_routes));
    if (!sched_ext) {
        sched_ready = false;
    }

    if (g_ggml_sycl_mmid_mc_xmx && ggml_sycl_xmx_enabled(ctx.device) && row_q8 % 64 == 0 &&
        ggml_sycl_mmid_mc_xmx_supported(src0->type, reordered, ne10, src0->nb[2], src0->data)) {
        ggml_sycl_pool_alloc<char> packed(ctx.pool(), (size_t) n_routes * row_q8);
        if ((uintptr_t) packed.get() % 64 == 0) {
            ggml_sycl_mmid_mc_xmx(src0->type, src0->data, src0->nb[2], ne10, (int) src0->ne[1],
                                  (const char *) src1->data, src1->nb[1], src1->nb[2], ne11, packed.get(), row_q8,
                                  (const int32_t *) ids->data, ids->nb[1], n_ids, ne12, n_as, (float *) dst->data,
                                  dst->nb[1], dst->nb[2], sched, sched_ready, stream);
            return true;
        }
    }
    if (ne12 > g_ggml_sycl_mmid_mc_max) {
        return false;  // the grouped GEMM is faster than the DP4A kernels here
    }

    ggml_sycl_pool_alloc<char> src1_q8(ctx.pool(), (size_t) nrows_q8 * row_q8);
    if (reordered) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) src1->data, src1_q8.get(), ne10,
                                                               nrows_q8, padded, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>((const float *) src1->data, src1_q8.get(), ne10, nrows_q8, padded,
                                              stream);
    }
    ggml_sycl_mmid_mc(src0->type, reordered, src0->data, src0->nb[2], (int) src0->ne[0], (int) src0->ne[1],
                      src1_q8.get(), row_q8, ne11, (const int32_t *) ids->data, ids->nb[1], n_ids, ne12, n_as,
                      (float *) dst->data, dst->nb[1], dst->nb[2], sched, sched_ready, stream);
    return true;
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src,
        std::vector<mmid_row_mapping> & skipped_row_dst) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i == -1 || (row_id_i >= 0 && row_id_i < n_as));
            if (row_id_i == -1) {
                continue;
            }
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    for (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    skipped_row_dst.clear();
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i == -1 || (row_id_i >= 0 && row_id_i < n_as));

            // a skipped slot has no expert, its dst row is zeroed instead
            if (row_id_i == -1) {
                skipped_row_dst.push_back({(int32_t) id, (int32_t) iid1});
                continue;
            }

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
    routed_row_src.resize(expert_row_offsets[n_as]);
}

// true when the device MoE schedule may handle this MUL_MAT_ID: enabled, prefill, and an
// f32/i32 layout it accepts; ggml_sycl_grouped_dequant_gemm_device decides type and width
static bool ggml_sycl_mul_mat_id_device_sched_eligible(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DEVICE) && src1->ne[2] != 1 && ggml_is_contiguous(src0) &&
           ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
           src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->nb[1] == sizeof(float)*src1->ne[0] && dst->nb[1] == sizeof(float)*dst->ne[0];
}

// The device-scheduled grouped GEMM of a MUL_MAT_ID that ggml_sycl_mul_mat_id_device_sched_eligible() accepts.
// situ: dst->src[1] is the output of that GLU_OP_SITU node, which was not run; the pack reads its sources and
// applies it instead. sched: a schedule to build or reuse, see ggml_sycl_mmid_sched. epi: the SiTU epilogue to
// try, for the next projection run with combination epi_combo (see ggml_sycl_moe_block_fused). Returns false
// before queuing work when the grouped GEMM does not take the case.
// comb_w: the router weights [1, n_used, n_tokens] of the MoE combine that follows (ggml_sycl_moe_block_fused): dst
// is then written as weighted half rows [n_tokens][ne1][ne0] for ggml_sycl_moe_combine_half(), not as f32.
static bool ggml_sycl_mul_mat_id_device(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                        const ggml_tensor * situ, ggml_sycl_mmid_sched * sched = nullptr,
                                        const ggml_sycl_grouped_gemm_situ_out * epi = nullptr, int epi_combo = -1,
                                        const ggml_tensor * comb_w = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();
    const int64_t   n_as   = ne02;
    const int64_t   n_ids  = ids->ne[0];

    if (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
        ggml_sycl_xmx_gather_type_enabled(src0->type)) {
        opt_for_reorder_id(&ctx, src0);
    }
    const bool reordered = src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
    ggml_sycl_grouped_gemm_rows src1_rows = { (char *) src1->data, nullptr, ne11, nb11, nb12 };
#if GGML_SYCL_DNNL
    // dense rows of a producer that wrote their f16 copy for a dense matmul (the router): the same half rows
    if (!situ && nb11 == ne10*sizeof(float) && nb12 == ne11*nb11) {
        src1_rows.h = ggml_sycl_dense_dnn_act_lookup(ctx, src1->data, ne10, ne11*ne12, ne10);
    }
#endif
    ggml_sycl_grouped_gemm_situ situ_params = {};
    if (situ) {
        // x and g rows of the SiTU sources at the routed (slot, token), see ggml_sycl_mul_mat_id_situ_fused()
        const ggml_tensor * sx = situ->src[0];
        const ggml_tensor * sg = situ->src[1];
        const size_t        ts = sizeof(float);
        char *              x  = (char *) sx->data;
        char *              g  = (char *) (sg ? sg->data : sx->data);
        if (!sg) {
            const bool swapped = ggml_get_op_params_i32(situ, 1) != 0;
            x += swapped ? ne10 * ts : 0;
            g += swapped ? 0 : ne10 * ts;
        }
        src1_rows             = { x, nullptr, ne11, sx->nb[1], sx->nb[2] };
        situ_params.g_off     = g - x;
        situ_params.beta      = ggml_get_op_params_f32(situ, 2);
        situ_params.lb        = ggml_get_op_params_f32(situ, 3);
    }
    ggml_sycl_grouped_gemm_rows dst_rows = { (char *) dst->data, nullptr, ne1, nb1, nb2 };
    const size_t hnb1 = ne0 * sizeof(sycl::half), hnb2 = ne1 * hnb1;
    if (comb_w) {
        dst_rows.nb1   = hnb1;
        dst_rows.nb2   = hnb2;
        dst_rows.w     = (const char *) comb_w->data;
        dst_rows.w_nb1 = comb_w->nb[1];
        dst_rows.w_nb2 = comb_w->nb[2];
    }
    if (!ggml_sycl_grouped_dequant_gemm_device(src0->type, reordered, src0->data, nb02, src1_rows,
            ggml_sycl_src1_prec(dst), dst_rows, (const int32_t *) ids->data, ids->nb[1], n_as, n_ids,
            ids->ne[1], ne01, ne10, ctx.pool(), stream, situ ? &situ_params : nullptr, sched, epi, epi_combo)) {
        return false;
    }
    // the device schedule drops skipped slots (id == -1), so their dst rows are zeroed here: a work-item a
    // route (the grouped GEMM bounds them to INT32_MAX), which writes only where its slot is skipped
    const char *  ids_data = (const char *) ids->data;
    const size_t  ids_nb1  = ids->nb[1];
    char *        dst_data = (char *) dst->data;
    const int32_t nids     = (int32_t) n_ids;
    const int32_t n_routes = (int32_t) (n_ids * ids->ne[1]);
    constexpr int nth      = 256;
    if (comb_w) {
        // the half rows of the combine
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(((size_t) n_routes + nth - 1) / nth * nth), sycl::range<1>(nth)),
            [=](sycl::nd_item<1> item) {
                const int32_t r = (int32_t) item.get_global_id(0);
                if (r >= n_routes) {
                    return;
                }
                const int64_t iid1 = r / nids;
                const int64_t id   = r % nids;
                if (*(const int32_t *) (ids_data + iid1*ids_nb1 + id*sizeof(int32_t)) != -1) {
                    return;
                }
                sycl::half * row = (sycl::half *) (dst_data + id*hnb1 + iid1*hnb2);
                for (int64_t j = 0; j < ne0; ++j) {
                    row[j] = 0.0f;
                }
            });
        return true;
    }
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(((size_t) n_routes + nth - 1) / nth * nth), sycl::range<1>(nth)),
        [=](sycl::nd_item<1> item) {
            const int32_t r = (int32_t) item.get_global_id(0);
            if (r >= n_routes) {
                return;
            }
            const int64_t iid1 = r / nids;
            const int64_t id   = r % nids;
            if (*(const int32_t *) (ids_data + iid1*ids_nb1 + id*sizeof(int32_t)) != -1) {
                return;
            }
            float * row = (float *) (dst_data + id*nb1 + iid1*nb2);
            for (int64_t j = 0; j < ne0; ++j) {
                row[j] = 0.0f;
            }
        });
    return true;
}

// the grouped GEMM combination ggml_sycl_mul_mat_id_device would run dst with, -1 when it would not take it
static int ggml_sycl_mul_mat_id_device_combo(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor *       src0 = dst->src[0];
    const ggml_tensor * ids  = dst->src[2];
    if (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
        ggml_sycl_xmx_gather_type_enabled(src0->type)) {
        opt_for_reorder_id(&ctx, src0);
    }
    const bool reordered = src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
    if (dst->src[1]->ne[1] <= 0 || dst->ne[1] <= 0) {
        return -1;
    }
    return ggml_sycl_grouped_dequant_gemm_device_combo(src0->type, reordered, ggml_sycl_src1_prec(dst), src0->ne[2],
                                                       ids->ne[0], ids->ne[1], src0->ne[1], src0->ne[0], ctx.stream());
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    if (ne12 == 1) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }
    GGML_ASSERT(!ggml_sycl_mul_mat_id_should_use_mmvq(dst));

    if (ggml_sycl_mul_mat_id_mc(ctx, dst)) {
        return;
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    if (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    if (ggml_sycl_mul_mat_id_device_sched_eligible(dst) && ggml_sycl_mul_mat_id_device(ctx, dst, nullptr)) {
        return;
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 == -1 || (i02 >= 0 && i02 < n_as));

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            if (i02 == -1) { // skipped slot
                SYCL_CHECK(CHECK_TRY_ERROR(
                        stream->memset(dst_original + i1*nb1 + i2*nb2, 0, ne0*sizeof(float))));
                continue;
            }

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;
        // the slot/token pairs of the skipped slots, whose dst rows are zeroed
        std::vector<mmid_row_mapping> skipped_row_dst;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src, skipped_row_dst);

        const int64_t n_valid_rows = (int64_t) routed_row_src.size();

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_valid_rows*sizeof(mmid_row_mapping))));

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        if (!skipped_row_dst.empty()) {
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_skipped(ctx.pool(), skipped_row_dst.size());
            SYCL_CHECK(CHECK_TRY_ERROR(
                    stream->memcpy(dev_skipped.get(), skipped_row_dst.data(), skipped_row_dst.size()*sizeof(mmid_row_mapping))));

            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, skipped_row_dst.size());
            stream->submit([&](sycl::handler &cgh) {
                mmid_row_mapping *__restrict dev_skipped_get = dev_skipped.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_zero_dst_rows(dst_original, dev_skipped_get, ne0, nb1, nb2, item_ct1);
                    });
            });
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }

        if (n_valid_rows > 0) {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_valid_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        bool grouped = false;
        if (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            if (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            const bool src0_reordered =
                src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (const float *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (float *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        for (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        if (n_valid_rows > 0) {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_valid_rows);
            stream->submit([&](sycl::handler &cgh) {
                const char *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                const mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


static void ggml_sycl_set_main_device(const int main_device) try {
    if (get_current_device_id() == main_device) {
        return;
    }
    check_allow_gpu_index(main_device);
    SYCL_CHECK(ggml_sycl_set_device(main_device));

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SITU:
                    ggml_sycl_situ(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_L2_CLIP:
            ggml_sycl_l2_clip(ctx, dst);
            break;
        case GGML_OP_GATE_MIX:
            ggml_sycl_gate_mix(ctx, dst);
            break;
        case GGML_OP_GATE_MIX_MV:
            ggml_sycl_gate_mix_mv(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_CONV_HIST:
            ggml_sycl_ssm_conv_hist(ctx, dst);
            break;
        case GGML_OP_BLOCK_MIX:
            ggml_sycl_block_mix(ctx, dst);
            break;
        case GGML_OP_REL_BIAS:
            ggml_sycl_rel_bias(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool sycl_get_mem_info(int device, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#else
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    bool res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *free, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *free);
    return res;
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    if (!sycl_get_mem_info(device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *free, *total);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
static int ggml_sycl_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// t, not a graph output, is read (directly or through views) by no node after node `last`
static bool ggml_sycl_read_only_by(const ggml_cgraph * cgraph, const ggml_tensor * t, int last) {
    if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    for (int i = last + 1; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n == t || n->view_src == t) {
            return false;
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * src = n->src[k];
            if (src && (src == t || src->view_src == t)) {
                return false;
            }
        }
    }
    return true;
}

// Row-exact verify (GGML_SYCL_ROW_EXACT=1, the default): in a graph whose flash attention is row-exact
// (ggml_flash_attn_ext_set_row_exact, a speculative verify: every row an output in a context that can roll back) the
// quantized weight products of 2..GGML_SYCL_ROW_EXACT_MAX (8) columns run each column with the kernel a one-token
// decode uses, and an attention of more than 4 query rows runs as 4-row slices on the decode kernel
// (ggml_sycl_row_exact_node), so every row is bit-identical to the decode of its token alone and a speculative decode
// accepts the tokens the decode without speculation gives. GGML_SYCL_ROW_EXACT=0: the batched kernels (their
// q8_1 rounding of the activations differs from the f32 one of the one-column DMMV, and the MoE GEMM and the
// attention of 5+ rows round differently from the one-token kernels).
static bool g_ggml_sycl_row_exact = false; // the graph being computed is a row-exact one

static bool ggml_sycl_row_exact_graph(const ggml_cgraph * cgraph) {
    // 2: every graph (a diagnostic for tools that compute the nodes one by one, without the attention in the graph)
    static const int env = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT", 1);
    if (env != 1) {
        return env == 2;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_FLASH_ATTN_EXT) {
            return ggml_get_op_params_i32(n, 6) != 0;
        }
    }
    return false;
}

static int ggml_sycl_row_exact_max() {
    static const int n = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT_MAX", 8);
    return n;
}

// GLU_OP_SITU at node_idx whose only reader is the next node, a device-scheduled MUL_MAT_ID taking it as src1
// (the MoE down projection after the gate_up one), in a form the down pack can apply it in: the pack then
// gathers the gate_up rows and applies the SiTU, so its output is never written.
static bool ggml_sycl_situ_mmid_fusable(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion || node_idx + 1 >= cgraph->n_nodes || g_ggml_sycl_row_exact) {
        return false;
    }
    const ggml_tensor * situ = cgraph->nodes[node_idx];
    const ggml_tensor * mmid = cgraph->nodes[node_idx + 1];
    if (situ->op != GGML_OP_GLU || ggml_get_glu_op(situ) != GGML_GLU_OP_SITU || mmid->op != GGML_OP_MUL_MAT_ID ||
        (mmid->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || mmid->src[1] != situ || mmid->src[0] == situ ||
        mmid->src[2] == situ || !ggml_node_has_n_uses(cgraph, node_idx, 1)) {
        return false;
    }
    const ggml_tensor * sx = situ->src[0];
    const ggml_tensor * sg = situ->src[1];
    // the pack addresses x and g of a routed row with the strides of x, and the SiTU output with those of a
    // contiguous f32 tensor of the same shape
    return situ->type == GGML_TYPE_F32 && ggml_sycl_situ_supported(situ) && ggml_is_contiguous(situ) &&
           situ->ne[3] == 1 && sx->ne[3] == 1 && !ggml_backend_buffer_is_sycl_split(mmid->src[0]->buffer) &&
           (!sg || (sg->nb[1] == sx->nb[1] && sg->nb[2] == sx->nb[2] && sg->ne[0] == situ->ne[0] &&
                    sx->ne[0] == situ->ne[0])) &&
           (sg || sx->ne[0] == 2 * situ->ne[0]) && sx->ne[1] == situ->ne[1] && sx->ne[2] == situ->ne[2] &&
           mmid->src[1]->ne[2] != 1 && mmid->src[0]->ne[3] == mmid->src[1]->ne[3] &&
           ggml_sycl_mul_mat_id_device_sched_eligible(mmid) && !ggml_sycl_mul_mat_id_mc_eligible(mmid);
}

// The MoE combine after the down MUL_MAT_ID at node_idx: the index of an OUT_PROD of that output and the router
// weights (out[e, 0, t] = sum_k down[e, k, t] * w[0, k, t], see build_moe_ffn), past view nodes only, which
// nothing but it reads; -1 when there is none or it is not the f32 contiguous form the half combine takes
// (GGML_SYCL_MOE_COMBINE=0: never).
static int ggml_sycl_moe_combine_node(const ggml_cgraph * cgraph, int node_idx) {
    static const bool env = ggml_sycl_get_env("GGML_SYCL_MOE_COMBINE", 1) != 0;
    const ggml_tensor * down = cgraph->nodes[node_idx];
    int                 k    = node_idx + 1;
    while (k < cgraph->n_nodes && ggml_sycl_is_view_or_noop(cgraph->nodes[k]) && cgraph->nodes[k]->view_src != down) {
        ++k;
    }
    if (!env || k >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * op = cgraph->nodes[k];
    const ggml_tensor * w  = op->src[1];
    if (op->op != GGML_OP_OUT_PROD || op->src[0] != down || (op->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        down->type != GGML_TYPE_F32 || !ggml_is_contiguous(down) || down->ne[3] != 1 ||
        w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != down->ne[1] || w->ne[2] != down->ne[2] ||
        w->ne[3] != 1 || w->nb[1] % sizeof(float) != 0 || w->nb[2] % sizeof(float) != 0 ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || op->ne[0] != down->ne[0] || op->ne[1] != 1 ||
        op->ne[2] != down->ne[2] || op->ne[3] != 1 || down->ne[0] % 8 != 0 ||
        ggml_nbytes(w) > UINT32_MAX || !ggml_sycl_read_only_by(cgraph, down, k)) {
        return -1;
    }
    return k;
}

// SiTU + down MUL_MAT_ID as above; false, with nothing queued, when the pair does not qualify
static bool ggml_sycl_mul_mat_id_situ_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    return ggml_sycl_situ_mmid_fusable(cgraph, node_idx) &&
           ggml_sycl_mul_mat_id_device(ctx, cgraph->nodes[node_idx + 1], cgraph->nodes[node_idx]);
}

// The MoE expert block at a few tokens (ggml_sycl_mul_mat_id_mc_eligible): gate_up MUL_MAT_ID, views, SiTU, down
// MUL_MAT_ID with the same ids, the two multi-column GEMVs sharing one schedule. Returns the nodes to skip, 0 with
// nothing queued when the block does not qualify.
static int ggml_sycl_moe_block_mc(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * gate_up = cgraph->nodes[node_idx];
    int           j       = node_idx + 1;
    while (j < cgraph->n_nodes && ggml_sycl_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * situ = cgraph->nodes[j];
    ggml_tensor * down = cgraph->nodes[j + 1];
    if (situ->op != GGML_OP_GLU || ggml_get_glu_op(situ) != GGML_GLU_OP_SITU || down->op != GGML_OP_MUL_MAT_ID ||
        (down->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (situ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        down->src[1] != situ || down->src[2] != gate_up->src[2] || down->src[0]->ne[2] != gate_up->src[0]->ne[2] ||
        !ggml_sycl_mul_mat_id_mc_eligible(down)) {
        return 0;
    }
    const ggml_tensor *           ids = gate_up->src[2];
    ggml_sycl_pool_alloc<int32_t> sched(
        ctx.pool(), ggml_sycl_mmid_mc_sched_ints((int) gate_up->src[0]->ne[2], (int) (ids->ne[0] * ids->ne[1])));
    if (!ggml_sycl_mul_mat_id_mc(ctx, gate_up, sched.get(), false)) {
        return 0;
    }
    ggml_sycl_situ(ctx, situ);
    if (!ggml_sycl_mul_mat_id_mc(ctx, down, sched.get(), true)) {
        ggml_sycl_mul_mat_id(ctx, down);
    }
    return j + 1 - node_idx;
}

// The MoE expert block, gate_up MUL_MAT_ID, views, SiTU, down MUL_MAT_ID with the same ids: one device
// schedule serves both projections, and the SiTU goes into the down pack. Returns the nodes to skip, 0 with
// nothing queued when the block does not qualify.
static int ggml_sycl_moe_block_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * gate_up = cgraph->nodes[node_idx];
    if (!g_ggml_sycl_enable_fusion || gate_up->op != GGML_OP_MUL_MAT_ID || gate_up->src[1]->ne[2] == 1 ||
        gate_up->src[0]->ne[3] != gate_up->src[1]->ne[3] ||
        ggml_backend_buffer_is_sycl_split(gate_up->src[0]->buffer)) {
        return 0;
    }
    if (ggml_sycl_mul_mat_id_mc_eligible(gate_up)) {
        const int n = ggml_sycl_moe_block_mc(ctx, cgraph, node_idx);
        if (n > 0) {
            return n;
        }
    }
    if (!ggml_sycl_mul_mat_id_device_sched_eligible(gate_up)) {
        return 0;
    }
    int j = node_idx + 1;
    while (j < cgraph->n_nodes && ggml_sycl_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j >= cgraph->n_nodes || !ggml_sycl_situ_mmid_fusable(cgraph, j)) {
        return 0;
    }
    ggml_tensor * situ = cgraph->nodes[j];
    ggml_tensor * down = cgraph->nodes[j + 1];
    if (down->src[2] != gate_up->src[2] || down->src[0]->ne[2] != gate_up->src[0]->ne[2]) {
        return 0;
    }
    // The SiTU in the gate_up epilogue, which packs the down B and does not write the gate_up output: its x
    // and g are row ranges of that output, which nothing but the block reads (GGML_SYCL_MOE_EPI=0: off)
    static const bool epi_env = ggml_sycl_get_env("GGML_SYCL_MOE_EPI", 1) != 0;
    ggml_sycl_grouped_gemm_situ_out epi       = {};
    int                             epi_combo = -1;
    if (epi_env && ggml_sycl_read_only_by(cgraph, gate_up, j)) {
        const ggml_tensor * sx = situ->src[0];
        const ggml_tensor * sg = situ->src[1];
        const int64_t       D  = situ->ne[0];
        const int64_t       ts = sizeof(float);
        int64_t             xo = (const char *) sx->data - (const char *) gate_up->data;
        int64_t             go = (const char *) (sg ? sg->data : sx->data) - (const char *) gate_up->data;
        if (!sg) {
            const bool swapped = ggml_get_op_params_i32(situ, 1) != 0;
            xo += swapped ? D * ts : 0;
            go += swapped ? 0 : D * ts;
        }
        auto same_rows = [&](const ggml_tensor * v) {
            return !v || (v->view_src == gate_up && v->nb[1] == gate_up->nb[1] && v->nb[2] == gate_up->nb[2]);
        };
        if (sx->view_src == gate_up && same_rows(sx) && same_rows(sg) && xo >= 0 && go >= 0 && xo % ts == 0 &&
            go % ts == 0 && xo / ts + D <= gate_up->ne[0] && go / ts + D <= gate_up->ne[0] &&
            down->src[0]->ne[0] == D && D <= INT32_MAX) {
            epi       = { nullptr, (int) (xo / ts), (int) (go / ts), (int) D, ggml_get_op_params_f32(situ, 2),
                          ggml_get_op_params_f32(situ, 3), nullptr, 0 };
            epi_combo = ggml_sycl_mul_mat_id_device_combo(ctx, down);
        }
    }
    ggml_sycl_mmid_sched sched;
    if (!ggml_sycl_mul_mat_id_device(ctx, gate_up, nullptr, &sched, epi_combo >= 0 ? &epi : nullptr, epi_combo)) {
        return 0;
    }
    const bool packed_by_epi = sched.next_K != 0;
    // The combine (OUT_PROD of the down output and the router weights) in two halves: the down epilogue scales
    // each route's row by its weight and stores it as half, ggml_sycl_moe_combine_half() sums a token's rows.
    // The [n_embd, n_used, n_tokens] f32 down output was written and read back once (151 MB a ub2048 ubatch).
    const int     c      = packed_by_epi ? ggml_sycl_moe_combine_node(cgraph, j + 1) : -1;
    ggml_tensor * comb   = c > 0 ? cgraph->nodes[c] : nullptr;
    const bool    halves = comb && ggml_sycl_mul_mat_id_device(ctx, down, situ, &sched, nullptr, -1, comb->src[1]);
    if (!halves && !ggml_sycl_mul_mat_id_device(ctx, down, situ, &sched)) {
        // the combination check above makes the down call take the epilogue's B
        GGML_ASSERT(!packed_by_epi && "the gate_up output was not written");
        ggml_sycl_situ(ctx, situ);
        ggml_sycl_mul_mat_id(ctx, down);
    }
    GGML_ASSERT(sched.next_K == 0 && "the down projection did not take the epilogue's B");
    if (halves) {
        ggml_sycl_moe_combine_half(ctx, comb, (const sycl::half *) down->data, (int) down->ne[1]);
        return c - node_idx;
    }
    return j + 1 - node_idx;
}
// GGML_SYCL_NODE_PROF=1: wait on the queue around every computed node (not while recording) and print, per graph of at
// least GGML_SYCL_NODE_PROF_MIN_NODES nodes, the host time per op and weight (blk.N stripped). A diagnostic: the
// waits serialise the queue, so the totals are only shares.
struct ggml_sycl_node_prof {
    int enabled = -1;
    int min_nodes = 1000;
    std::map<std::string, std::pair<double, int>> acc;
    std::string cur;
    std::chrono::steady_clock::time_point t0;

    static std::string key(const ggml_tensor * node) {
        std::string k = ggml_op_desc(node);
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0] && node->src[1]) {
            const ggml_tensor * w = node->src[0]->view_src ? node->src[0]->view_src : node->src[0];
            std::string n = w->name;
            n = std::regex_replace(n, std::regex("blk\\.[0-9]+\\."), "");
            char buf[160];
            snprintf(buf, sizeof(buf), " %s[%s %lldx%lld%s] n=%lld%s", n.c_str(), ggml_type_name(node->src[0]->type),
                     (long long) node->src[0]->ne[1], (long long) node->src[0]->ne[0],
                     node->src[0]->view_src ? " view" : "", (long long) node->src[1]->ne[1],
                     ggml_is_contiguous(node->src[1]) ? "" : " src1-nc");
            k += buf;
        }
        return k;
    }
    void mark(queue_ptr q, const ggml_tensor * next) {
        q->wait();
        auto t = std::chrono::steady_clock::now();
        if (!cur.empty()) {
            auto & a = acc[cur];
            a.first += std::chrono::duration<double, std::micro>(t - t0).count();
            a.second += 1;
        }
        cur = next ? key(next) : std::string();
        t0  = std::chrono::steady_clock::now();
    }
    void report(int n_nodes) {
        if (n_nodes >= min_nodes) {
            std::vector<std::pair<std::string, std::pair<double, int>>> v(acc.begin(), acc.end());
            std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
            double tot = 0;
            for (auto & e : v) tot += e.second.first;
            fprintf(stderr, "[node-prof] graph of %d nodes: %.2f ms\n", n_nodes, tot / 1e3);
            for (auto & e : v) {
                fprintf(stderr, "[node-prof] %9.3f ms %5d calls %8.1f us  %s\n", e.second.first / 1e3, e.second.second,
                        e.second.first / e.second.second, e.first.c_str());
            }
        }
        acc.clear();
    }
};
static ggml_sycl_node_prof g_ggml_sycl_node_prof;

#if GGML_SYCL_DNNL
// a MUL_MAT among the next nodes that ggml_sycl_mul_mat() hands to the dense oneDNN path with node i as src1,
// converted to f16 there: a fused producer of node i can write that copy itself (ggml_sycl_dense_dnn_act_reserve)
static bool ggml_sycl_dense_dnn_reads_f16(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * t = cgraph->nodes[i];
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 16; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && n->src[1] == t && !ggml_backend_buffer_is_sycl_split(n->src[0]->buffer) &&
            ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
            ggml_sycl_dense_dnn_converts_src1(ctx.device, n)) {
            return true;
        }
    }
    return false;
}

// the queries of BLOCK_MIX node i whose plane-0 rows [n, n_tokens] (a reshape or row view of the node) are src1 of
// a MUL_MAT among the next nodes that the dense oneDNN path converts to f16: the kernel can write those copies
static bool ggml_sycl_block_mix_dense_reads(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i, bool * f16) {
    const ggml_tensor * t = cgraph->nodes[i];
    bool any = false;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 96; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_CONCAT && ggml_sycl_dense_dnn_concat_reads_f16(ctx.device, cgraph, j)) {
            // a half [ne0, 1, n_tokens] of the controller gate input: the plane-0 rows of a query
            for (int h = 0; h < 2; ++h) {
                const ggml_tensor * b = n->src[h];
                if ((b != t && b->view_src != t) || b->type != GGML_TYPE_F32 || b->ne[0] != t->ne[0] ||
                    b->ne[1] != 1 || b->ne[2] != t->ne[2] || b->ne[3] != 1 || b->nb[0] != sizeof(float) ||
                    b->nb[2] != t->nb[2] || (const char *) b->data < (const char *) t->data) {
                    continue;
                }
                const size_t offs = (const char *) b->data - (const char *) t->data;
                if (offs % t->nb[1] == 0 && (int64_t) (offs / t->nb[1]) < t->ne[1]) {
                    any = any || !f16[offs / t->nb[1]];
                    f16[offs / t->nb[1]] = true;
                }
            }
            continue;
        }
        if (n->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * b = n->src[1];
        if ((b != t && b->view_src != t) || b->type != GGML_TYPE_F32 || b->ne[0] != t->ne[0] || b->ne[1] != t->ne[2] ||
            b->ne[2] != 1 || b->ne[3] != 1 || b->nb[0] != sizeof(float) || b->nb[1] != t->nb[2] ||
            (const char *) b->data < (const char *) t->data) {
            continue;
        }
        const size_t offs = (const char *) b->data - (const char *) t->data;
        if (offs % t->nb[1] != 0 || (int64_t) (offs / t->nb[1]) >= t->ne[1] || f16[offs / t->nb[1]]) {
            continue;
        }
        if (ggml_backend_buffer_is_sycl_split(n->src[0]->buffer) ||
            ggml_get_op_params_i32(n, 1) == GGML_HINT_SRC0_IS_HADAMARD || !ggml_sycl_dense_dnn_converts_src1(ctx.device, n)) {
            continue;
        }
        f16[offs / t->nb[1]] = true;
        any = true;
    }
    return any;
}

// GGML_SYCL_DROP_F32: a producer that writes the f16 copy of its result for the dense matmuls can skip the f32
// result when no node can read those f32 bytes. The proof is over the whole graph: every node after the producer
// that references bytes of the result's root tensor (through any view) is checked, and those that overlap the rows
// must all be MUL_MATs that the dense oneDNN path runs with exactly those rows as src1 (so they read the f16 copy,
// which is pinned in the activation cache until the last of them, see ggml_sycl_dense_dnn_act_reserve).
// touch[root] = the non-view nodes whose sources or in-place destination are views of root (or root itself)
struct ggml_sycl_drop_f32_index {
    const ggml_cgraph * graph = nullptr;
    std::map<const ggml_tensor *, std::vector<int>> touch;
};
static thread_local ggml_sycl_drop_f32_index g_ggml_sycl_drop_f32_index;

static const ggml_tensor * ggml_sycl_root(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// the bytes of t (a view of root) can overlap the rows [M x K floats, row stride ld floats] at byte offset roff of root
static bool ggml_sycl_rows_overlap(const ggml_tensor * root, const ggml_tensor * t, size_t roff, int64_t K, int64_t M,
                                   int64_t ld) {
    const size_t toff = (const char *) t->data - (const char *) root->data;
    const size_t row  = (size_t) ld * sizeof(float);
    // rows of the same stride: disjoint column ranges never meet
    if (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[3] == 1 &&
        ((t->ne[2] == 1 && t->nb[1] == row) || (t->ne[1] == 1 && t->nb[2] == row))) {
        const size_t c0 = toff % row, c1 = c0 + t->ne[0] * sizeof(float);
        const size_t r0 = roff % row, r1 = r0 + K * sizeof(float);
        if (c1 <= row && r1 <= row && (c1 <= r0 || r1 <= c0)) {
            return false;
        }
    }
    const size_t rend = roff + (size_t) (M - 1) * row + K * sizeof(float);
    return toff < rend && roff < toff + ggml_nbytes(t);
}

// the last node reading the f32 rows [data, M x K floats at row stride ld] of the result of node `producer` (all of
// whose nodes up to `last_fused` are fused into it), when only dense matmuls that read the f16 copy read them; -1
// when the f32 rows must be written
static int ggml_sycl_drop_f32_last_reader_impl(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph,
                                               int last_fused, const ggml_tensor * base, const void * data, int64_t K,
                                               int64_t M, int64_t ld) {
    if (!ggml_sycl_dense_dnn_drop_f32() || base->view_src || (base->flags & GGML_TENSOR_FLAG_OUTPUT) || !base->data) {
        return -1;
    }
    auto & idx = g_ggml_sycl_drop_f32_index;
    if (idx.graph != cgraph) {
        idx.graph = cgraph;
        idx.touch.clear();
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_sycl_is_view_or_noop(n)) {
                continue;
            }
            const ggml_tensor * roots[GGML_MAX_SRC + 1];
            int nr = 0;
            if (n->view_src) {
                roots[nr++] = n->view_src;
            }
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (n->src[k]) {
                    roots[nr++] = ggml_sycl_root(n->src[k]);
                }
            }
            for (int k = 0; k < nr; ++k) {
                auto & v = idx.touch[roots[k]];
                if (v.empty() || v.back() != j) {
                    v.push_back(j);
                }
            }
        }
    }
    auto it = idx.touch.find(base);
    if (it == idx.touch.end()) {
        return -1;
    }
    const size_t roff = (const char *) data - (const char *) base->data;
    int last = -1;
    for (const int j : it->second) {
        if (j <= last_fused) {
            continue;
        }
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->view_src == base && ggml_sycl_rows_overlap(base, n, roff, K, M, ld)) {
            return -1; // writes into the rows
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = n->src[k];
            if (!s || ggml_sycl_root(s) != base || !ggml_sycl_rows_overlap(base, s, roff, K, M, ld)) {
                continue;
            }
            const bool rows2d = s->data == data && s->type == GGML_TYPE_F32 && s->ne[0] == K && s->ne[1] == M &&
                                s->ne[2] == 1 && s->ne[3] == 1 && s->nb[0] == sizeof(float) &&
                                s->nb[1] == (size_t) ld * sizeof(float);
            const bool rows3d = s->data == data && s->type == GGML_TYPE_F32 && s->ne[0] == K && s->ne[1] == 1 &&
                                s->ne[2] == M && s->ne[3] == 1 && s->nb[0] == sizeof(float) &&
                                s->nb[2] == (size_t) ld * sizeof(float);
            // a half of a controller gate input on the f16 path of ggml_sycl_dense_dnn_concat_mul_mat()
            if (n->op == GGML_OP_CONCAT && k < 2 && (rows2d || rows3d) && (n->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_sycl_dense_dnn_concat_ready(ctx.device, cgraph, j)) {
                last = std::max(last, j);
                continue;
            }
            const bool exact = k == 1 && rows2d;
            // a dense oneDNN matmul of a copied weight, not taken by the MUL_MAT + MUL_MAT + GLU fusion
            const bool dense = exact && n->op == GGML_OP_MUL_MAT && (n->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                               ggml_sycl_root(n->src[0]) != base &&
                               !ggml_backend_buffer_is_sycl_split(n->src[0]->buffer) &&
                               ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
                               ggml_sycl_dense_dnn_converts_src1(ctx.device, n) &&
                               ggml_sycl_dense_dnn_weight_ready(ctx.device, n->src[0]) &&
                               !(j + 1 < cgraph->n_nodes && cgraph->nodes[j + 1]->op == GGML_OP_GLU) &&
                               !(j + 2 < cgraph->n_nodes && cgraph->nodes[j + 2]->op == GGML_OP_GLU);
            if (!dense) {
                return -1;
            }
            last = std::max(last, j);
        }
    }
    return last;
}

static int ggml_sycl_drop_f32_last_reader(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int last_fused,
                                          const ggml_tensor * base, const void * data, int64_t K, int64_t M, int64_t ld) {
    const int last = ggml_sycl_drop_f32_last_reader_impl(ctx, cgraph, last_fused, base, data, K, M, ld);
    // GGML_SYCL_DROP_F32_DEBUG=1: the decisions in the first graphs of at least 1000 nodes (diagnostic)
    static int dbg = ggml_sycl_get_env("GGML_SYCL_DROP_F32_DEBUG", 0) ? 600 : 0;
    if (dbg > 0 && cgraph->n_nodes >= 1000) {
        --dbg;
        fprintf(stderr, "[drop-f32] node %d %s rows %lld x %lld +%zu: last reader %d %s\n", last_fused, base->name,
                (long long) M, (long long) K, (size_t) ((const char *) data - (const char *) base->data), last,
                last >= 0 ? cgraph->nodes[last]->name : "-");
    }
    return last;
}
#endif

// One-column Q4_K products that share their activation in one launch (ggml_sycl_dmmv_multi, dmmv.cpp): the products
// after node i run at node i, so every node in between that is not moved must neither write what a moved product
// reads nor touch what it writes. GGML_SYCL_DMMV_MULTI=0 restores the per-node path.
static bool ggml_sycl_dmmv_multi_eligible(ggml_backend_sycl_context & ctx, const ggml_tensor * n) {
    if (n->op != GGML_OP_MUL_MAT || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * w = n->src[0];
    const ggml_tensor * x = n->src[1];
    // GGML_SYCL_DMMV_MULTI_Q6=1: also the Q6_K weights (slower than the Q6_K kernel of the per-node path)
    static const bool q6 = ggml_sycl_get_env("GGML_SYCL_DMMV_MULTI_Q6", 0) != 0;
    if ((w->type != GGML_TYPE_Q4_K && !(q6 && w->type == GGML_TYPE_Q6_K)) || w->view_src || !ggml_is_contiguous(w) ||
        !w->extra || ggml_backend_buffer_is_sycl_split(w->buffer) || w->ne[2] != 1 || w->ne[3] != 1 ||
        !ggml_sycl_dmmv_multi_supported(w->type, w->ne[0])) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != w->ne[0] || x->ne[1] != 1 || x->ne[2] != 1 ||
        x->ne[3] != 1 || n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) || (uintptr_t) x->data % 16 != 0 ||
        (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    return ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD && g_ggml_sycl_enable_esimd &&
           !g_ggml_sycl_prioritize_dmmv && should_reorder_tensor(ctx, n);
}

static bool ggml_sycl_bytes_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// a and b cannot swap order or run at once: one writes what the other reads or writes
static bool ggml_sycl_nodes_conflict(const ggml_tensor * a, const ggml_tensor * b) {
    if (ggml_sycl_bytes_overlap(a, b)) {
        return true;
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (ggml_sycl_bytes_overlap(a, b->src[s]) || ggml_sycl_bytes_overlap(b, a->src[s])) {
            return true;
        }
    }
    return false;
}

static bool ggml_sycl_dmmv_multi_try(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i,
                                     std::vector<uint8_t> & done) {
    // GGML_SYCL_DMMV_MULTI_MAXSEG: products per launch (1..4; 1 still uses the kernel, which beats the per-node one)
    static const int max_seg = std::min(ggml_sycl_get_env("GGML_SYCL_DMMV_MULTI_MAXSEG", GGML_SYCL_DMMV_MULTI_MAX),
                                        GGML_SYCL_DMMV_MULTI_MAX);
    const ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_sycl_dmmv_multi_eligible(ctx, node)) {
        return false;
    }
    const ggml_tensor * x = node->src[1];
    const int64_t       K = x->ne[0];

    // a product read by a GLU keeps the gate/up fusions of ggml_sycl_fuse
    auto feeds_glu = [&](int j) {
        for (int k = j + 1; k < cgraph->n_nodes && k <= j + 4; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (n->op == GGML_OP_GLU && (n->src[0] == cgraph->nodes[j] || n->src[1] == cgraph->nodes[j])) {
                return true;
            }
        }
        return false;
    };
    if (feeds_glu(i)) {
        return false;
    }

    std::vector<int> moved = { i }; // the products of this launch
    auto try_move = [&](int q) {
        const ggml_tensor * t = cgraph->nodes[q];
        for (int k = i + 1; k < q; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || done[k] ||
                std::find(moved.begin(), moved.end(), k) != moved.end()) {
                continue;
            }
            if (ggml_sycl_nodes_conflict(t, n)) {
                return false;
            }
        }
        for (int m : moved) {
            if (ggml_sycl_nodes_conflict(t, cgraph->nodes[m])) {
                return false;
            }
        }
        moved.push_back(q);
        return true;
    };
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 40 && (int) moved.size() < max_seg; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || done[j]) {
            continue;
        }
        if (ggml_sycl_bytes_overlap(n, x)) {
            break; // the shared activation is rewritten
        }
        if (n->src[1] && n->src[1]->data == x->data && ggml_sycl_dmmv_multi_eligible(ctx, n) && n->src[0]->ne[0] == K &&
            n->src[0]->type == node->src[0]->type && !feeds_glu(j)) {
            try_move(j);
        }
    }

    ggml_sycl_dmmv_multi_args args = {};
    for (int m : moved) {
        const ggml_tensor * t = cgraph->nodes[m];
        opt_for_reorder(&ctx, t->src[0], t->src[1], const_cast<ggml_tensor *>(t), mul_mat_algo::DMMV);
        const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(t->src[0]->extra);
        if (!extra || !extra->optimized_feature.reorder) {
            return false;
        }
        ggml_sycl_dmmv_multi_seg & s = args.seg[args.nseg++];
        s.w     = t->src[0]->data;
        s.dst   = (float *) t->data;
        s.nrows = (int) t->src[0]->ne[1];
        s.type  = t->src[0]->type;
    }
    if (!ggml_sycl_dmmv_multi(args, (const float *) x->data, (int) K, ctx.stream())) {
        return false;
    }
    for (int m : moved) {
        done[m] = 1;
    }
    return true;
}

// The MoE GEMV of the n tokens of a row-exact graph (n * K routes of K experts) in one launch of the one-token fused
// MMVQ kernel (ggml_sycl_mul_mat_id_mmvq_fused): the routes sorted by expert (GGML_SYCL_ROW_EXACT_MOE_SORT=0: in token
// order), so the work-groups of one expert run together and its blocks come from the cache; the activation row of each
// route gathered into a row of its own (the one-token launch of the up and gate products shares one row across the K
// routes); the route results scattered back. Every route runs the work-groups of the one-token launch on the same
// quantized row, so the results are those of n one-token launches. By default the launch is the grouped form of that
// kernel (ggml_sycl_mul_mat_vec_q_id_grouped: one work-group row per expert, its routes in turn). False, with nothing
// queued, if not handled.
static bool ggml_sycl_row_exact_moe_flat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const int64_t K     = ids->ne[0];
    const int64_t n     = ids->ne[1];
    const int64_t ne10  = src1->ne[0];
    const int64_t ne11  = src1->ne[1];
    const int64_t nrows = src0->ne[1];
    const int     R     = (int) (K * n);
    constexpr int R_MAX = 256;
    if (R > R_MAX || src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src1->nb[1] != ne10 * sizeof(float) ||
        src1->nb[2] % sizeof(float) != 0 || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] % sizeof(int32_t) != 0 || dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) ||
        dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0 || (ne11 != 1 && ne11 != K) ||
        dst->ne[0] != nrows || dst->ne[1] != K) {
        return false;
    }
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;
    if (use_reorder && !ggml_sycl_mul_mat_vec_q_id_reorder_supports_type(src0->type)) {
        return false;
    }
    static const bool sort = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT_MOE_SORT", 1) != 0;
    // GGML_SYCL_ROW_EXACT_MOE_GROUP=0: a work-group row per route (else per group of routes of one expert, which
    // reads the expert's blocks once for the group: ggml_sycl_mul_mat_vec_q_id_grouped)
    static const bool group_env = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT_MOE_GROUP", 1) != 0;
    const bool grouped = group_env && sort && n <= 8 && ggml_sycl_mul_mat_vec_q_id_grouped_supports(src0->type, use_reorder);
    const queue_ptr stream = ctx.stream();
    const int       padded = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const size_t    qrow   = (size_t) padded * sizeof(block_q8_1) / QK8_1;

    ggml_sycl_pool_alloc<int32_t> rid_a(ctx.pool(), 5 * R);
    ggml_sycl_pool_alloc<float>   xs_a(ctx.pool(), (size_t) R * ne10);
    ggml_sycl_pool_alloc<char>    q_a(ctx.pool(), (size_t) R * qrow);
    ggml_sycl_pool_alloc<float>   dt_a(ctx.pool(), (size_t) R * nrows);
    int32_t *       rid  = rid_a.get();   // [R] expert of the route in slot p
    int32_t *       perm = rid + R;       // [R] route (t * K + k) in slot p
    int32_t *       grp  = rid + 2 * R;   // [3R] expert, first slot, slot count of each group of slots
    float *         xs   = xs_a.get();
    float *         dt   = dt_a.get();
    const int32_t * ids_d = (const int32_t *) ids->data;
    const size_t    ids_s = ids->nb[1] / sizeof(int32_t);
    const int       Ki    = (int) K;
    const bool      by_e  = sort;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<int32_t, 1> e(sycl::range<1>(R_MAX), h);
        sycl::local_accessor<int32_t, 1> se(sycl::range<1>(R_MAX), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(R), sycl::range<1>(R)), [=](sycl::nd_item<1> it) {
            const int r = (int) it.get_local_id(0);
            const int t = r / Ki;
            e[r]        = ids_d[t * ids_s + (r - t * Ki)];
            it.barrier(sycl::access::fence_space::local_space);
            int rank = r;
            if (by_e) {
                const int er = e[r];
                rank         = 0;
                for (int j = 0; j < R; ++j) {
                    const int ej = e[j];
                    rank += ej < er || (ej == er && j < r);
                }
            }
            rid[rank]  = e[r];
            perm[rank] = r;
            se[rank]   = e[r];
            it.barrier(sycl::access::fence_space::local_space);
            // the groups: runs of one expert in the sorted slots
            int gi = 0;
            int ng = 0;
            for (int j = 0; j < R; ++j) {
                const int st = j == 0 || se[j - 1] != se[j];
                ng += st;
                gi += st && j < r;
            }
            if (r == 0 || se[r - 1] != se[r]) {
                int len = 1;
                while (r + len < R && se[r + len] == se[r]) {
                    ++len;
                }
                grp[gi]         = se[r];
                grp[R + gi]     = r;
                grp[2 * R + gi] = len;
            }
            if (r >= ng) {
                grp[2 * R + r] = 0;
            }
        });
    });
    {
        const float *  x   = (const float *) src1->data;
        const size_t   s11 = ne11 == 1 ? 0 : src1->nb[1] / sizeof(float);
        const size_t   s12 = src1->nb[2] / sizeof(float);
        const int      nc  = (int) ne10;
        const size_t   tot = (size_t) R * nc;
        constexpr int  WG  = 256;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((tot + WG - 1) / WG * WG), sycl::range<1>(WG)),
                             [=](sycl::nd_item<1> it) {
                                 const size_t gi = it.get_global_id(0);
                                 if (gi >= tot) {
                                     return;
                                 }
                                 const int p   = (int) (gi / nc);
                                 const int col = (int) (gi - (size_t) p * nc);
                                 const int r   = perm[p];
                                 const int t   = r / Ki;
                                 xs[gi]        = x[t * s12 + (r - t * Ki) * s11 + col];
                             });
    }
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(xs, q_a.get(), (int) ne10, R, padded, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(xs, q_a.get(), (int) ne10, R, padded, stream);
    }
    if (grouped) {
        ggml_sycl_mul_mat_vec_q_id_grouped(src0->type, use_reorder, src0->data, q_a.get(), grp, dt, (int) ne10,
                                           (int) nrows, R, src0->nb[2], nrows * sizeof(float), qrow, stream);
    } else if (use_reorder) {
        ggml_sycl_mul_mat_vec_q_id_reorder(src0->type, src0->data, q_a.get(), rid, dt, (int) ne10, (int) nrows, R,
                                           src0->nb[2], nrows * sizeof(float), qrow, stream);
    } else {
        ggml_sycl_mul_mat_vec_q_id(src0->type, src0->data, q_a.get(), rid, dt, (int) ne10, (int) nrows, R,
                                   src0->nb[2], nrows * sizeof(float), qrow, stream);
    }
    {
        float *       d   = (float *) dst->data;
        const size_t  d1  = dst->nb[1] / sizeof(float);
        const size_t  d2  = dst->nb[2] / sizeof(float);
        const int     nr  = (int) nrows;
        const size_t  tot = (size_t) R * nr;
        constexpr int WG  = 256;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((tot + WG - 1) / WG * WG), sycl::range<1>(WG)),
                             [=](sycl::nd_item<1> it) {
                                 const size_t gi = it.get_global_id(0);
                                 if (gi >= tot) {
                                     return;
                                 }
                                 const int p   = (int) (gi / nr);
                                 const int row = (int) (gi - (size_t) p * nr);
                                 const int r   = perm[p];
                                 const int t   = r / Ki;
                                 d[t * d2 + (r - t * Ki) * d1 + row] = dt[gi];
                             });
    }
    return true;
}

// A quantized weight product of 2..GGML_SYCL_ROW_EXACT_MAX columns in a row-exact graph (g_ggml_sycl_row_exact):
// each column (MUL_MAT) or token (MUL_MAT_ID) as the one-token decode computes it, through a one-column view of the
// activation and the result: the one-column DMMV (ggml_sycl_dmmv_multi) when the decode takes it, else the per-node
// mat-vec (ggml_sycl_mul_mat), and the one-token MoE GEMV (ggml_sycl_mul_mat_id); and a row-exact attention of 5..max
// query rows as slices of 4 rows, the rows of the decode kernel (FA_DEC_MAX_Q). False, with nothing queued, when the
// node is none of these.
static bool ggml_sycl_row_exact_node(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node  = cgraph->nodes[i];
    const int     max_n = ggml_sycl_row_exact_max();
    // a single row of a tensor as a contiguous one-row tensor at row r (dims 1..3 of size 1)
    auto row_of = [](const ggml_tensor * t, int dim, int64_t r) {
        ggml_tensor v = *t;
        v.data        = (char *) t->data + r * t->nb[dim];
        v.ne[dim]     = 1;
        for (int d = dim + 1; d < GGML_MAX_DIMS; ++d) {
            v.nb[d] = v.nb[d - 1] * v.ne[d - 1];
        }
        return v;
    };
    if (node->op == GGML_OP_MUL_MAT) {
        const ggml_tensor * w = node->src[0];
        const ggml_tensor * x = node->src[1];
        const int64_t       n = x->ne[1];
        if (n < 2 || n > max_n || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_quantized(w->type) ||
            x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) ||
            node->nb[0] != sizeof(float) || ggml_backend_buffer_is_sycl_split(w->buffer) ||
            ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            return false;
        }
        // a product read by a GLU: the decode fuses gate, up and GLU (not mirrored here)
        for (int k = i + 1; k < cgraph->n_nodes && k <= i + 4; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (t->op == GGML_OP_GLU && (t->src[0] == node || t->src[1] == node)) {
                return false;
            }
        }
        // GGML_SYCL_ROW_EXACT_NC=0: one launch per column (else the columns of a one-column DMMV share a launch)
        static const bool nc = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT_NC", 1) != 0;
        const size_t      ys = x->nb[1] / sizeof(float);
        const size_t      ds = node->nb[1] / sizeof(float);
        for (int64_t c = 0; c < n; ++c) {
            ggml_tensor xc = row_of(x, 1, c);
            xc.nb[1] = xc.nb[2] = xc.nb[3] = xc.ne[0] * sizeof(float);
            ggml_tensor dc = row_of(node, 1, c);
            dc.nb[1] = dc.nb[2] = dc.nb[3] = dc.ne[0] * sizeof(float);
            dc.src[1] = &xc;
            const int ncol = nc && c == 0 && x->nb[1] % sizeof(float) == 0 && node->nb[1] % sizeof(float) == 0 ? (int) n : 1;
            bool done = false;
            if (ggml_sycl_dmmv_multi_eligible(ctx, &dc)) {
                opt_for_reorder(&ctx, w, &xc, &dc, mul_mat_algo::DMMV);
                const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(w->extra);
                if (extra && extra->optimized_feature.reorder) {
                    ggml_sycl_dmmv_multi_args args = {};
                    ggml_sycl_dmmv_multi_seg & sg = args.seg[args.nseg++];
                    sg.w     = w->data;
                    sg.dst   = (float *) dc.data;
                    sg.nrows = (int) w->ne[1];
                    sg.type  = w->type;
                    args.ncol       = ncol;
                    args.y_stride   = ys;
                    args.dst_stride = ds;
                    done = ggml_sycl_dmmv_multi(args, (const float *) xc.data, (int) xc.ne[0], ctx.stream());
                    if (done && ncol > 1) {
                        return true;
                    }
                }
            }
            if (!done) {
                // the one-column ESIMD DMMV of the per-node path takes every column when it runs (dmmv.hpp)
                g_ggml_sycl_dmmv_nc = { (const float *) xc.data, (float *) dc.data, ncol, ys, ds, false };
                ggml_sycl_mul_mat(ctx, w, &xc, &dc);
                const bool all = g_ggml_sycl_dmmv_nc.used;
                g_ggml_sycl_dmmv_nc = {};
                if (all && ncol > 1) {
                    return true;
                }
            }
        }
        return true;
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        // past the 4 query tokens of the decode kernel (fattn-dpas.cpp, FA_DEC_MAX_Q, whose rows do not depend on
        // the batch) the attention runs in chunks of 4 tokens through it: views of Q, of the per-head mask and of the
        // result at the chunk's tokens (the sinks are per head)
        constexpr int64_t   CH   = 4;
        const ggml_tensor * q    = node->src[0];
        const ggml_tensor * mask = node->src[3];
        const int64_t       n    = q->ne[1];
        if (n <= CH || n > max_n || ggml_get_op_params_i32(node, 6) == 0 || !mask || mask->ne[1] < n ||
            node->src[5] || node->src[6] || node->src[7] || q->ne[3] != 1 || mask->ne[3] != 1 || node->ne[2] != n ||
            node->ne[3] != 1) {
            return false;
        }
        // a REL_BIAS mask packed for the decode kernel (it takes this row-exact op whole, GGML_SYCL_FA_DEC_ROWEXACT)
        // is no per-row f16 mask that views could slice: the op runs as it is
        if (mask->op == GGML_OP_REL_BIAS && ggml_sycl_flash_attn_ext_dpas_dec_rel_ok(node, mask)) {
            return false;
        }
        for (int64_t c0 = 0; c0 < n; c0 += CH) {
            const int64_t m  = std::min(CH, n - c0);
            ggml_tensor   qv = *q;
            qv.ne[1]         = m;
            qv.data          = (char *) q->data + c0 * q->nb[1];
            ggml_tensor mv   = *mask;
            mv.ne[1]         = m;
            mv.data          = (char *) mask->data + c0 * mask->nb[1];
            ggml_tensor dv   = *node;
            dv.ne[2]         = m;
            dv.data          = (char *) node->data + c0 * node->nb[2];
            dv.src[0]        = &qv;
            dv.src[3]        = &mv;
            ggml_sycl_flash_attn_ext(ctx, &dv);
        }
        return true;
    }
    if (node->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * src1 = node->src[1];
        const ggml_tensor * ids  = node->src[2];
        const int64_t       n    = src1->ne[2];
        if (n < 2 || n > max_n || ids->ne[1] != n || src1->ne[3] != 1 || node->ne[3] != 1 || ids->ne[2] != 1 ||
            ggml_backend_buffer_is_sycl_split(node->src[0]->buffer)) {
            return false;
        }
        ggml_tensor s1[2], it[2], dt[2];
        auto token = [&](int64_t t, int k) {
            s1[k] = row_of(src1, 2, t);
            it[k] = row_of(ids, 1, t);
            dt[k] = row_of(node, 2, t);
            dt[k].src[1] = &s1[k];
            dt[k].src[2] = &it[k];
        };
        token(0, 0);
        if (!ggml_sycl_mul_mat_id_should_use_mmvq(&dt[0])) {
            return false;
        }
        // GGML_SYCL_ROW_EXACT_MOE_FLAT=0: one launch per token
        static const bool flat = ggml_sycl_get_env("GGML_SYCL_ROW_EXACT_MOE_FLAT", 1) != 0;
        if (flat && ggml_sycl_row_exact_moe_flat(ctx, node)) {
            return true;
        }
        for (int64_t t = 0; t < n; ++t) {
            token(t, 1);
            ggml_sycl_mul_mat_id(ctx, &dt[1]);
        }
        return true;
    }
    return false;
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_conv_window_match conv_window;
    auto & prof = g_ggml_sycl_node_prof;
    if (prof.enabled < 0) {
        prof.enabled   = ggml_sycl_get_env("GGML_SYCL_NODE_PROF", 0);
        prof.min_nodes = ggml_sycl_get_env("GGML_SYCL_NODE_PROF_MIN_NODES", 1000);
    }
    bool prof_on = prof.enabled != 0;
    {
        // GGML_SYCL_GRAPH_DUMP=<min nodes>: print the nodes of the first graph of at least that many nodes (diagnostic)
        static int dump_min = ggml_sycl_get_env("GGML_SYCL_GRAPH_DUMP", 0);
        if (dump_min > 0 && cgraph->n_nodes >= dump_min) {
            dump_min = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "[gd] %5d %-14s %-28s [%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu]%s", i, ggml_op_desc(n), n->name,
                        (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        n->nb[1], n->nb[2], n->nb[3], ggml_is_contiguous(n) ? "" : " nc");
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (n->src[j]) {
                        fprintf(stderr, " | %s", n->src[j]->name);
                    }
                }
                fprintf(stderr, "\n");
            }
        }
    }
#ifdef GGML_SYCL_GRAPH
    // a queue that is recording a SYCL graph cannot be waited on
    prof_on = prof_on && sycl_ctx->stream()->ext_oneapi_get_state() !=
                             sycl::ext::oneapi::experimental::queue_state::recording;
#endif
#if GGML_SYCL_DNNL
    ggml_sycl_dense_dnn_graph_begin(*sycl_ctx);
    int dense_dnn_checked = 0; // nodes before this one have been checked for writes into converted activations
    g_ggml_sycl_drop_f32_index.graph = nullptr; // the same cgraph object may hold a new graph
    // the f32 rows of the result of the fused nodes [i, last] as a 2D tensor t: their last reader, or -1 (keep them)
    auto drop_f32_pin = [&](int last, const ggml_tensor * t) {
        int64_t M = t->ne[1];
        int64_t ld = t->nb[1] / sizeof(float);
        if (t->type != GGML_TYPE_F32 || t->ne[3] != 1 || t->nb[0] != sizeof(float) || t->nb[1] % sizeof(float) != 0 ||
            t->ne[2] != 1) {
            return -1;
        }
        return ggml_sycl_drop_f32_last_reader(*sycl_ctx, cgraph, last, ggml_sycl_root(t), t->data, t->ne[0], M, ld);
    };
#endif
    // nodes already run by a fusion that moved them forward (ggml_sycl_dmmv_multi_try)
    static const bool dmmv_multi = ggml_sycl_get_env("GGML_SYCL_DMMV_MULTI", 1) != 0;
    std::vector<uint8_t> moved_done(dmmv_multi ? cgraph->n_nodes : 0, 0);
    const bool row_exact = ggml_sycl_row_exact_graph(cgraph);
    g_ggml_sycl_row_exact = row_exact;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
#if GGML_SYCL_DNNL
        ggml_sycl_dense_dnn_set_node(*sycl_ctx, i);
        // the nodes a fusion skipped are checked here too
        for (; dense_dnn_checked <= i; ++dense_dnn_checked) {
            const ggml_tensor * n = cgraph->nodes[dense_dnn_checked];
            if (!ggml_sycl_is_view_or_noop(n)) {
                ggml_sycl_dense_dnn_node_writes(*sycl_ctx, n);
            }
        }
#endif
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (dmmv_multi && moved_done[i]) {
            continue;
        }
        if (prof_on) {
            prof.mark(sycl_ctx->stream(), node);
        }
        if (row_exact &&
            (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_FLASH_ATTN_EXT) &&
            ggml_sycl_row_exact_node(*sycl_ctx, cgraph, i)) {
            continue;
        }
        if (dmmv_multi && node->op == GGML_OP_MUL_MAT && ggml_sycl_dmmv_multi_try(*sycl_ctx, cgraph, i, moved_done)) {
            continue;
        }

        // Run the window and cache updates at the convolution.
        if (conv_window.concat == nullptr && node->op == GGML_OP_CONCAT &&
            ggml_sycl_match_conv_window(cgraph, i, conv_window)) {
            continue;
        }
        if (conv_window.concat != nullptr) {
            if (std::find(conv_window.cont, conv_window.cont + conv_window.n_slots, node) != conv_window.cont + conv_window.n_slots ||
                std::find(conv_window.cpy, conv_window.cpy + conv_window.n_slots, node) != conv_window.cpy + conv_window.n_slots) {
                continue;
            }
            if (node == conv_window.conv) {
                ggml_sycl_ssm_conv_window(*sycl_ctx, conv_window);
                i           = conv_window.last_idx;
                conv_window = {};
                continue;
            }
        }

        // a REL_BIAS node whose flash attention computes the bias in its kernel (GGML_SYCL_FA_REL_INKERNEL)
        if (node->op == GGML_OP_REL_BIAS && ggml_sycl_rel_bias_fa_pack_node(*sycl_ctx, cgraph, i)) {
            continue;
        }
        if (node->op == GGML_OP_UNARY) {
            const int n_route = ggml_sycl_route_fuse(*sycl_ctx, cgraph, i);
            if (n_route != 0) {
                i += n_route;
                continue;
            }
        }
        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            i += nodes_to_skip;
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            const int gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            if (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, node, fused_state_cpy);
                i += gdn_nodes_to_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, {}) &&
            ggml_sycl_op_add_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
            i += 2;
            continue;
        }
#if GGML_SYCL_DNNL
        // the gains|forgets matmul, the mix, the norm and its weight as one kernel (GGML_SYCL_GATE_MIX_FUSED, see
        // gate-mix-mm.hpp), at the batch sizes of the oneDNN path below
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_gate_mix_mm_match(*sycl_ctx, cgraph, i) &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GATE_MIX, GGML_OP_RMS_NORM, GGML_OP_MUL }, {}) &&
            !ggml_backend_buffer_is_sycl_split(node->src[0]->buffer) &&
            ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
            ggml_sycl_dense_dnn_converts_src1(sycl_ctx->device, node) && ggml_sycl_gate_mix_mm_ready(*sycl_ctx, cgraph, i)) {
            // an f16 copy of z in the cache is copied out first: the result's f16 copy may take its slot
            ggml_sycl_pool_alloc<sycl::half> zc(sycl_ctx->pool());
            const void * z16 = ggml_sycl_gate_mix_mm_z16(*sycl_ctx, cgraph, i);
            if (z16) {
                const size_t nz = (size_t) node->src[1]->ne[0] * ggml_nrows(node->src[1]);
                zc.alloc(nz);
                sycl_ctx->stream()->memcpy(zc.get(), z16, nz * sizeof(sycl::half));
                z16 = zc.get();
            }
            // node i has read its activation: the fused writes below may reuse its bytes
            ggml_sycl_dense_dnn_set_node(*sycl_ctx, i + 1);
            for (; dense_dnn_checked <= i + 3; ++dense_dnn_checked) {
                ggml_sycl_dense_dnn_node_writes(*sycl_ctx, cgraph->nodes[dense_dnn_checked]);
            }
            const bool f16_for_dense = ggml_sycl_dense_dnn_reads_f16(*sycl_ctx, cgraph, i + 3);
            const int  pin = f16_for_dense ? drop_f32_pin(i + 3, cgraph->nodes[i + 3]) : -1;
            ggml_sycl_gate_mix_mm_run(*sycl_ctx, cgraph, i, z16, f16_for_dense, pin);
            i += 3;
            continue;
        }
#endif
#if GGML_SYCL_DNNL
        // the gains|forgets matmul of a mix: oneDNN writes tanh(gains|forgets) as f16 into a scratch buffer, which
        // the fused mix + norm reads instead of the f32 result (never written)
        static const bool gate_mm_f16 = ggml_sycl_get_env("GGML_SYCL_GATE_MIX_MM_F16", 1) != 0;
        if (gate_mm_f16 && node->op == GGML_OP_MUL_MAT && i + 3 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_GATE_MIX &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GATE_MIX, GGML_OP_RMS_NORM, GGML_OP_MUL }, {}) &&
            !ggml_backend_buffer_is_sycl_split(node->src[0]->buffer) &&
            ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
            ggml_sycl_dense_dnn_converts_src1(sycl_ctx->device, node) &&
            ggml_sycl_op_gate_mix_rms_norm_mul(*sycl_ctx, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                                               cgraph->nodes[i + 3], false, nullptr, true)) {
            ggml_sycl_pool_alloc<sycl::half> gates(sycl_ctx->pool(), ggml_nelements(node));
            if (ggml_sycl_dense_dnn_mul_mat_f16(*sycl_ctx, node, gates.get(), true)) {
                // node i has read its activation: the fused writes below may reuse its bytes
                ggml_sycl_dense_dnn_set_node(*sycl_ctx, i + 1);
                for (; dense_dnn_checked <= i + 3; ++dense_dnn_checked) {
                    ggml_sycl_dense_dnn_node_writes(*sycl_ctx, cgraph->nodes[dense_dnn_checked]);
                }
                const bool f16_for_dense = ggml_sycl_dense_dnn_reads_f16(*sycl_ctx, cgraph, i + 3);
                const int  pin = f16_for_dense ? drop_f32_pin(i + 3, cgraph->nodes[i + 3]) : -1;
                const bool ok = ggml_sycl_op_gate_mix_rms_norm_mul(*sycl_ctx, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                                                                   cgraph->nodes[i + 3], f16_for_dense, gates.get(),
                                                                   false, pin);
                GGML_ASSERT(ok);
                i += 3;
                continue;
            }
        }
#endif
        if (node->op == GGML_OP_GATE_MIX &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_GATE_MIX, GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            bool f16_for_dense = false;
#if GGML_SYCL_DNNL
            // the fused nodes write their bytes now: drop the converted copies they overlap before the kernel
            // registers its own f16 copy of the result
            for (; dense_dnn_checked <= i + 2; ++dense_dnn_checked) {
                ggml_sycl_dense_dnn_node_writes(*sycl_ctx, cgraph->nodes[dense_dnn_checked]);
            }
            f16_for_dense = ggml_sycl_dense_dnn_reads_f16(*sycl_ctx, cgraph, i + 2);
            const int pin = f16_for_dense ? drop_f32_pin(i + 2, cgraph->nodes[i + 2]) : -1;
#else
            const int pin = -1;
#endif
            if (ggml_sycl_op_gate_mix_rms_norm_mul(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                                                   f16_for_dense, nullptr, false, pin)) {
                i += 2;
                continue;
            }
        }
#if GGML_SYCL_DNNL
        // a mix whose rows feed dense oneDNN matmuls writes their f16 copies (its own writes were checked above)
        if (node->op == GGML_OP_BLOCK_MIX && node->ne[1] <= 4) {
            bool f16[4] = {};
            if (ggml_sycl_block_mix_dense_reads(*sycl_ctx, cgraph, i, f16)) {
                // plane-0 rows of query j: [ne0, n_tokens] at row stride nb[2]
                int pin[4] = { -1, -1, -1, -1 };
                if (node->ne[3] >= 1 && node->nb[2] % sizeof(float) == 0) {
                    for (int j = 0; j < node->ne[1]; ++j) {
                        if (f16[j]) {
                            pin[j] = ggml_sycl_drop_f32_last_reader(*sycl_ctx, cgraph, i, node,
                                                                    (const char *) node->data + j*node->nb[1],
                                                                    node->ne[0], node->ne[2], node->nb[2] / sizeof(float));
                        }
                    }
                }
                ggml_sycl_block_mix(*sycl_ctx, node, f16, pin);
                continue;
            }
        }
#endif
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
#if GGML_SYCL_DNNL
            if (ggml_sycl_dense_dnn_reads_f16(*sycl_ctx, cgraph, i + 1)) {
                // as for GATE_MIX above: the pair's writes are checked before the kernel registers its f16 copy
                for (; dense_dnn_checked <= i + 1; ++dense_dnn_checked) {
                    ggml_sycl_dense_dnn_node_writes(*sycl_ctx, cgraph->nodes[dense_dnn_checked]);
                }
                if (ggml_sycl_op_rms_norm_mul_f16(*sycl_ctx, node, cgraph->nodes[i + 1],
                                                  drop_f32_pin(i + 1, cgraph->nodes[i + 1]))) {
                    i++;
                    continue;
                }
            }
#endif
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(node) })) {
#if GGML_SYCL_DNNL
            if (ggml_sycl_dense_dnn_reads_f16(*sycl_ctx, cgraph, i + 1)) {
                // as for GATE_MIX: the pair's writes are checked before its f16 copy is registered
                for (; dense_dnn_checked <= i + 1; ++dense_dnn_checked) {
                    ggml_sycl_dense_dnn_node_writes(*sycl_ctx, cgraph->nodes[dense_dnn_checked]);
                }
                if (ggml_sycl_op_unary_mul_fused_f16(*sycl_ctx, node, cgraph->nodes[i + 1],
                                                     drop_f32_pin(i + 1, cgraph->nodes[i + 1]))) {
                    i++;
                    continue;
                }
            }
#endif
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        if (node->op == GGML_OP_L2_NORM) {
            const int l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            if (l2_batch_skip > 0) {
                i += l2_batch_skip;
                continue;
            }
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, nullptr, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

#if GGML_SYCL_DNNL
        if (node->op == GGML_OP_CONCAT) {
            const int skip = ggml_sycl_dense_dnn_concat_mul_mat(*sycl_ctx, cgraph, i);
            if (skip > 0) {
                i += skip;
                continue;
            }
        }
#endif

        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int moe_skip = ggml_sycl_moe_block_fused(*sycl_ctx, cgraph, i);
            if (moe_skip > 0) {
                i += moe_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_GLU && ggml_sycl_mul_mat_id_situ_fused(*sycl_ctx, cgraph, i)) {
            i++;
            continue;
        }

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
    g_ggml_sycl_row_exact = false;
    if (prof_on) {
        prof.mark(sycl_ctx->stream(), nullptr);
        prof.report(cgraph->n_nodes);
    }
}

#ifdef GGML_SYCL_GRAPH
static const void * ggml_sycl_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_sycl_graph_update_required(ggml_sycl_graph * graph, ggml_cgraph * cgraph) {
    bool res = false;

    if (cgraph->uid != 0 && cgraph->uid == graph->uid) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] graph id %" PRIu64 " reused\n", cgraph->uid);
        GGML_ASSERT((int) graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    if ((int) graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_sycl_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

// Reports if ggml_sycl_mul_mat() gives this node to oneDNN. oneDNN chains the submission on events
// made before recording started, which SYCL graphs do not allow.
static bool mul_mat_uses_onednn(int device, ggml_tensor * dst) {
#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_enable_dnn) {
        return false;
    }

    if (ggml_sycl_dense_dnn_eligible(device, dst)) {
        return true;
    }

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // the branch order below follows the dispatch in ggml_sycl_mul_mat(); the F16 branches end in
    // a vector kernel or the batched oneMKL path, neither of which uses oneDNN
    if (!ggml_backend_buffer_is_sycl_split(src0->buffer) && src0->type == GGML_TYPE_F16) {
        if (ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
            return false;
        }
        if (!ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
            return false;
        }
        if (!ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
            return false;
        }
    }

    // ggml_sycl_op_mul_mat_sycl() is the last resort of the dispatch and the only user of oneDNN.
    // The reorder decision that ggml_sycl_mul_mat() makes between DMMV and MMVQ is left out here:
    // neither uses a library.
    if (can_use_dequantize_mul_mat_vec(src0, src1, dst) || can_use_mul_mat_vec_q(device, src0, src1, dst) ||
        can_use_mul_mat_q(src0, src1, dst)) {
        return false;
    }

    // the branch order below follows ggml_sycl_op_mul_mat_sycl(); graphs are single device only, so
    // row_diff == src0->ne[1] and src1_ncols == src1->ne[1] there
#if defined(GGML_SYCL_HAS_BF16)
    if (src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0)) {
        return true;
    }
#endif
#ifdef GGML_SYCL_F16
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) &&
        ggml_prec_get_acc(dst) == GGML_PREC_UNDEFINED) {
        return true;
    }
#endif
    // the F32 branch only uses oneDNN for large GEMMs and calls oneMKL directly otherwise
    const int64_t gemm_flops = src0->ne[1] * src1->ne[1] * src0->ne[0];
    return gemm_flops >= SYCL_F32_DNNL_MIN_GEMM_FLOPS;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

// returns true when ggml_sycl_mul_mat_id takes the fallback path that requires stream synchronization:
// neither the fused MMVQ path nor the device-scheduled grouped GEMM (GGML_SYCL_MMID_SCHED) handles dst
static bool ggml_sycl_mul_mat_id_needs_sync(ggml_backend_sycl_context * ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    if (!src0 || !src1 || !ids || ggml_backend_buffer_is_sycl_split(src0->buffer)) {
        return true;
    }
    const bool reordered = src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
    if (ggml_sycl_mul_mat_id_mmvq_eligible(src0, src1, ids, dst)) {
        return reordered ? !ggml_sycl_mul_mat_vec_q_id_reorder_supports_type(src0->type) :
                           !ggml_sycl_mul_mat_vec_q_id_supports_type(src0->type);
    }
    if (ggml_sycl_mul_mat_id_mc_eligible(dst) && ggml_sycl_mmid_mc_supports_type(src0->type, reordered) &&
        (src1->ne[2] <= g_ggml_sycl_mmid_mc_max ||
         (ggml_sycl_xmx_enabled(ctx->device) &&
          ggml_sycl_mmid_mc_xmx_supported(src0->type, reordered, (int) src0->ne[0], src0->nb[2], src0->data)))) {
        return false;
    }
    return !ggml_sycl_mul_mat_id_device_sched_eligible(dst) ||
           !ggml_sycl_grouped_dequant_gemm_device_supported(src0->type, reordered, ggml_sycl_src1_prec(dst),
               src0->ne[2], ids->ne[0], ids->ne[1], src0->ne[1], src0->ne[0], ctx->stream());
}

// mirrors the paths in ggml_sycl_mul_mat_id() that call opt_for_reorder_id()
static bool mul_mat_id_reorders_src0(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    if (ggml_sycl_mul_mat_id_should_use_mmvq(dst) || ggml_sycl_mul_mat_id_mc_eligible(dst) ||
        src0->type == GGML_TYPE_MXFP4) {
        return true;
    }
    return ggml_sycl_mul_mat_id_device_sched_eligible(dst) &&
           g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           ggml_sycl_xmx_gather_type_enabled(src0->type);
}

// true when the node and its SYCL sources live in this device's buffers (no split or other-device buffers)
static bool graph_node_is_in_local_device(ggml_backend_buffer_type_t device_buft, const ggml_tensor * node) {
    if (!node->buffer || node->buffer->buft != device_buft) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * src = node->src[j];
        if (!src || !src->buffer) {
            continue;
        }
        if (ggml_backend_buffer_is_sycl_split(src->buffer) ||
            (ggml_backend_buffer_is_sycl(src->buffer) && src->buffer->buft != device_buft)) {
            return false;
        }
    }
    return true;
}

static bool check_graph_compatibility(ggml_backend_sycl_context * ctx, ggml_cgraph * cgraph) {
    if (ggml_sycl_info().active_device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_DEBUG("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }

    const ggml_backend_buffer_type_t device_buft = ggml_backend_sycl_buffer_type(ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // skip the nodes that ggml_backend_sycl_graph_compute_impl() does not run
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_sycl_is_view_or_noop(node)) {
            continue;
        }

        const ggml_op node_op = node->op;

        if (!graph_node_is_in_local_device(device_buft, node)) {
            GGML_LOG_DEBUG("%s: disabling SYCL graphs due to node %s not being device local\n", __func__,
                           ggml_op_name(node_op));
            return false;
        }

        // Reorder expert weights before recording, so the reorder state that
        // ggml_sycl_mul_mat_id_needs_sync() sees is the one the recorded graph runs with.
        if (node_op == GGML_OP_MUL_MAT_ID && mul_mat_id_reorders_src0(node)) {
            opt_for_reorder_id(ctx, node->src[0]);
        }

        switch (node_op) {
            default:
                break;
            case GGML_OP_MUL_MAT_ID:
                // The ggml_sycl_mul_mat_id() fallback path does a blocking host wait on the sycl
                // queue after submitting a memcpy operation, but wait() can't be called on a queue
                // that is recording to a graph.
                if (ggml_sycl_mul_mat_id_needs_sync(ctx, node)) {
                    GGML_LOG_DEBUG("%s: disabling SYCL graphs due to host wait in %s\n", __func__,
                                   ggml_op_name(node_op));
                    return false;
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
#ifndef GGML_SYCL_GRAPH_DNNL_SUPPORTED
                if (ggml_sycl_flash_attn_ext_uses_onednn(ctx->device, node)) {
                    GGML_LOG_DEBUG("%s: disabling SYCL graphs due to %s using oneDNN\n", __func__,
                                   ggml_op_name(node_op));
                    return false;
                }
#endif
                break;
            case GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_DEBUG(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
#ifndef GGML_SYCL_GRAPH_DNNL_SUPPORTED
                if (mul_mat_uses_onednn(ctx->device, node)) {
                    GGML_LOG_DEBUG("%s: disabling SYCL graphs due to %s using oneDNN\n", __func__,
                                   ggml_op_name(node_op));
                    return false;
                }
#endif
                break;
            case GGML_OP_OUT_PROD:
                if (ggml_sycl_out_prod_direct(node)) {
                    break;
                }
                [[fallthrough]];
            case GGML_OP_CONV_3D:
            case GGML_OP_SOLVE_TRI:
                // these call oneMKL and wait on the host
                GGML_LOG_DEBUG("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                               ggml_op_name(node_op));
                return false;
        }
    }
    return true;
}
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    // set the device on every call: a graph replay does not go through
    // ggml_backend_sycl_graph_compute_impl()
    ggml_sycl_set_main_device(sycl_ctx->device);
#if GGML_SYCL_DNNL
    // the packed weights of the fused gate kernel, before anything runs or records (see gate-mix-mm.hpp)
    ggml_sycl_gate_mix_mm_prepare(*sycl_ctx, cgraph);
#endif

#ifdef GGML_SYCL_GRAPH
    bool              use_sycl_graph = false;
    ggml_sycl_graph * graph          = nullptr;

    if (g_ggml_sycl_enable_graph) {
        if (!ggml_sycl_info().devices[sycl_ctx->device].graph_support) {
            // the device does not change, so report this once
            if (sycl_ctx->graph_compatible != false) {
                GGML_LOG_DEBUG("%s: disabling SYCL graphs due to unsupported graph feature on device\n", __func__);
                GGML_LOG_INFO("%s: disabling SYCL graphs\n", __func__);
                sycl_ctx->graph_compatible = false;
            }
        } else {
            graph = sycl_ctx->sycl_graph(ggml_sycl_graph_get_key(cgraph));

            // The nodes of a cgraph do not change while its uid does not, so the compatibility scan
            // runs again only when the uid changes. A uid of 0 means it is not set.
            if (cgraph->uid == 0 || cgraph->uid != graph->compatible_uid) {
                graph->compatible_uid = cgraph->uid;
                graph->compatible     = check_graph_compatibility(sycl_ctx, cgraph);

                if (sycl_ctx->graph_compatible != graph->compatible) {
                    if (graph->compatible) {
                        GGML_LOG_INFO("%s: enabling SYCL graphs\n", __func__);
                    } else {
                        GGML_LOG_INFO("%s: disabling SYCL graphs\n", __func__);
                    }
                    sycl_ctx->graph_compatible = graph->compatible;
                }
            }

            use_sycl_graph = graph->compatible;
        }
    }
    if (use_sycl_graph) {
        const bool properties_changed = ggml_sycl_graph_update_required(graph, cgraph);
        bool replay_graph = false;
        bool sycl_graph_update_required = false;

        if (!graph->warmup_complete) {
            // Warmup: need at least 2 calls with no property change on the 2nd call.
            if (!properties_changed) {
                graph->warmup_complete = true;
                replay_graph = true;
                sycl_graph_update_required = true;
                GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup complete\n");
            }
        } else {
            // Post-warmup: replay graph while properties remain stable.
            if (properties_changed) {
                graph->warmup_complete = false;
                GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup reset\n");
            } else {
                replay_graph = true;
                sycl_graph_update_required = graph->exec_graph == nullptr;
            }
        }

        if (replay_graph) {
            if (sycl_graph_update_required) {
                const bool graph_update_support = ggml_sycl_info().devices[sycl_ctx->device].graph_update_support;

                sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

                model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
                model_sycl_graph.end_recording();

                if (!graph->exec_graph || !graph_update_support) {
                    auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                             model_sycl_graph.finalize();
                    graph->exec_graph = std::make_unique<
                        sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
                } else {
                    try {
                        graph->exec_graph->update(model_sycl_graph);
                        GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
                    } catch (sycl::exception const & e) {
                        // the executable graph cannot be updated due to violated constraints,
                        // so finalize the recorded graph again
                        GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                        auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                        graph->exec_graph = std::make_unique<
                            sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
                    }
                }
            }

            sycl_ctx->stream()->ext_oneapi_graph(*(graph->exec_graph));
        } else {
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        }
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph,
                                              ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_sycl_conv_window_match match;
        if (cgraph->nodes[i]->op != GGML_OP_CONCAT || !ggml_sycl_match_conv_window(cgraph, i, match)) {
            continue;
        }
        // Inputs and delayed cache writes must stay allocated until the fused kernel.
        for (ggml_tensor * t : { match.concat->src[0], match.concat->src[1], match.cpy[0]->src[1] }) {
            params->add_alloc_dep(params->user_data, t->view_src ? t->view_src : t, match.silu);
        }
        i = match.last_idx;
    }
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ggml_backend_sycl_graph_optimize,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count(void) {
    return ggml_sycl_info().device_count;
}


// backend device

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    if (!sycl_get_mem_info(ctx->device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *free);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *free, *total);
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    const sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    if (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ true,
        /* .fast_async_host_input = */ false,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    return ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                case GGML_GLU_OP_SITU:
                    return ggml_sycl_situ_supported(op);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                if (op->op == GGML_OP_MUL_MAT &&
                    (ggml_prec_get_src(op, 1) == GGML_PREC_BF16 || ggml_prec_get_src(op, 1) == GGML_PREC_F32)) {
                    return false;
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            // return true;
             return op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                if (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    return false;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                return res;
            }
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                return true;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
            return true;
        case GGML_OP_L2_CLIP:
            return ggml_sycl_l2_clip_supported(op);
        case GGML_OP_GATE_MIX:
            return ggml_sycl_gate_mix_supported(op);
        case GGML_OP_GATE_MIX_MV:
            return ggml_sycl_gate_mix_mv_supported(op);
        case GGML_OP_RMS_NORM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            // the scale kernel is f32 only, and reads src0 flat
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_CONT:
            return true;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D: {
            const int32_t * opts = (const int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            const int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            const int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            // replay of pending tokens (op param 1) is not implemented
            return ggml_get_op_params_i32(op, 1) == 0;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SSM_CONV_HIST:
            return ggml_sycl_ssm_conv_hist_supported(op);
        case GGML_OP_BLOCK_MIX:
            return ggml_sycl_block_mix_supported(op);
        case GGML_OP_REL_BIAS:
            return ggml_sycl_rel_bias_supported(op);
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            // K/V row lists and the indexed bias added to a mask are not supported, the one in place of the mask
            // (ggml_flash_attn_ext_set_rel_bias) is
            return op->src[5] == nullptr && (!ggml_flash_attn_ext_has_indexed_bias(op) || ggml_flash_attn_ext_has_rel_bias(op)) &&
                   ggml_sycl_flash_attn_ext_supported(device, op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_out[2] = { nullptr, nullptr };
    uint8_t * host_in[2]  = { nullptr, nullptr };  // only used by the 4-buffer path (separate SYCL contexts)
    size_t    host_bytes  = 0;

    // SYCL_ENABLE_DEFAULT_CONTEXTS=1 sets comm_exchange_fast on Linux - PR #30956
    using exchange_fn_t = void (*)(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                   void * dst0, const void * src1, void * dst1, const void * src0,
                                   size_t nbytes);
    exchange_fn_t comm_exchange = nullptr; // Moves same context check out of hot path; tests only once

    void free_host() {
        for (int i = 0; i < 2; ++i) {
            const sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            if (host_out[i]) { sycl::free(host_out[i], sctx); host_out[i] = nullptr; }
            if (host_in[i])  { sycl::free(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory.
static void ggml_sycl_comm_exchange_fast(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                         void * dst0, const void * src1, void * dst1, const void * src0,
                                         size_t nbytes) {

    // Increases Linux tensor split prompt eval speeds on par with layer split speeds
    // Reallocate host buffers if they are too small
    if (comm->host_bytes < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());

        GGML_ASSERT(comm->host_out[0] && comm->host_out[1]);
        comm->host_bytes = nbytes;
    }

    // Step 1: Concurrently copy from GPUs into host buffers (Asynchronous)
    sycl::event e_d2h_0 = q0->memcpy(comm->host_out[0], src0, nbytes);
    sycl::event e_d2h_1 = q1->memcpy(comm->host_out[1], src1, nbytes);

    // Step 2: Asynchronous Barrier Interlocking (NO HOST BLOCKING), wait() causes MTP conflicts
    q0->ext_oneapi_submit_barrier({e_d2h_1}); // q0 waits for q1's device-to-host copy
    q1->ext_oneapi_submit_barrier({e_d2h_0}); // q1 waits for q0's device-to-host copy

    // Step 3: Cross-upload directly from the opposing host buffer.
    sycl::event e_h2d_0 = q0->memcpy(dst0, comm->host_out[1], nbytes); // q0 pulls from host_out[1]
    sycl::event e_h2d_1 = q1->memcpy(dst1, comm->host_out[0], nbytes); // q1 pulls from host_out[0]

    // Step 4: Finalize the exchange before returning to host code
    e_h2d_0.wait();
    e_h2d_1.wait();
}

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
static void ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    void * dst0, const void * src1, void * dst1, const void * src0,
                                    size_t nbytes) {
    if (comm->host_bytes < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_in[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_in[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_out[0] && comm->host_in[0] && comm->host_out[1] && comm->host_in[1]);
        comm->host_bytes = nbytes;
    }

    sycl::event e0 = q0->memcpy(comm->host_out[0], src0, nbytes);
    sycl::event e1 = q1->memcpy(comm->host_out[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::memcpy(comm->host_in[0], comm->host_out[1], nbytes);
    std::memcpy(comm->host_in[1], comm->host_out[0], nbytes);
    q0->memcpy(dst0, comm->host_in[0], nbytes);
    q1->memcpy(dst1, comm->host_in[1], nbytes);
}

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    if (n_backends != 2) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());

#if defined(_WIN32) || defined(_WIN64) // Not supported on Windows
    ctx->comm_exchange = ggml_sycl_comm_exchange;
#else
    // SYCL_ENABLE_DEFAULT_CONTEXTS=1 sets comm_exchange_fast on Linux - PR #30956
    ctx->comm_exchange = (sctx0->stream()->get_context() == sctx1->stream()->get_context())
                       ? ggml_sycl_comm_exchange_fast
                       : ggml_sycl_comm_exchange;
#endif
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    if (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        try {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } catch (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (n_backends != 2) {
        return false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    if (tensors[0]->type != tensors[1]->type) {
        return false;
    }
    if (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        return false;
    }
    if (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        return false;
    }

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        comm_ctx->comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        float * tmp0 = (float *) buf0;
        float * tmp1 = (float *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        comm_ctx->comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->data, tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    comm_ctx->comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

static void set_device_active(int device) {
    ggml_sycl_device_info & info = ggml_sycl_info();
    info.devices[device].active = true;
    int cnt = 0;
    for (int i = 0; i < info.device_count; i++) {
        if (info.devices[i].active)
            cnt++;
    }
    info.active_device_count = cnt;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init. device=%d\n", device);
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    set_device_active(device);
    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
