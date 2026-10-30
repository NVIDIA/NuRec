// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * LD_PRELOAD shim that makes aarch64 GPUs whose real compute capability CUDA
 * 12.8 does not recognise behave like the nearest lower arch in the same family
 * that runs them via CUDA forward compatibility: DGX Spark / GB10 (SM 12.1) ->
 * SM 12.0, and GB300 Blackwell Ultra (SM 10.3) -> SM 10.0. The shim caps two
 * independent paths:
 *
 *   1. nvrtc compile path: rewrite the rejected arch tokens (sm_121/compute_121,
 *      sm_103/compute_103) to their lower counterparts in nvrtcCompileProgram
 *      options (PyTorch/Triton JIT). PyTorch dlopen's libnvrtc RTLD_LOCAL, so
 *      we intercept dlsym to reach the real nvrtcCompileProgram.
 *
 *   2. runtime device-capability query: clamp the reported compute capability
 *      (12.1 -> 12.0, 10.3 -> 10.0) in cudaGetDeviceProperties_v2 /
 *      cudaDeviceGetAttribute. This is needed because torch's SDPA hardware
 *      gate (and other consumers) read the device capability via
 *      at::cuda::getCurrentDeviceProperties() and reject the unknown arch —
 *      torch 2.7.0+cu128 only ships flash/mem-efficient SDPA kernels up to
 *      sm120/sm100, so the uncapped device falls back to the MATH backend
 *      (materialises the O(N^2) attention matrix) and exhausts unified memory.
 *      Reporting the lower capability re-enables the flash/mem-efficient
 *      backends, whose kernels run on the newer device.
 *
 * Unlike libnvrtc, libcudart is a normal DT_NEEDED dependency of torch, so the
 * cudart wrappers below are interposed by the dynamic linker directly (no dlsym
 * trick needed); the real functions are reached via dlsym(RTLD_NEXT, ...).
 *
 * Activation is gated by the stage1 bootstrap (only on aarch64 GPUs that report
 * compute_cap 12.1 or 10.3), so the clamp is a no-op everywhere else; the
 * (major, minor) guards make it a no-op on any other GPU regardless.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h> /* struct cudaDeviceProp, cudaError_t, enum cudaDeviceAttr */
/* cudaGetDeviceProperties is a macro -> cudaGetDeviceProperties_v2; we only
 * interpose the _v2 symbol (the one torch links), so drop the macro. */
#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif

typedef unsigned int nvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;
typedef nvrtcResult (*nvrtcCompileProgram_fn)(nvrtcProgram, int, const char**);

/* Forward declaration so the dlsym interceptor can reference it. */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                const char** options);

/* ------------------------------------------------------------------ */
/* Shared state                                                         */
/* ------------------------------------------------------------------ */

typedef void* (*dlsym_fn)(void*, const char*);
static dlsym_fn _real_dlsym = NULL;

/*
 * Cached pointer to the real nvrtcCompileProgram, set the first time our
 * dlsym interceptor is called with a specific (non-pseudo) handle.
 */
static nvrtcCompileProgram_fn _real_nvrtc_compile = NULL;

/* Clamp a (major, minor) compute capability in place to the nearest lower arch
 * CUDA 12.8 accepts: 12.1 -> 12.0 (DGX Spark / GB10) and 10.3 -> 10.0 (GB300). */
static inline void sm_cap_clamp(int* major, int* minor) {
    if (!major || !minor)
        return;
    if (*major == 12 && *minor == 1)
        *minor = 0;
    else if (*major == 10 && *minor == 3)
        *minor = 0;
}

/* ------------------------------------------------------------------ */
/* dlsym interceptor                                                    */
/* ------------------------------------------------------------------ */

/*
 * Bootstrap via dlvsym — a distinct function from dlsym, no recursion risk.
 * GLIBC_2.34 is the last version bump of dlsym (libdl merged into libc.so.6)
 * and is our minimum supported glibc: glibc preserves old symvers forever, so
 * this lookup works on any glibc >= 2.34 regardless of how many further
 * version bumps land later. If it fails the system is below our minimum and
 * we abort loudly rather than let every later dlsym() return NULL and crash
 * mid-nvrtcCompileProgram.
 */
__attribute__((constructor)) static void init_real_dlsym(void) {
    _real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!_real_dlsym) {
        static const char msg[] =
            "sm_cap: ERROR: dlvsym(RTLD_NEXT, \"dlsym\", "
            "\"GLIBC_2.34\") failed — glibc too old\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        abort();
    }
}

void* dlsym(void* handle, const char* name) {
    if (strcmp(name, "nvrtcCompileProgram") == 0) {
        /*
         * When PyTorch calls dlsym(nvrtc_handle, "nvrtcCompileProgram"):
         * - nvrtc_handle is an RTLD_LOCAL library handle
         * - _real_dlsym on that handle searches only the library's local scope
         * - Our global-scope wrapper is not visible there
         * - We therefore get back the real function pointer and cache it
         *
         * Skip pseudo-handles (RTLD_DEFAULT, RTLD_NEXT) since those would
         * search the global scope and might return our own function.
         */
        if (!_real_nvrtc_compile &&
            handle != NULL &&
            handle != RTLD_DEFAULT &&
            handle != RTLD_NEXT) {
            nvrtcCompileProgram_fn candidate =
                (nvrtcCompileProgram_fn)_real_dlsym(handle, "nvrtcCompileProgram");
            /* Only cache if it differs from our own wrapper (safety guard in
             * case the library is RTLD_GLOBAL and global scope returns us). */
            if (candidate && candidate != nvrtcCompileProgram)
                _real_nvrtc_compile = candidate;
        }
        return (void*)nvrtcCompileProgram;
    }
    return _real_dlsym ? _real_dlsym(handle, name) : NULL;
}

/* ------------------------------------------------------------------ */
/* nvrtcCompileProgram interceptor                                      */
/* ------------------------------------------------------------------ */

/*
 * Resolve the real nvrtcCompileProgram if not yet cached.
 * Returns 0 on success, -1 if the function cannot be found.
 */
static int resolve_real_nvrtc(void) {
    if (_real_nvrtc_compile)
        return 0;

    /* Try RTLD_NEXT first (works when libnvrtc is in the global symbol table). */
    _real_nvrtc_compile = (nvrtcCompileProgram_fn)_real_dlsym(
        RTLD_NEXT, "nvrtcCompileProgram");
    if (_real_nvrtc_compile)
        return 0;

    /*
     * RTLD_NEXT failed: libnvrtc was loaded RTLD_LOCAL (e.g. by a C++
     * extension or PyTorch).  Use RTLD_NOLOAD to get a handle to the
     * already-loaded library without promoting it to global visibility.
     */
    static const char* const nvrtc_names[] = {"libnvrtc.so.12", NULL};
    for (int i = 0; nvrtc_names[i] && !_real_nvrtc_compile; i++) {
        void* h = dlopen(nvrtc_names[i], RTLD_LAZY | RTLD_NOLOAD);
        if (h) {
            nvrtcCompileProgram_fn candidate =
                (nvrtcCompileProgram_fn)_real_dlsym(h, "nvrtcCompileProgram");
            if (candidate && candidate != nvrtcCompileProgram)
                _real_nvrtc_compile = candidate;
            dlclose(h);
        }
    }
    return _real_nvrtc_compile ? 0 : -1;
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                const char** options) {
    if (resolve_real_nvrtc() != 0) {
        static const char msg[] =
            "sm_cap: ERROR: could not find real nvrtcCompileProgram\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        abort();
    }

    if (numOptions <= 0 || !options)
        return _real_nvrtc_compile(prog, numOptions, options);

    /*
     * Arch tokens CUDA 12.8 nvrtc rejects, mapped to the nearest lower arch in
     * the same family that the device runs via forward compatibility. Each
     * pair MUST be equal length so the rewrite can happen in place.
     *   sm_121 / compute_121 -> ..._120  (DGX Spark / GB10, SM 12.1)
     *   sm_103 / compute_103 -> ..._100  (GB300 Blackwell Ultra, SM 10.3)
     */
    static const struct {
        const char* from;
        const char* to;
        size_t len;
    } kRewrites[] = {
        {"compute_121", "compute_120", 11},
        {"compute_103", "compute_100", 11},
        {"sm_121", "sm_120", 6},
        {"sm_103", "sm_100", 6},
    };
    static const int kNumRewrites = (int)(sizeof(kRewrites) / sizeof(kRewrites[0]));

    /* Scan once to see whether any option contains a rewritable arch token. */
    int needs_rewrite = 0;
    for (int i = 0; i < numOptions && !needs_rewrite; i++) {
        if (!options[i])
            continue;
        for (int r = 0; r < kNumRewrites; r++) {
            if (strstr(options[i], kRewrites[r].from)) {
                needs_rewrite = 1;
                break;
            }
        }
    }
    if (!needs_rewrite)
        return _real_nvrtc_compile(prog, numOptions, options);

    /* Rewrite the rejected arch tokens in each affected option, then compile. */
    char** modified = (char**)malloc((size_t)numOptions * sizeof(char*));
    if (!modified)
        return _real_nvrtc_compile(prog, numOptions, options); /* OOM: best effort */
    for (int j = 0; j < numOptions; j++)
        modified[j] = (char*)options[j];
    for (int i = 0; i < numOptions; i++) {
        if (!options[i])
            continue;
        int hit = 0;
        for (int r = 0; r < kNumRewrites && !hit; r++)
            if (strstr(options[i], kRewrites[r].from))
                hit = 1;
        if (!hit)
            continue;
        char* copy = strdup(options[i]);
        if (!copy)
            continue;
        /* compute_* entries come first so an "sm_" pass never partially
         * matches inside a "compute_" token (they share no overlap anyway). */
        for (int r = 0; r < kNumRewrites; r++) {
            char* p = copy;
            while ((p = strstr(p, kRewrites[r].from)) != NULL) {
                memcpy(p, kRewrites[r].to, kRewrites[r].len);
                p += kRewrites[r].len;
            }
        }
        modified[i] = copy;
    }

    nvrtcResult result =
        _real_nvrtc_compile(prog, numOptions, (const char**)modified);

    for (int i = 0; i < numOptions; i++)
        if (modified[i] != options[i])
            free(modified[i]);
    free(modified);
    return result;
}

/* ------------------------------------------------------------------ */
/* Runtime compute-capability clamp (12.1 -> 12.0, 10.3 -> 10.0)        */
/* ------------------------------------------------------------------ */

typedef cudaError_t (*cudaGetDeviceProperties_v2_fn)(struct cudaDeviceProp*, int);
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device) {
    static cudaGetDeviceProperties_v2_fn real = NULL;
    if (!real)
        real = (cudaGetDeviceProperties_v2_fn)_real_dlsym(
            RTLD_NEXT, "cudaGetDeviceProperties_v2");
    if (!real)
        return cudaErrorSymbolNotFound;
    cudaError_t rc = real(prop, device);
    if (rc == cudaSuccess && prop)
        sm_cap_clamp(&prop->major, &prop->minor);
    return rc;
}

typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int*, enum cudaDeviceAttr, int);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr,
                                   int device) {
    static cudaDeviceGetAttribute_fn real = NULL;
    if (!real)
        real = (cudaDeviceGetAttribute_fn)_real_dlsym(
            RTLD_NEXT, "cudaDeviceGetAttribute");
    if (!real)
        return cudaErrorSymbolNotFound;
    cudaError_t rc = real(value, attr, device);
    if (rc == cudaSuccess && value &&
        attr == cudaDevAttrComputeCapabilityMinor) {
        int major = 0;
        if (real(&major, cudaDevAttrComputeCapabilityMajor, device) ==
            cudaSuccess) {
            int minor = *value;
            sm_cap_clamp(&major, &minor);
            *value = minor;
        }
    }
    return rc;
}
