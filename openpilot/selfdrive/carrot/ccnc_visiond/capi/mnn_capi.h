#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MnnSessionHandle MnnSessionHandle;

/* Backend constants matching MNNForwardType */
#define MNN_CAPI_BACKEND_CPU    0
#define MNN_CAPI_BACKEND_OPENCL 3

/* Precision constants matching MNN::BackendConfig::PrecisionMode */
#define MNN_CAPI_PRECISION_NORMAL 0
#define MNN_CAPI_PRECISION_HIGH   1
#define MNN_CAPI_PRECISION_LOW    2

/* Default Adreno GPU mode: MNN_GPU_TUNING_NORMAL (8) | MNN_GPU_MEMORY_IMAGE (128) = 136 */
#define MNN_CAPI_GPU_MODE_DEFAULT 136

typedef struct MnnCreateOptions {
    int32_t backend;          /* MNN_CAPI_BACKEND_OPENCL or MNN_CAPI_BACKEND_CPU */
    int32_t precision;        /* MNN_CAPI_PRECISION_LOW (FP16) */
    int32_t num_threads;      /* CPU threads when backend == CPU */
    int32_t gpu_mode;         /* MNN_CAPI_GPU_MODE_DEFAULT (136) */
    int32_t allow_cpu_backup; /* 0 = fail if requested GPU backend fell back to CPU */
    const char* cache_file;   /* NULL or empty = no cache file */
} MnnCreateOptions;

MnnSessionHandle* mnn_session_create(
    const char* model_path,
    const MnnCreateOptions* opts,
    char* err_buf,
    size_t err_buf_len
);

int32_t mnn_session_active_backend(const MnnSessionHandle* handle);

int32_t mnn_session_input_shape(const MnnSessionHandle* handle, int32_t* dims, int32_t max_dims);
int32_t mnn_session_output_shape(const MnnSessionHandle* handle, int32_t* dims, int32_t max_dims);

int32_t mnn_session_run(
    MnnSessionHandle* handle,
    const float* in_buf,
    size_t in_elems,
    float* out_buf,
    size_t out_cap,
    size_t* out_elems_written,
    char* err_buf,
    size_t err_buf_len
);

int32_t mnn_session_update_cache(
    MnnSessionHandle* handle,
    char* err_buf,
    size_t err_buf_len
);

void mnn_session_destroy(MnnSessionHandle* handle);

#ifdef __cplusplus
}
#endif
