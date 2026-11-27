#include "mnn_capi.h"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct MnnSessionHandle {
    std::unique_ptr<MNN::Interpreter, void (*)(MNN::Interpreter*)> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* output = nullptr;
    std::unique_ptr<MNN::Tensor> host_in;
    std::unique_ptr<MNN::Tensor> host_out;
    std::vector<int> in_shape;
    std::vector<int> out_shape;
    size_t in_elems = 0;
    size_t out_elems = 0;
    int32_t active_backend = -1;
    std::string cache_file;

    MnnSessionHandle() : net(nullptr, MNN::Interpreter::destroy) {}
};

static void set_err(char* buf, size_t len, const char* msg) {
    if (!buf || len == 0) return;
    if (!msg) {
        buf[0] = '\0';
        return;
    }
    std::snprintf(buf, len, "%s", msg);
}

extern "C" {

MnnSessionHandle* mnn_session_create(
    const char* model_path,
    const MnnCreateOptions* opts,
    char* err_buf,
    size_t err_buf_len
) {
    if (!model_path || model_path[0] == '\0') {
        set_err(err_buf, err_buf_len, "mnn_session_create: empty model_path");
        return nullptr;
    }
    if (!opts) {
        set_err(err_buf, err_buf_len, "mnn_session_create: null opts");
        return nullptr;
    }

    try {
        auto handle = std::make_unique<MnnSessionHandle>();
        MNN::Interpreter* raw_net = MNN::Interpreter::createFromFile(model_path);
        if (!raw_net) {
            char msg[512];
            std::snprintf(msg, sizeof(msg), "MNN::Interpreter::createFromFile failed: %s", model_path);
            set_err(err_buf, err_buf_len, msg);
            return nullptr;
        }
        handle->net.reset(raw_net);

        if (opts->cache_file && opts->cache_file[0] != '\0') {
            handle->cache_file = opts->cache_file;
            handle->net->setCacheFile(opts->cache_file);
        }
        handle->net->setSessionMode(MNN::Interpreter::Session_Release);

        MNN::ScheduleConfig config;
        MNN::BackendConfig bcfg;
        switch (opts->precision) {
            case MNN_CAPI_PRECISION_LOW:
                bcfg.precision = MNN::BackendConfig::Precision_Low;
                break;
            case MNN_CAPI_PRECISION_HIGH:
                bcfg.precision = MNN::BackendConfig::Precision_High;
                break;
            default:
                bcfg.precision = MNN::BackendConfig::Precision_Normal;
                break;
        }
        bcfg.power = MNN::BackendConfig::Power_High;
        bcfg.memory = MNN::BackendConfig::Memory_Normal;

        if (opts->backend == MNN_CAPI_BACKEND_OPENCL) {
            config.type = MNN_FORWARD_OPENCL;
            config.mode = opts->gpu_mode != 0 ? opts->gpu_mode : MNN_CAPI_GPU_MODE_DEFAULT;
        } else if (opts->backend == MNN_CAPI_BACKEND_CPU) {
            config.type = MNN_FORWARD_CPU;
            config.numThread = opts->num_threads > 0 ? opts->num_threads : 2;
        } else {
            set_err(err_buf, err_buf_len, "mnn_session_create: unsupported backend");
            return nullptr;
        }
        config.backupType = MNN_FORWARD_CPU;
        config.backendConfig = &bcfg;

        handle->session = handle->net->createSession(config);
        if (!handle->session) {
            char msg[512];
            std::snprintf(msg, sizeof(msg), "MNN createSession failed for %s", model_path);
            set_err(err_buf, err_buf_len, msg);
            return nullptr;
        }

        int backends[4] = {-1, -1, -1, -1};
        handle->net->getSessionInfo(handle->session, MNN::Interpreter::BACKENDS, backends);
        handle->active_backend = backends[0];

        if (opts->backend == MNN_CAPI_BACKEND_OPENCL && !opts->allow_cpu_backup &&
            handle->active_backend != MNN_FORWARD_OPENCL) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "OpenCL backend unavailable (active_backend=%d, expected=%d)",
                          handle->active_backend, (int)MNN_FORWARD_OPENCL);
            set_err(err_buf, err_buf_len, msg);
            return nullptr;
        }

        handle->input = handle->net->getSessionInput(handle->session, nullptr);
        handle->output = handle->net->getSessionOutput(handle->session, nullptr);
        if (!handle->input || !handle->output) {
            set_err(err_buf, err_buf_len, "MNN session has null input or output tensor");
            return nullptr;
        }

        handle->in_shape = handle->input->shape();
        handle->out_shape = handle->output->shape();
        handle->host_in = std::make_unique<MNN::Tensor>(handle->input, MNN::Tensor::CAFFE);
        handle->host_out = std::make_unique<MNN::Tensor>(handle->output, MNN::Tensor::CAFFE);
        handle->in_elems = static_cast<size_t>(handle->host_in->elementSize());
        handle->out_elems = static_cast<size_t>(handle->host_out->elementSize());

        if (handle->in_elems == 0 || handle->out_elems == 0) {
            set_err(err_buf, err_buf_len, "MNN session tensor has zero elements (dynamic shape not supported)");
            return nullptr;
        }

        return handle.release();
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return nullptr;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in mnn_session_create");
        return nullptr;
    }
}

int32_t mnn_session_active_backend(const MnnSessionHandle* handle) {
    if (!handle) return -1;
    return handle->active_backend;
}

int32_t mnn_session_input_shape(const MnnSessionHandle* handle, int32_t* dims, int32_t max_dims) {
    if (!handle) return -1;
    int32_t rank = static_cast<int32_t>(handle->in_shape.size());
    if (dims && max_dims > 0) {
        int32_t n = rank < max_dims ? rank : max_dims;
        for (int32_t i = 0; i < n; ++i) dims[i] = handle->in_shape[i];
    }
    return rank;
}

int32_t mnn_session_output_shape(const MnnSessionHandle* handle, int32_t* dims, int32_t max_dims) {
    if (!handle) return -1;
    int32_t rank = static_cast<int32_t>(handle->out_shape.size());
    if (dims && max_dims > 0) {
        int32_t n = rank < max_dims ? rank : max_dims;
        for (int32_t i = 0; i < n; ++i) dims[i] = handle->out_shape[i];
    }
    return rank;
}

int32_t mnn_session_run(
    MnnSessionHandle* handle,
    const float* in_buf,
    size_t in_elems,
    float* out_buf,
    size_t out_cap,
    size_t* out_elems_written,
    char* err_buf,
    size_t err_buf_len
) {
    if (!handle || !in_buf) {
        set_err(err_buf, err_buf_len, "mnn_session_run: null handle or input");
        return -1;
    }
    if (in_elems != handle->in_elems) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "mnn_session_run: input elems %zu != expected %zu",
                      in_elems, handle->in_elems);
        set_err(err_buf, err_buf_len, msg);
        return -2;
    }
    if (out_buf && out_cap < handle->out_elems) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "mnn_session_run: output cap %zu < required %zu",
                      out_cap, handle->out_elems);
        set_err(err_buf, err_buf_len, msg);
        return -3;
    }

    try {
        std::memcpy(handle->host_in->host<float>(), in_buf, in_elems * sizeof(float));
        handle->input->copyFromHostTensor(handle->host_in.get());
        MNN::ErrorCode rc = handle->net->runSession(handle->session);
        if (rc != MNN::NO_ERROR) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "MNN runSession failed with error code %d", (int)rc);
            set_err(err_buf, err_buf_len, msg);
            return -4;
        }
        handle->output->copyToHostTensor(handle->host_out.get());
        if (out_buf) {
            std::memcpy(out_buf, handle->host_out->host<float>(), handle->out_elems * sizeof(float));
        }
        if (out_elems_written) {
            *out_elems_written = handle->out_elems;
        }
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return -5;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in mnn_session_run");
        return -5;
    }
}

int32_t mnn_session_update_cache(
    MnnSessionHandle* handle,
    char* err_buf,
    size_t err_buf_len
) {
    if (!handle) {
        set_err(err_buf, err_buf_len, "mnn_session_update_cache: null handle");
        return -1;
    }
    if (handle->cache_file.empty()) {
        return 0;
    }
    try {
        handle->net->updateCacheFile(handle->session);
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return -2;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in mnn_session_update_cache");
        return -2;
    }
}

void mnn_session_destroy(MnnSessionHandle* handle) {
    if (!handle) return;
    delete handle;
}

} /* extern "C" */
