#include <unordered_map>
#include <mutex>
#include <cassert>
#include <thread>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "fastermoe/status.h"
#include "stream_manager.h"

#define SMGR_N_STREAMS 16
#include <nccl.h>

cudaStream_t CudaStreamManager::stream(size_t idx) {
    if (this->use_default) {
        return c10::cuda::getCurrentCUDAStream().stream();
    }
    return this->streams[idx % SMGR_N_STREAMS];
}

cudaStream_t CudaStreamManager::torchStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

cublasHandle_t CudaStreamManager::handle(size_t idx) {
    if (this->use_default) {
        return at::cuda::getCurrentCUDABlasHandle();
    }
    return this->handles[idx % SMGR_N_STREAMS];
}

#ifdef FMOE_USE_NCCL
ncclComm_t CudaStreamManager::getComm(size_t idx) {
    // return ncclcomm;
    return this->ncclcomm[idx % SMGR_N_STREAMS];
}
char CudaStreamManager::getncclgood(size_t idx) {
    return this->ncclgood[idx % SMGR_N_STREAMS];
}
#endif

void CudaStreamManager::syncTorch() {
    cudaStreamSynchronize(this->torchStream());
}

void CudaStreamManager::sync(int idx) {
    if (this->use_default) {
        return;
    }
    for (int i = 0; i < idx && i < SMGR_N_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
}

void CudaStreamManager::setup(const int device) {
#ifdef FMOE_USE_NCCL
    this->ncclgood = new char[SMGR_N_STREAMS];
    this->ncclcomm = new ncclComm_t[SMGR_N_STREAMS];
#endif
    this->device = device;
    checkCudaErrors(cudaSetDevice(device));
    this->streams = new cudaStream_t[SMGR_N_STREAMS];
    this->handles = new cublasHandle_t[SMGR_N_STREAMS];
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        // SHOULD NOT USE: cudaStreamCreate(...)
        // more details in
        // https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
        checkCudaErrors(cudaStreamCreateWithFlags(streams + i,
                        cudaStreamNonBlocking));
        checkCudaErrors(cublasCreate(handles + i));
        cublasSetStream(handles[i], streams[i]);
    }
}

void CudaStreamManager::destroy() {
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cublasDestroy(handles[i]));
#ifdef FMOE_USE_NCCL
        // Detroy NCCL
        if (ncclcomm[i] != nullptr) {
            ncclCommDestroy(ncclcomm[i]);
        }
#endif
    }

    delete[] streams;
    delete[] handles;
#ifdef FMOE_USE_NCCL
    // NCCL memory delete
    delete[] ncclgood;
    delete[] ncclcomm;
#endif
}

std::unordered_map<int, CudaStreamManager*> smgrs;
std::mutex smgr_mtx;

CudaStreamManager* getCudaStreamManager(const int device) {
    auto it = smgrs.find(device);
    if (it == smgrs.end()) {
        smgr_mtx.lock();
        it = smgrs.find(device);
        if (it == smgrs.end()) {
            auto smgr = new CudaStreamManager(device);
            smgrs.insert(std::pair<int, CudaStreamManager*>(device, smgr));
            smgr_mtx.unlock();
            return smgr;
        } else {
            smgr_mtx.unlock();
        }
    }
    return it->second;
}

