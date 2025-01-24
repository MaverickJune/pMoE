#include "no.h"

// #include <torch/torch.h>     // PyTorch C++ API의 핵심 헤더
// #include <ATen/Tensor.h>     // ATen의 Tensor 클래스
// #include <ATen/ATen.h>       // ATen 네임스페이스 연산
// #include <cstddef>           // size_t 정의

// compress 에서 output 버퍼까지 init 을 하고 version 을 받아오기. 
// torch::Tensor NoCompressor::compress(const torch::Tensor &input, const torch::Tensor &idx, const torch::Tensor &gidx) {
//     g_output = at::empty_like(input);
//     if (!idx.is_cuda() || !gidx.is_cuda()) {
//         throw std::runtime_error("idx and gidx tensors must be on GPU");
//     }
//     std::cout << "idx.numel(): " << idx.numel() << std::endl;
//     std::cout << "gidx.numel(): " << gidx.numel() << std::endl;
//     std::cout << "Max size of local vector: " << this->comm_ptr->idx.max_size() << std::endl;
//     std::cout << "Max size of global vector: " << this->comm_ptr->gidx.max_size() << std::endl;
//     this->comm_ptr->idx.assign(idx.to(at::kCPU).data_ptr<int64_t>(), 
//                       idx.to(at::kCPU).data_ptr<int64_t>() + idx.numel());
//     this->comm_ptr->gidx.assign(gidx.to(at::kCPU).data_ptr<int64_t>(), 
//                        gidx.to(at::kCPU).data_ptr<int64_t>() + gidx.numel());
//     for (auto &nccl_stream : *comm_ptr->stream) {
//         c10::cuda::CUDACachingAllocator::recordStream(g_output.storage().data_ptr(), nccl_stream);
//     }
//     std::cout << "Done " << std::endl;
    
//     return input;
// }
torch::Tensor NoCompressor::compress(const torch::Tensor &output, const torch::Tensor &idx, const torch::Tensor &gidx) {
    g_output = at::empty_like(output);

    // Ensure tensors are on GPU
    if (idx.is_cuda() || gidx.is_cuda()) {
        throw std::runtime_error("idx and gidx tensors must be on CPU");
    }
    // Check tensor properties
    if (idx.numel() == 0 || gidx.numel() == 0) {
        throw std::runtime_error("idx or gidx tensor is empty");
    }
    // if (idx.scalar_type() != torch::kL || gidx.scalar_type() != torch::kInt) {
    //     throw std::runtime_error("idx or gidx tensor is not integer type");
    // }
    if (!idx.is_contiguous() || !gidx.is_contiguous()) {
        throw std::runtime_error("idx and gidx tensor must be contiguous");
    }
    // Assign to comm_ptr
    if (idx.dtype() != torch::kLong || gidx.dtype() != torch::kLong ) {
    throw std::runtime_error("gidx tensor must have dtype torch::kLong");
    }
    this->comm_ptr->idx = idx.data_ptr<int64_t>(); 
    this->comm_ptr->gidx = gidx.data_ptr<int64_t>();
    
    this->comm_ptr->i_batch = idx.sum().item<int64_t>();
    this->comm_ptr->g_batch = gidx.sum().item<int64_t>();
    // this->comm_ptr->i_batch = this->comm_ptr->_i_batch.data_ptr<int64_t>();
    // this->comm_ptr->g_batch = this->comm_ptr->_g_batch.data_ptr<int64_t>();
    
    // this->comm_ptr->i_offset = at::zeros({g_world_size}, gidx.options());
    // this->comm_ptr->o_offset = at::zeros({g_world_size}, gidx.options());
    // this->comm_ptr->i_offsets = this->comm_ptr->i_offset.data_ptr<int64_t>();
    // this->comm_ptr->o_offsets = this->comm_ptr->o_offset.data_ptr<int64_t>();
    // std::cout << "idx and gidx location: " << this->comm_ptr->idx << " " << this->comm_ptr->gidx << std::endl;

    // Record streams
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(g_output.storage().data_ptr(), nccl_stream);
        // c10::cuda::CUDACachingAllocator::recordStream(idx.storage().data_ptr(), nccl_stream);
        // c10::cuda::CUDACachingAllocator::recordStream(gidx.storage().data_ptr(), nccl_stream);
    }

    // std::cout << "Done." << std::endl;
    return output;
}

torch::Tensor NoCompressor::decompress(const torch::Tensor &input) {
    return input;
}

NoCompressor::NoCompressor(std::shared_ptr<AbstractComm> comm_ptr) :
    AbstractCompressor(comm_ptr) {
}

void NoCompressor::all_to_all(const torch::Tensor &input, const torch::Tensor &output, size_t version) {
    comm_ptr->all_to_all(input, output, version);
}