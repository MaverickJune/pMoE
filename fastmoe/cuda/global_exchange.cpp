#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>
#include <iostream>

std::mutex reshape_mtx;
std::mutex nccl_mtx;
// #include <c10d/ProcessGroup.hpp>
// #include <c10d/ProcessGroupNCCL.hpp>

const char* deviceTypeToString(torch::DeviceType deviceType) {
    switch (deviceType) {
        case torch::kCPU:
            return "CPU";
        case torch::kCUDA:
            return "CUDA";
        case torch::kHIP:
            return "HIP";
        case torch::kXLA:
            return "XLA";
        case torch::kVulkan:
            return "Vulkan";
        case torch::kMetal:
            return "Metal";
        case torch::kXPU:
            return "XPU";
        default:
            return "Unknown";
    }
}

void pmoe_cuda_expert_gather_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int total_experts, int world_size,
        CudaStreamManager* smgr,
        size_t idx) {
    pid_t pid = getpid();
    // pthread_t tid = pthread_self();

    fprintf(stderr, "PID: %d, TID: %zu, DEBUG: Entering Exper Gather \n", pid, idx);
    // std::lock_guard<std::mutex> lock(nccl_mtx);
    fprintf(stderr, "PID: %d, TID: %zu, DEBUG: After Lock \n", pid, idx);
    ncclComm_t _ncclcomm = smgr->ncclcomm[idx];
    fprintf(stderr, "DEBUG: ncclComm_t for idx %zu: %p\n", idx, (void*)_ncclcomm);

    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend( // B*K
                local_expert_count,
                total_experts,
                ncclInt64,
                i,
                _ncclcomm,
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv( // B*K*N
                global_expert_count + total_experts * i,
                total_experts,
                ncclInt64,
                i,
                _ncclcomm,
                smgr->torchStream()));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    fprintf(stderr, "PID: %d, TID: %zu, DEBUG: Before Stream Sync\n", pid, idx);
    // NCCL_SAFE_CALL(cudaStreamSynchronize(smgr->torchStream())); // smgr->syncTorch();
    fprintf(stderr, "PID: %d, TID: %zu, DEBUG: After Stream Sync\n", pid, idx);
}

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr,
        size_t idx) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->getComm(idx),
                smgr->torchStream()));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->getComm(idx),
                smgr->torchStream()));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    // NCCL_SAFE_CALL(cudaStreamSynchronize(smgr->torchStream()));
}

torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers, size_t idx) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr,
            idx
            );
    // pid_t pid = getpid();
    // pthread_t tid = pthread_self();

    // fprintf(stderr, "PID: %d, TID: %lu, DEBUG: Exiting Exper Gather\n", pid, (unsigned long)tid);
    
    return global_expert_count;
}

// shan pMoE
torch::Tensor _expert_gather(
        torch::Tensor local_expert_count,
        long total_experts, long n_workers, size_t idx) {
    auto global_expert_count = torch::empty({local_expert_count.size(0) * n_workers}, local_expert_count.options());
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    // printf("[DEBUG Rank %ld]: With torch stream %p\n", smgr->device, (void*)smgr->torchStream());
    // std::cerr << "cudaStream_t: " << smgr->torchStream() << std::endl;
    pmoe_cuda_expert_gather_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            total_experts, n_workers,
            smgr,
            idx);
    pid_t pid = getpid();

    fprintf(stderr, "PID: %d, TID: %zu, DEBUG: Exiting Exper Gather\n", pid, idx);
    return global_expert_count;
}
// shan pMoE
torch::Tensor _global_reshape(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, size_t idx) {
    CHECK_INPUT(input_buf);
    
    auto total_experts = local_expert_count.size(0); // E
    auto sub_in_feat = input_buf.size(1) / n_workers; // h : H / n
    auto global_input_buf = input_buf.new_empty({batch_size, sub_in_feat}); //[B x k, h]
    auto smgr = getCudaStreamManager(input_buf.device().index());

    // { std::lock_guard<std::mutex> lock(reshape_mtx);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            input_buf.scalar_type(), "fmoe_cuda_global_reshape", ([&] {
        fmoe_cuda_global_reshape_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(), // [b x k, H], idx = location of token related to experts
            local_expert_count.data_ptr<long>(), // [E]
            global_expert_count.data_ptr<long>(), // [E x n]
            global_input_buf.data_ptr<scalar_t>(), // [B x k, h]
            sub_in_feat, total_experts, n_workers, // h, E, n
            smgr,
            idx
        );
    }));
    return global_input_buf;
    // }
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, size_t idx) {
    CHECK_INPUT(input_buf);


    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            input_buf.scalar_type(), "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr,
            idx
        );
    }));
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers, size_t idx) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output_buf.scalar_type(), "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr,
            idx
        );
    }));
    return local_output_buf;
}

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13))
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        std::lock_guard<std::mutex> lock(reshape_mtx);
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
    // printf(stderr, "Generated NCCL Unique ID for idx %zu: ", idx);
    // for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    //     printf(stderr, "%02x", ncclID.internal[i]);
    // }
    // printf("\n");
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                false,
                "fastmoe_nccl_comm",
                rank);
#elif defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        // fprintf(stderr, "Rank %d received NCCL Unique ID\n", rank);

        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        // fprintf(stderr, "DEBUG: ncclCommInitRank rank=%d, world_size=%d\n", rank, getSize());

        return comm;
    }
};

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
void _ensure_nccl(c10d::ProcessGroup& p, torch::Tensor t, size_t idx) {
#else
void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t, size_t idx) {
#endif  // TORCH_VERSION
    std::lock_guard<std::mutex> lock(nccl_mtx);
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood[idx] == 1) {
        return;
    }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)
        (p.getBackend(c10d::ProcessGroup::NCCL).get());
#else
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
#endif  // TORCH_VERSION
    // fprintf(stderr, "DEBUG: Tensor device type: %s, index: %d\n", deviceTypeToString(t.device().type()), t.device().index());
    // fprintf(stderr, "DEBUG: Thread set CUDA device %d\n", t.device());
    smgr->ncclcomm[idx] = h->getcomm(t.device());
    // fprintf(stderr, "DEBUG: ncclComm_t initialization for idx %zu completed.\n", idx);
    if (smgr->ncclcomm[idx] != 0) {
        smgr->ncclgood[idx] = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}
#endif  // FMOE_USE_NCCL
