#include "stream_manager.h"

#ifdef FMOE_USE_NCCL

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count,
        long* global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr);

// shan
template<typename scalar_t>
void fmoe_cuda_global_reshape_impl(
    const scalar_t* local_input_buf, // [b x k, H]
    const long* local_expert_count, // [E]
    const long* global_expert_count, // [E x n]
    scalar_t* global_input_buf, // [B x k, h]
    size_t sub_in_feat, size_t total_experts, size_t n_workers, // h, E, n
    CudaStreamManager* smgr) {

    // fprintf(stderr, "DEBUG: Entering fmoe_cuda_global_reshape_impl\n");
    // fprintf(stderr, "DEBUG: sub_in_feat = %zu, total_experts = %zu, n_workers = %zu\n",
    //         sub_in_feat, total_experts, n_workers);
    
    int recv_ptr = 0;
    size_t send_offset = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[total_experts];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < total_experts; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1]; // covers [b x k]
    }
    // fprintf(stderr, "DEBUG: Expert pointer initialized.\n");
    for (size_t i = 0; i < total_experts; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        int idx = i;
        // fprintf(stderr, "DEBUG: Processing expert %zu/%zu\n", i, total_experts);
        
        // global receive
        for (size_t j = 0; j < n_workers; ++j) {
            send_offset = j * sub_in_feat;
            int g_idx = i + j * total_experts;
            
            if (local_expert_count[idx]) {
            // fprintf(stderr, "[DEBUG Rank %ld]: Sending %ld elements to worker %zu\n",
            //         smgr->device, local_expert_count[idx], j);
                NCCL_SAFE_CALL(ncclSend(
                    local_input_buf + send_offset + expert_ptr[idx] * (sub_in_feat* n_workers),
                    local_expert_count[idx] * sub_in_feat * sizeof(scalar_t),
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->torchStream()));
            }
            
            if (global_expert_count[g_idx]) {
                // fprintf(stderr, "[DEBUG Rank %ld]: Receiving %ld elements from worker %d\n",
                //         smgr->device, global_expert_count[g_idx], j);
                NCCL_SAFE_CALL(ncclRecv(
                        global_input_buf + recv_ptr * sub_in_feat,
                        global_expert_count[g_idx] * sub_in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                recv_ptr += global_expert_count[g_idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
        // fprintf(stderr, "[DEBUG Rank %ld]: Completed communication for expert %zu\n",smgr->device, i);
    }
    delete [] expert_ptr;
    // fprintf(stderr, "DEBUG: Exiting fmoe_cuda_global_reshape_impl\n");
}
// shan
template<typename scalar_t>
void fmoe_cuda_global_restore_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat,
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}

template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat,
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                recv_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}


template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat,
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->torchStream()));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}


#endif  // FMOE_USE_NCCL
