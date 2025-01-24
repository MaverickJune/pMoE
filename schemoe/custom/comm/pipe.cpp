#include "pipe.h"
// #include <mutex>
// std::mutex cout_mutex;

PipeComm::PipeComm(std::vector<at::cuda::CUDAStream> *stream,
                   std::vector<ncclComm_t>            g_nccl_comm,
                   const int                         &g_world_size,
                   const int                         &g_world_rank,
                   const int                         &g_local_size,
                   const int                         &g_local_rank) :
    AbstractComm(stream, g_nccl_comm, g_world_size, g_world_rank, g_local_size, g_local_rank) {
}

void PipeComm::all_to_all(const torch::Tensor &input, const torch::Tensor &output, size_t version) {
    
    // if (idx.numel() != g_world_size || gidx.numel() != g_world_size) {
    //         throw std::runtime_error("idx or gidx size does not match g_world_size");
    //     }
    size_t length = input.nbytes(); 
    // revoke
    std::vector<size_t> i_offsets(g_world_size, 0);
    std::vector<size_t> o_offsets(g_world_size, 0);
    // i_offsets = at::empty({g_world_size}, idx.options());
    // g_offsets = at::empty({g_world_size}, gidx.options());
    // i_offsets = at::zeros({g_world_size}, gidx.options());
    // g_offsets = at::zeros({g_world_size}, gidx.options());

    // original
    if (version == 0) {
        length = length / i_batch; // 보낼 최소 단위 model_dim (token 단일일)

        for (int i = 1; i < g_world_size; ++i) {
            i_offsets[i] = i_offsets[i - 1] + idx[i - 1] * length;
            o_offsets[i] = o_offsets[i - 1] + gidx[i - 1] * length;
        }
        
        CHECK_EQ(0, ncclGroupStart());
        for (int i = 0; i < g_world_size; ++i) {
            bool is_intra = 0; // (g_world_rank / g_local_size) == (i / g_local_size);
            // if (g_world_rank ==0) {
            //     std::cout << "current_length: " << idx[i] << std::endl;
            //     std::cout << "current output length: " << gidx[i] << std::endl;
            //     std::cout << "i: " << i << " input bytes: " << i_current_length << std::endl;
            //     std::cout << "i: " << i << " output bytes: " << o_current_length << std::endl;
            // }
            CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + i_offsets[i],
                                 idx[i] * length,
                                 ncclInt8,
                                 i,
                                 g_nccl_comm[is_intra],
                                 stream->at(is_intra).stream()));

            CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + o_offsets[i],
                                 gidx[i] * length,
                                 ncclInt8,
                                 i,
                                 g_nccl_comm[is_intra],
                                 stream->at(is_intra).stream()));
            // if (g_world_rank == 0) {
            //     std::cout << "Send and Recv Done for iter: " << i << std::endl;
            // }
        }
        CHECK_EQ(0, ncclGroupEnd());
    }
    // reshape a2a
    else if (version == 1) { // (토큰 / n) 
        length = length / g_world_size; // local batch * model_dim / world_size
        length = length / i_batch;
        
        for (int i = 1; i < g_world_size; ++i) {
            i_offsets[i] = i_offsets[i - 1] + i_batch * length; // i_batch = idx.sum()
            o_offsets[i] = o_offsets[i - 1] + gidx[i - 1] * length; // gidx = i_batch(gpu)
        }
        
        
        CHECK_EQ(0, ncclGroupStart());
        for (int i = 0; i < g_world_size; ++i) {
            bool is_intra = 0; // (g_world_rank / g_local_size) == (i / g_local_size);

            CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + i_offsets[i],
                                 i_batch * length,
                                 ncclInt8,
                                 i,
                                 g_nccl_comm[is_intra],
                                 stream->at(is_intra).stream()));

            CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + o_offsets[i],
                                 gidx[i] * length,
                                 ncclInt8,
                                 i,
                                 g_nccl_comm[is_intra],
                                 stream->at(is_intra).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());
    }
    else if (version == 2) { // reduce_scatter 
        length = length / g_world_size; // for reshape the data
        
        for (int i = 1; i < g_world_size; ++i) {
            i_offsets[i] = i_offsets[i - 1] + i * length;
            o_offsets[i] = o_offsets[i - 1] + i * length;
        }
        CHECK_EQ(0, ncclGroupStart());
        for (int i = 0; i < g_world_size; ++i) {
            // size_t current_length = gidx[i] * length;

            bool is_intra = 0; // (g_world_rank / g_local_size) == (i / g_local_size);
            CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + i_offsets[i],
                                length,
                                ncclInt8,
                                i,
                                g_nccl_comm[is_intra],
                                stream->at(is_intra).stream()));
            CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + o_offsets[i],
                                length,
                                ncclInt8,
                                i,
                                g_nccl_comm[is_intra],
                                stream->at(is_intra).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());
    } 
    else if (version == 3) { // Global Reshape
        size_t oput = output.nbytes();
        size_t iput = input.nbytes();
        length = length / i_batch;
        // std::cout << "Chunk length: " << length << " i_batch: " << i_batch << " g_batch: " << g_batch << std::endl;
        for (int i = 1; i < g_world_size; ++i) {
            i_offsets[i] = i_offsets[i - 1] + idx[i - 1] * length;
            o_offsets[i] = o_offsets[i - 1] + gidx[i - 1] * length;
            // std::cout << "i_offsets[" << i << "]: " << i_offsets[i]
            //       << ", o_offsets[" << i << "]: " << o_offsets[i]
            //       << ", Recv Length: " << gidx[i] * length << std::endl;
            // if (i_offsets[i] + idx[i] * length > iput) {
            // std::cerr << "RANK: " << g_local_rank
            //           << ", i_offsets[" << i << "] exceeds input size." << std::endl;
            // }
            // if (o_offsets[i] + gidx[i] * length > oput) {
            // std::cerr << "RANK: " << g_local_rank
            //           << ", o_offsets[" << i << "] exceeds output size." << std::endl;
            // }
            // std::cerr << "RANK: " << g_local_rank
            //           << ", idx: " << idx
            //           << ", gidx: " << gidx 
            //           << ", oput: " << oput
            //           << ", iput: " << iput
            //           << std::endl;
        }
        CHECK_EQ(0, ncclGroupStart());
        for (int i = 0; i < g_world_size; ++i) {
            bool is_intra = 0; // (g_world_rank / g_local_size) == (i / g_local_size);
            CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + i_offsets[i],
                                idx[i] * length,
                                ncclInt8,
                                i,
                                g_nccl_comm[is_intra],
                                stream->at(is_intra).stream()));
            CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + i * o_offsets[i],
                                gidx[i] * length,
                                ncclInt8,
                                i,
                                g_nccl_comm[is_intra],
                                stream->at(is_intra).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());
    }
    else {
        throw std::runtime_error("Invalid version");
    }
    // if (g_world_rank == 0) {
    //             std::cout << "Send and Recv Done for iter: " << std::endl;
    // }
}