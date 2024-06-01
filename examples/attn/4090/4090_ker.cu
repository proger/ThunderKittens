#include "../../../src/kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

#define NUM_WORKERS 1 // This kernel uses this many workers in parallel per block, to help issue instructions more quickly.
#define DIMENSION 16 // This kernel operates over 16-dimensional vectors

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.

__device__ void tileprint(rt_bf_1x1<> reg, char *name) {
    auto warpid        = kittens::warpid();
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            for (int k = 0; k < reg.packed_per_thread; k++) {
                auto item = __bfloat1622float2(reg.tiles[i][j].data[k]);
                printf("warpid=%d tid=%d rows %s[%d][%d].data[%d] = {%f,%f}\n", warpid, threadIdx.x, name, i, j, k, item.x, item.y);
            }
        }
    }
}

__global__ void attend_ker16(int n, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*DIMENSION);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x1<ducks::st_layout::swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::swizzle>, NUM_WORKERS>();
    st_bf_1x1<ducks::st_layout::swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::swizzle>, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x1<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x1<> o_reg;
    const auto S = q_reg.rows; // source time
    const auto T = q_reg.rows; // target time
    const auto D = q_reg.cols; // headdim
    
    int qo_blocks = n / (S*NUM_WORKERS);
    //printf("have warpid=%d blockIdx.x=%d,%d,%d threadIdx=%d,%d,%d  block_start=%d qo_blocks=%d __q__=%p\n", warpid,  blockIdx.x, blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z, block_start, qo_blocks, __q__);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        // each warp loads its own Q tile of 16x16
        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*S*D, D);

        if (q_blk == 0) {
            tileprint(q_reg, "q_reg");
            // make_causal(q_reg, q_reg, kittens::base_types::constants<bf16>::zero());
            // tileprint(q_reg, "CAUSAL");
        }

        // zero flash attention O registers.
        zero(o_reg);

        // iterate over k, v for these q's that have been loaded
        for(auto kv_idx = 0; kv_idx < q_blk + 1; kv_idx++) {
            // each warp loads its own chunk of k, v into shared memory
            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*S*D, D);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*S*D, D);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {
                if (kittens::laneid() >= 0) {
                    printf("warpid=%d threadIdx.x=%d q_source_range=%d:%d, kv_target_range=%d:%d\n",
                            warpid, threadIdx.x,
                            (q_blk*NUM_WORKERS + warpid)*S, (q_blk*NUM_WORKERS + warpid)*S + S, // q_range
                            (kv_idx*NUM_WORKERS + subtile)*S, (kv_idx*NUM_WORKERS + subtile)*S + S); // kv_range
                }

                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile

                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                if (kv_idx == q_blk) {
                    make_causal(att_block_mma, att_block_mma, kittens::base_types::constants<bf16>::zero());
                }

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x1<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.

                //one(o_reg);
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*S*D, o_reg, D); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

#include "harness.impl"
