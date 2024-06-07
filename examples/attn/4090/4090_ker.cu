#include "../../../src/kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

#define NUM_WORKERS 1 // This kernel uses this many workers in parallel per block, to help issue instructions more quickly.
#define DIMENSION 16 // This kernel operates over 16-dimensional vectors
#define DEBUG 0

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.

__device__ void tileprint(rt_bf_1x1<> reg, char *name, int a, int b, int c, int d) {
    auto warpid        = kittens::warpid();
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            for (int k = 0; k < reg.packed_per_thread; k++) {
                auto item = __bfloat1622float2(reg.tiles[i][j].data[k]);
                printf("warpid=%d tid=%d laneid=%d rows q=%d:%d kv=%d:%d %s[%d][%d].data[%d] = {%f,%f}\n", warpid, threadIdx.x, laneid(), a,b,c,d, name, i, j, k, item.x, item.y);
            }
        }
    }
}

__device__ void tileprint_fl(rt_fl_1x1<> reg, char *name, int a, int b, int c, int d) {
    auto warpid        = kittens::warpid();
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            static_assert(reg.packed_per_thread == 4, "packed_per_thread must be 4");

            int row_top = laneid() / 4;
            int row_bottom = row_top + 8;
            int colL = laneid() % 4 * 2; // stride 4
            int colR = colL + 8;

            auto itemTL = reg.tiles[i][j].data[0];
            auto itemTR = reg.tiles[i][j].data[2];
            auto itemBL = reg.tiles[i][j].data[1];
            auto itemBR = reg.tiles[i][j].data[3];
            //printf("warpid=%d tid=%d laneid=%d rows q=%d:%d kv=%d:%d %s[%d][%d].data[%d] = {%f,%f}\n", warpid, threadIdx.x, laneid(), a,b,c,d, name, i, j, k, item.x, item.y);
            printf("q=%d:%d kv=%d:%d warpid=%d laneid=%d top=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                a,b,c,d, warpid, laneid(), row_top, colL, itemTL.x, itemTL.y, colR, itemTR.x, itemTR.y);
            printf("q=%d:%d kv=%d:%d warpid=%d laneid=%d bottom=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                a,b,c,d, warpid, laneid(), row_bottom, colL, itemBL.x, itemBL.y, colR, itemBR.x, itemBR.y);
        }
    }
}


__device__ void tileprint_bf(rt_bf_1x1<> reg, char *name, int a, int b, int c, int d) {
    auto warpid        = kittens::warpid();
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            static_assert(reg.packed_per_thread == 4, "packed_per_thread must be 4");

            int row_top = laneid() / 4;
            int row_bottom = row_top + 8;
            int colL = laneid() % 4 * 2; // stride 4
            int colR = colL + 8;

            auto itemTL = __bfloat1622float2(reg.tiles[i][j].data[0]);
            auto itemTR = __bfloat1622float2(reg.tiles[i][j].data[2]);
            auto itemBL = __bfloat1622float2(reg.tiles[i][j].data[1]);
            auto itemBR = __bfloat1622float2(reg.tiles[i][j].data[3]);
            //printf("warpid=%d tid=%d laneid=%d rows q=%d:%d kv=%d:%d %s[%d][%d].data[%d] = {%f,%f}\n", warpid, threadIdx.x, laneid(), a,b,c,d, name, i, j, k, item.x, item.y);
            printf("%s q=%d:%d kv=%d:%d warpid=%d laneid=%d top=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                name, a,b,c,d, warpid, laneid(), row_top, colL, itemTL.x, itemTL.y, colR, itemTR.x, itemTR.y);
            printf("%s q=%d:%d kv=%d:%d warpid=%d laneid=%d bottom=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                name, a,b,c,d, warpid, laneid(), row_bottom, colL, itemBL.x, itemBL.y, colR, itemBR.x, itemBR.y);
        }
    }
}



__device__ void vecprint(rv<float2, 1> reg, char *name) {
    auto warpid        = kittens::warpid();
    auto item = reg.data[0][0];
    printf("warpid=%d tid=%d %s[0] = {%f,%f}\n", warpid, threadIdx.x, name, item.x, item.y);
}


/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_eye(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i || j < i) { // below or above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            } else { // on the diagonal, interesting!
                dst.tiles[i][j].data[1] = packed_val; // below diagonal, zero
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero

                if (laneid() == 0 || laneid() == 9 || laneid() == 18 || laneid() == 27) {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                } else {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }

                if (laneid() == 4 || laneid() == 13 || laneid() == 22 || laneid() == 31) {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                } else {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
            }
        }
    }
}



/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_with_diag(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero

                if (laneid() == 0 || laneid() == 9 || laneid() == 18 || laneid() == 27) {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                } else {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }

                if (laneid() == 4 || laneid() == 13 || laneid() == 22 || laneid() == 31) {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                } else {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
            }
        }
    }
}


template<typename T>
__device__ static inline T packed_shfl_up_sync(uint32_t mask, const T &f, int delta) {
    return __shfl_up_sync(mask, f, delta);
}

template<>
__device__ inline float2 packed_shfl_up_sync<float2>(uint32_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_up_sync(mask, f.x, delta);
    r.y = __shfl_up_sync(mask, f.y, delta);
    return r;
}


template<typename T2> using rt_1x1_row = rt<T2, 1, 1, ducks::rt_layout::row>;

/**
 * @brief Perform a row-wise multiplication scan on a matrix in row-major layout from right to left (backwards).
 *
 * This function template performs a parallel scan across the rows of a matrix using multiplication operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam dtype The 2-element vector type for row elements (float2, bf16_2).
 * @param[inout] src The source matrix where which to perform the scan.
 */
template<typename dtype>
__device__ static inline void row_scan_backwards(rt_1x1_row<dtype> &src) {
    dtype rTc01 = src.tiles[0][0].data[0]; // top row: r0, cols 0 1
    dtype rTc89 = src.tiles[0][0].data[2]; // top row: r0, cols 8 9
    dtype rBc01 = src.tiles[0][0].data[1]; // bottom row: r0 + 8, cols 0 1
    dtype rBc89 = src.tiles[0][0].data[3]; // bottom row: r0 + 8, cols 8 9

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        src.tiles[0][0].data[i].x *= src.tiles[0][0].data[i].y;
        src.tiles[0][0].data[i].y = src.tiles[0][0].data[i].x;
    }


    int row_top = laneid() / 4;
    int row_bottom = row_top + 8;
    int colL = laneid() % 4 * 2; // stride 4
    int colR = colL + 8;

    // thread (laneid() % 4 == 3) should not be receiving from anyone
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dtype recv = packed_shfl_down_sync(MASK_ALL, src.tiles[0][0].data[i], 1);
        if (laneid() % 4 < 3) {
            src.tiles[0][0].data[i].x *= recv.x;
            src.tiles[0][0].data[i].y *= recv.y;
        }

        recv = packed_shfl_down_sync(MASK_ALL, src.tiles[0][0].data[i], 2);

        if (laneid() % 4 < 2) {
            src.tiles[0][0].data[i].x *= recv.x;
            src.tiles[0][0].data[i].y *= recv.y;
        }
    }

    if (colL == 0) {
        src.tiles[0][0].data[0].x *= src.tiles[0][0].data[2].x; // r0: 8 to 0
        src.tiles[0][0].data[0].y *= src.tiles[0][0].data[2].y; // r0: 8 to 0
        src.tiles[0][0].data[1].x *= src.tiles[0][0].data[3].x; // r8: 8 to 0
        src.tiles[0][0].data[1].y *= src.tiles[0][0].data[3].y; // r8: 8 to 0
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto recv = packed_shfl_up_sync(MASK_ALL, src.tiles[0][0].data[i+2], 2);

        if (laneid() % 4 == 2) { // only (laneid() % 4 == 2) is receiving
            src.tiles[0][0].data[i].x *= recv.x;
            src.tiles[0][0].data[i].y *= recv.y;

            recv = packed_shfl_up_sync(MASK_ALL, recv, 1);
        } else {
            recv = packed_shfl_up_sync(MASK_ALL, src.tiles[0][0].data[i+2], 1);
        }
        if (laneid() % 4 == 1 || laneid() % 4 == 3) {  // only (laneid() % 4 == 1, 3) are receiving
            src.tiles[0][0].data[i].x *= recv.x;
            src.tiles[0][0].data[i].y *= recv.y;
        }
    }

    src.tiles[0][0].data[0].y /= rTc01.x;
    src.tiles[0][0].data[2].y /= rTc89.x;
    src.tiles[0][0].data[1].y /= rBc01.x;
    src.tiles[0][0].data[3].y /= rBc89.x;
}



/**
 * @brief Repeat the leading (leftmost) column of a matrix in a row-major layout across other columns.
 *
 * @tparam dtype The 2-element vector type for row elements (float2, bf16_2).
 * @param[inout] src The source matrix.
 */
template<typename dtype>
__device__ static inline void repeat_leading_col(rt_1x1_row<dtype> &src) {

#define shuffle_far(x) {auto recv = packed_shfl_up_sync(MASK_ALL, x, 2); if (laneid() % 4 == 2) x = recv;}
#define shuffle_near(x) {auto recv = packed_shfl_up_sync(MASK_ALL, x, 1); if (laneid() % 2 == 1) x = recv;}

    src.tiles[0][0].data[0].y = src.tiles[0][0].data[0].x; // top: copy 0 to 1
    src.tiles[0][0].data[2].x = src.tiles[0][0].data[0].x; // top: copy 0 to 8
    src.tiles[0][0].data[2].y = src.tiles[0][0].data[0].x; // top: copy 0 to 9

    src.tiles[0][0].data[1].y = src.tiles[0][0].data[1].x; // bottom: copy 0 to 1
    src.tiles[0][0].data[3].x = src.tiles[0][0].data[1].x; // bottom: copy 0 to 8
    src.tiles[0][0].data[3].y = src.tiles[0][0].data[1].x; // bottom: copy 0 to 9

    // top far shuffle: 0, 1 to 4, 5; 8, 9 to 12, 13
    shuffle_far(src.tiles[0][0].data[0]);
    shuffle_far(src.tiles[0][0].data[2]);

    // bottom far shuffle: 0, 1 to 4, 5; 8, 9 to 12, 13
    shuffle_far(src.tiles[0][0].data[1]);
    shuffle_far(src.tiles[0][0].data[3]);

    // top near shuffle: 0, 1 to 2, 3; 8, 9 to 10, 11
    shuffle_near(src.tiles[0][0].data[0]);
    shuffle_near(src.tiles[0][0].data[2]);

    // bottom near shuffle: 0, 1 to 2, 3; 8, 9 to 10, 11
    shuffle_near(src.tiles[0][0].data[1]);
    shuffle_near(src.tiles[0][0].data[3]);
}



__global__ void attend_ker16(
    int n,
    const bf16* __restrict__ __q__,
    const bf16* __restrict__ __k__,
    const bf16* __restrict__ __v__,
    const float* __restrict__ __f__,
    bf16* __o__
) {

    auto warpid      = kittens::warpid();
    auto gate_start  = blockIdx.x*n;
    auto block_start = blockIdx.x*(n*DIMENSION);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
    bf16 *_o = __o__ + block_start;
    const float *_f = __f__ + gate_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x1<ducks::st_layout::swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::swizzle>, NUM_WORKERS>();
    st_bf_1x1<ducks::st_layout::swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::swizzle>, NUM_WORKERS>();
    sv_fl_1 (&f_smem)[NUM_WORKERS] = al.allocate<sv_fl_1, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x1<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rv<float2,1> f_reg;
    rt_fl_1x1<> F_reg; // multiplied matrix
    rt_bf_1x1<> F_reg_mul;
    rt_bf_1x1<> F_reg2;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x1<> o_reg;
    const auto S = q_reg.rows; // source time
    const auto T = q_reg.rows; // target time
    const auto D = q_reg.cols; // headdim

    int qo_blocks = n / (S*NUM_WORKERS);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {
        // each warp loads its own Q tile of 16x16
        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*S*D, D);

        // zero flash attention O registers.
        zero(o_reg);

        auto q_source_range_start = (q_blk*NUM_WORKERS + warpid)*S;
        auto q_source_range_end = (q_blk*NUM_WORKERS + warpid)*S + S;

        // iterate over k, v for these q's that have been loaded BACKWARDS
        for(auto kv_idx = q_blk; kv_idx >= 0; kv_idx--) {
            // each warp loads its own chunk of k, v into shared memory
            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*S*D, D);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*S*D, D);
            // each warp loads the chunk of forget gates into shared memory
            load(f_smem[warpid], _f + (kv_idx*NUM_WORKERS + warpid)*S);

            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            auto kv_target_range_start = (kv_idx*NUM_WORKERS + warpid)*S;
            auto kv_target_range_end = (kv_idx*NUM_WORKERS + warpid)*S + S;

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {
                load(k_reg, k_smem[subtile]); // load k from shared into registers
                load(f_reg, f_smem[subtile]); // load f from shared into registers

                if (DEBUG) {
                    auto item = f_reg.data[0][0];
                    printf("warpid=%d tid=%d q_source_range=%d:%d, kv_target_range=%d:%d forget[0] = {%f,%f}\n",
                           kittens::warpid(), threadIdx.x,
                           q_source_range_start, q_source_range_end,
                           kv_target_range_start, kv_target_range_end,
                           item.x, item.y);
                    vecprint(f_reg, "f_reg");
                }

                zero(att_block); // zero 16x16 attention tile

                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                // copy f_reg into every row of F_reg
                rt_bf_1x1<ducks::rt_layout::col> F_reg_col;
                rv<bf16_2, 1> f_reg_bf;
                copy(f_reg_bf, f_reg);
                broadcast_col(F_reg_col, f_reg_bf);
                rt_bf_1x1<ducks::rt_layout::row> F_reg_row = swap_layout_inplace(F_reg_col);
                copy(F_reg, F_reg_row);

                if (kv_idx == q_blk) {
                    make_causal(att_block_mma, att_block_mma, kittens::base_types::constants<bf16>::zero());

                    make_causal(F_reg, F_reg, kittens::base_types::constants<float>::one());
                    make_causal_with_diag(F_reg, F_reg, kittens::base_types::constants<float>::one());
                }

                row_scan_backwards<float2>(F_reg);

                if (kv_idx < q_blk) {
                    // take the left column of F_reg_mul and broadcast-multiply with every element in F_reg
                    repeat_leading_col<bf16_2>(F_reg_mul);
                    if (DEBUG) tileprint_bf(F_reg_mul, "F_reg_mul", q_source_range_start, q_source_range_end, kv_target_range_start, kv_target_range_end);
                    copy(F_reg2, F_reg);
                    mul(F_reg_mul, F_reg_mul, F_reg2);
                } else {
                    copy(F_reg_mul, F_reg);
                }

                if (DEBUG) tileprint_fl(F_reg, "F_reg", q_source_range_start, q_source_range_end, kv_target_range_start, kv_target_range_end);

                mul(att_block_mma, att_block_mma, F_reg_mul);

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x1<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*S*D, o_reg, D); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

#include "harness.impl"
