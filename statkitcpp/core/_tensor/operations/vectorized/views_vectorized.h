#include <immintrin.h>

namespace statkitcpp {

namespace vec {

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) { //NOLINT
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}

void transpose_block_SSE4x4float(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) { //NOLINT
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}

template <typename T>
inline void transpose_scalar_block(T *A, T *B, const int lda, const int ldb, const int block_size) { //NOLINT
    #pragma omp parallel for
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            B[j*ldb + i] = A[i*lda +j];
        }
    }
}
template <typename T>
inline void transpose_block(T *A, T *B, const int n, const int m, const int lda, const int ldb, const int block_size) {//NOLINT
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*lda +j], &B[j*ldb + i], lda, ldb, block_size);
        }
    }
}

template <typename T>
void transpose(T* src, T* dest, const int& n, const int& m) { //NOLINT
    #define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
    int lda = ROUND_UP(m, 16);
    int ldb = ROUND_UP(n, 16);
    transpose_block(src, dest, n, m, lda, ldb, 64);
}

}

}