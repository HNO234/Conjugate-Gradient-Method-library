#include <_matrix_acc.hpp>
#include <thread>

#define BLOCK_SIZE 1024

namespace Matrix {

int Accelerated_Matrix::number_of_threads = std::thread::hardware_concurrency();

Accelerated_Matrix::Accelerated_Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol)
{
    size_t nelement = nrow * ncol;
    reset_buffer(nrow, ncol);
    memset(m_buffer, 0, nelement * sizeof(double));
}
//

Accelerated_Matrix::Accelerated_Matrix(Accelerated_Matrix const &mat) : m_nrow(mat.m_nrow), m_ncol(mat.m_ncol)
{
    reset_buffer(mat.m_nrow, mat.m_ncol);
    memcpy(m_buffer, mat.m_buffer, m_nrow * m_ncol * sizeof(double));
}

Accelerated_Matrix::Accelerated_Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec) : m_nrow(nrow), m_ncol(ncol)
{
    if (vec.size() != nrow * ncol){
        throw std::out_of_range("Accelerated_Matrix::Accelerated_Matrix(): vector size differs from matrix size");
    }
        reset_buffer(nrow, ncol);
        (*this) = vec;
}

Accelerated_Matrix::Accelerated_Matrix(std::vector<double> const & vec) : m_nrow(vec.size()), m_ncol(1)
{
    reset_buffer(vec.size(), 1);
    (*this) = vec;
}

Accelerated_Matrix::Accelerated_Matrix(std::vector<std::vector<double>> const & vec) : m_nrow(vec.size()), m_ncol(vec[0].size())
{
    reset_buffer(m_nrow, m_ncol);
    (*this) = vec;
}

Accelerated_Matrix::~Accelerated_Matrix()
{
    reset_buffer(0, 0);
}

Accelerated_Matrix::Accelerated_Matrix(Accelerated_Matrix && other)
  : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
    reset_buffer(0, 0);
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_buffer, other.m_buffer);
}

Accelerated_Matrix & Accelerated_Matrix::operator=(Accelerated_Matrix && other){
    if (this == &other) { return *this; }
    reset_buffer(0, 0);
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_buffer, other.m_buffer);
    return *this;
}

Accelerated_Matrix & Accelerated_Matrix::operator=(std::vector<double> const & vec)
{
    if (size() != vec.size()){
        throw std::out_of_range("number of elements mismatch");
    }
    size_t shape = m_nrow * m_ncol;
    memcpy(m_buffer, vec.data(), shape * sizeof(double));
    return *this;
}


Accelerated_Matrix & Accelerated_Matrix::operator=(std::vector<std::vector<double>> const & vec2d)
{
    if (m_nrow != vec2d.size() || m_ncol != vec2d[0].size()){
        throw std::out_of_range("number of elements mismatch");
    }
    size_t i;
    for (i=0; i<m_nrow; ++i)
    {
        memcpy(m_buffer + i * m_ncol, vec2d[i].data(), m_ncol * sizeof(double));
    }
    return *this;
}

Accelerated_Matrix & Accelerated_Matrix::operator=(Accelerated_Matrix const & other)
{
    if (this == &other) { return *this; }
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
    }
    size_t shape = m_nrow * m_ncol;
    memcpy(m_buffer, other.m_buffer, shape * sizeof(double));
    return *this;
}

Accelerated_Matrix Accelerated_Matrix::operator+(Accelerated_Matrix const & other) const { 
    Accelerated_Matrix temp = *this;
    return temp += other;
}

 __global__ void iadd_gpu(double* a_mat, double* b_mat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] += b_mat[i];
 }

Accelerated_Matrix& Accelerated_Matrix::operator+=(Accelerated_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }
    size_t shape = m_nrow * m_ncol;
    double *d_m_buffer, *d_other_m_buffer;
    cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
    cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
    cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    cudaMemcpy(d_other_m_buffer, other.m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
//     cudaMalloc((void**)&device_arr, sizeof(int) * elementSize);

// // Data Transfer and Kernel Function
// cudaMemcpy(device_arr, host_input_arr, sizeof(int) * elementSize, cudaMemcpyHostToDevice);
// kernel<<<blockSize, threadsPerBlock>>>(device_arr, elementSize);
// cudaDeviceSynchronize();
// cudaMemcpy(host_output_arr, device_arr, sizeof(int) * elementSize, cudaMemcpyDeviceToHost);
// cudaFree(device_arr);
    // cudaHostGetDevicePointer(&d_m_buffer, m_buffer, 0);
    // cudaHostGetDevicePointer(&d_other_m_buffer, other.m_buffer, 0);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
    iadd_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, shape);
    cudaDeviceSynchronize();
    cudaMemcpy(m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
    cudaFree(d_m_buffer);
    cudaFree(d_other_m_buffer);
    return (*this);
}

Accelerated_Matrix Accelerated_Matrix::operator-(Accelerated_Matrix const & other) const {
    Accelerated_Matrix temp = *this;
    return temp -= other;
}

 __global__ void isub_gpu(double* a_mat, double* b_mat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] -= b_mat[i];
 }

Accelerated_Matrix& Accelerated_Matrix::operator-=(Accelerated_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }
    size_t shape = m_nrow * m_ncol;
    double *d_m_buffer, *d_other_m_buffer;
    cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
    cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
    cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    cudaMemcpy(d_other_m_buffer, other.m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
    isub_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, shape);
    cudaDeviceSynchronize();
    cudaMemcpy(m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
    cudaFree(d_m_buffer);
    cudaFree(d_other_m_buffer);
    return (*this);
}

 __global__ void neg_gpu(double* a_mat, double* b_mat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] = -b_mat[i];
 }

Accelerated_Matrix Accelerated_Matrix::operator-() const {
    Accelerated_Matrix temp(m_nrow, m_ncol);
    size_t shape = m_nrow * m_ncol;
    double *d_m_buffer, *d_other_m_buffer;
    cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
    cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
    cudaMemcpy(d_other_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
    neg_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, shape);
    cudaDeviceSynchronize();
    cudaMemcpy(temp.m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
    cudaFree(d_m_buffer);
    cudaFree(d_other_m_buffer);
    return temp;
}

__global__ void imul_gpu(double* a_mat, double other, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] *= other;
 }

 __global__ void matmul1_gpu(double* a_mat, double* b_mat, double other, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] = b_mat[i] * other;
 }

 __device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

 __global__ void matmul2_gpu(double* out_mat, double* a_mat, double* b_mat, int ncol) {
    __shared__ double sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int a_mat_i = blockIdx.x * ncol + tid;
    unsigned int b_mat_i = tid;
    sdata[tid] = 0;
    while (b_mat_i < ncol) {
        sdata[tid] += a_mat[a_mat_i] * b_mat[b_mat_i];
        a_mat_i += blockDim.x;
        b_mat_i += blockDim.x;
    }
    __syncthreads();
    if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        out_mat[blockIdx.x] = sdata[0];
 }

Accelerated_Matrix Accelerated_Matrix::operator*(Accelerated_Matrix const & mat) const {
    if(mat.nrow() == 1 && mat.ncol() == 1){
        Accelerated_Matrix temp((*this).nrow(), (*this).ncol());
        double value = mat(0, 0);
        size_t shape = (*this).nrow() * (*this).ncol();
        double *d_m_buffer, *d_other_m_buffer;
        cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
        cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
        cudaMemcpy(d_other_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
        dim3 blockSize(BLOCK_SIZE);
        dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul1_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, value, shape);
        cudaDeviceSynchronize();
        cudaMemcpy(temp.m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
        cudaFree(d_m_buffer);
        cudaFree(d_other_m_buffer);
        return temp;
    }

    if( (*this).nrow() == 1 && (*this).ncol() == 1){
        Accelerated_Matrix temp(mat.nrow(), mat.ncol());
        double value = (*this)(0, 0);
        size_t shape = mat.nrow() * mat.ncol();
        double *d_m_buffer, *d_other_m_buffer;
        cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
        cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
        cudaMemcpy(d_other_m_buffer, mat.m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
        dim3 blockSize(BLOCK_SIZE);
        dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul1_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, value, shape);
        cudaDeviceSynchronize();
        cudaMemcpy(temp.m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
        cudaFree(d_m_buffer);
        cudaFree(d_other_m_buffer);
        return temp;
    }

    if( (*this).ncol() == 1 && mat.ncol() == 1 && mat.nrow() != 1){
        Accelerated_Matrix return_value(1, 1);
        double *d_m_buffer, *d_other_m_buffer, *d_output_m_buffer;
        // cudaHostGetDevicePointer(&d_m_buffer, m_buffer, 0);
        // cudaHostGetDevicePointer(&d_other_m_buffer, mat.m_buffer, 0);
        // cudaHostGetDevicePointer(&d_output_m_buffer, return_value.m_buffer, 0);
        auto shape = (*this).nrow();
        cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
        cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * shape);
        cudaMalloc((void**)&d_output_m_buffer, sizeof(double) * 1);
        cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
        cudaMemcpy(d_other_m_buffer, mat.m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
        dim3 blockSize(BLOCK_SIZE);
        dim3 numBlock(1);
        matmul2_gpu<<<numBlock, blockSize>>>(d_output_m_buffer, d_m_buffer, d_other_m_buffer, (*this).nrow());
        cudaDeviceSynchronize();
        cudaMemcpy(return_value.m_buffer, d_output_m_buffer, sizeof(double) * 1, cudaMemcpyDeviceToHost);
        cudaFree(d_m_buffer);
        cudaFree(d_other_m_buffer);
        cudaFree(d_output_m_buffer);
        return return_value;
    }
    
    if (mat.ncol() == 1) {
        Accelerated_Matrix return_value((*this).nrow(), mat.ncol());
        double *d_m_buffer, *d_other_m_buffer, *d_output_m_buffer;
        cudaMalloc((void**)&d_m_buffer, sizeof(double) * (*this).nrow() * (*this).ncol());
        cudaMalloc((void**)&d_other_m_buffer, sizeof(double) * mat.nrow() * mat.ncol());
        cudaMalloc((void**)&d_output_m_buffer, sizeof(double) * (*this).nrow() * mat.ncol());
        cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * (*this).nrow() * (*this).ncol(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_other_m_buffer, mat.m_buffer, sizeof(double) * mat.nrow() * mat.ncol(), cudaMemcpyHostToDevice);
        dim3 blockSize(BLOCK_SIZE);
        dim3 numBlock((*this).nrow());
        matmul2_gpu<<<numBlock, blockSize>>>(d_output_m_buffer, d_m_buffer, d_other_m_buffer, (*this).ncol());
        cudaDeviceSynchronize();
        cudaMemcpy(return_value.m_buffer, d_output_m_buffer, sizeof(double) * (*this).nrow() * mat.ncol(), cudaMemcpyDeviceToHost);
        cudaFree(d_m_buffer);
        cudaFree(d_other_m_buffer);
        cudaFree(d_output_m_buffer);
        return return_value;
    }
    
    Accelerated_Matrix result((*this).nrow(), mat.ncol());
    size_t tsize = 32;
    size_t max_i = (*this).nrow();
    size_t max_j = mat.ncol();
    size_t max_k = (*this).ncol();
    for (int t = 0; t < number_of_threads; ++t) {
        size_t start_i = t * max_i / number_of_threads;
        size_t end_i = (t + 1) * max_i / number_of_threads;
        for (size_t j = 0; j < max_j; j += tsize) {
            for (size_t k = 0; k < max_k; k += tsize) {
                size_t upper_j = std::min(j + tsize, max_j);
                size_t upper_k = std::min(k + tsize, max_k);
                    for (size_t ii = start_i; ii < end_i; ++ii) {
                        for (size_t jj = j; jj < upper_j; ++jj) {
                            double sum = .0;
                            for (size_t kk = k; kk < upper_k; ++kk) {
                                sum += (*this)(ii, kk) * mat(kk, jj);
                            }
                            result(ii, jj) += sum;
                        }
                    }
            }
        }
    }

    return result;
}

Accelerated_Matrix Accelerated_Matrix::operator*(double const & other) const {
    Accelerated_Matrix temp = (*this);
    return temp *= other;
}

Accelerated_Matrix& Accelerated_Matrix::operator*=(double const & other) {
    size_t shape = m_nrow * m_ncol;
    double *d_m_buffer;
    cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
    cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
    imul_gpu<<<numBlock, blockSize>>>(d_m_buffer, other, shape);
    cudaDeviceSynchronize();
    cudaMemcpy(m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
    cudaFree(d_m_buffer);
    return (*this);
}


bool Accelerated_Matrix::operator==(Accelerated_Matrix const & mat) const
{
    if (mat.m_ncol != m_ncol || mat.m_nrow != m_nrow) return false;
    for (size_t i = 0; i < mat.m_nrow; ++i){
        for (size_t j = 0; j < mat.m_ncol; ++j){
            if(mat(i, j) != m_buffer[i*m_nrow+j]) return false;
        }
    }
    return true;
}

double Accelerated_Matrix::operator() (size_t row, size_t col) const
{
    if (row >= m_nrow || col >= m_ncol){
        throw std::out_of_range("Accelerated_Matrix::operator(): index out of range");
    }

    return m_buffer[index(row, col)];
}

double & Accelerated_Matrix::operator() (size_t row, size_t col)
{
    if (row >= m_nrow || col >= m_ncol){
        throw std::out_of_range("Accelerated_Matrix::operator(): index out of range");
    }

    return m_buffer[index(row, col)];
}

int & Accelerated_Matrix::set_number_of_threads(){
    return this -> number_of_threads;
}

int const & Accelerated_Matrix::get_number_of_threads() const{
    return this -> number_of_threads;
}

 __global__ void norm_gpu(double& output, double* a_mat, int ncol) {
    __shared__ double sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int a_mat_i = tid;
    sdata[tid] = 0;
    while (a_mat_i < ncol) {
        sdata[tid] += a_mat[a_mat_i] * a_mat[a_mat_i];
        a_mat_i += blockDim.x;
    }
    __syncthreads();
    if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        output = sdata[0];
 }

double Accelerated_Matrix::norm()
{
    double sum = 0.0;
    size_t shape = m_nrow * m_ncol;
    double *d_m_buffer;
    cudaMalloc((void**)&d_m_buffer, sizeof(double) * shape);
    cudaMemcpy(d_m_buffer, m_buffer, sizeof(double) * shape, cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock(1);
    norm_gpu<<<numBlock, blockSize>>>(sum, d_m_buffer, shape);
    cudaDeviceSynchronize();
    cudaMemcpy(m_buffer, d_m_buffer, sizeof(double) * shape, cudaMemcpyDeviceToHost);
    cudaFree(d_m_buffer);
    return sqrt(sum);
}

double* Accelerated_Matrix::data() const { return m_buffer; }

size_t Accelerated_Matrix::nrow() const { return m_nrow; }
size_t Accelerated_Matrix::ncol() const { return m_ncol; }

void Accelerated_Matrix::reset_buffer(size_t nrow, size_t ncol){
    if (m_buffer) {
        free(m_buffer);
    }
    const size_t nelement = nrow * ncol;
    if (nelement) {
        m_buffer = (double*)malloc(sizeof(double) * nelement);
    } else {
        m_buffer = nullptr;
    }
    m_nrow = nrow;
    m_ncol = ncol;
}

}