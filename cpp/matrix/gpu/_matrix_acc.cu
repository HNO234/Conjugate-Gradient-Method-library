#include <_matrix_acc.hpp>

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

Accelerated_Matrix& Accelerated_Matrix::operator+=(Accelerated_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }
    size_t i, shape = m_nrow * m_ncol;
// #pragma omp parallel for private(i) num_threads(number_of_threads)
//     for(i = 0 ; i < shape; ++i){
//         m_buffer[i] += other.m_buffer[i];
//     }
    double *d_m_buffer, *d_other_m_buffer;
    cudaHostGetDevicePointer(&d_m_buffer, m_buffer, 0);
    cudaHostGetDevicePointer(&d_other_m_buffer, other.m_buffer, 0);
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlock((shape + BLOCK_SIZE - 1) / BLOCK_SIZE);
    iadd_gpu<<<numBlock, blockSize>>>(d_m_buffer, d_other_m_buffer, shape);
    cudaDeviceSynchronize();
    return (*this);
}

 __global__ void iadd_gpu(float* a_mat, float* b_mat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a_mat[i] += b_mat[i];
 }

Accelerated_Matrix Accelerated_Matrix::operator-(Accelerated_Matrix const & other) const {
    Accelerated_Matrix temp = *this;
    return temp -= other;
}

Accelerated_Matrix& Accelerated_Matrix::operator-=(Accelerated_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }
    size_t i, shape = m_nrow * m_ncol;
// #pragma omp parallel for private(i) num_threads(number_of_threads)
    for(i = 0 ; i < shape; ++i){
        m_buffer[i] -= other.m_buffer[i];
    }
    return (*this);
}

Accelerated_Matrix Accelerated_Matrix::operator-() const {
    Accelerated_Matrix temp(m_nrow, m_ncol);
    size_t i, shape = m_nrow * m_ncol;
    // #pragma omp parallel for private(i) num_threads(number_of_threads)
    for(i = 0 ; i < shape; ++i){
        temp.m_buffer[i] = -m_buffer[i];
    }
    return temp;
}


Accelerated_Matrix Accelerated_Matrix::operator*(Accelerated_Matrix const & mat) const {
    if( mat.nrow() == 1 && mat.ncol() == 1){
        Accelerated_Matrix temp(m_nrow, m_ncol);
        size_t i, shape = m_nrow * m_ncol;
        double value = mat(0,0);
        // #pragma omp parallel for private(i, value) num_threads(number_of_threads)
        for(i = 0 ; i < shape; ++i){
            temp.m_buffer[i] = m_buffer[i] * value;
        }
        return temp;
    }

    if( (*this).nrow() == 1 && (*this).ncol() == 1){
        Accelerated_Matrix temp(mat.nrow(), mat.ncol());
        size_t i, shape = m_nrow * m_ncol;
        double value = (*this)(0,0);
        // #pragma omp parallel for private(i, value) num_threads(number_of_threads)
        for(i = 0 ; i < shape; ++i){
            temp.m_buffer[i] = mat.m_buffer[i] * value;
        }
        return temp;
    }

    if( (*this).ncol() == 1 && mat.ncol() == 1 && mat.nrow() != 1){
        Accelerated_Matrix return_value(1,1);
        double sum = .0f;
        size_t i;
        // #pragma omp parallel for private(i) reduction(+:sum) num_threads(number_of_threads)
        for(i = 0; i < (*this).nrow(); ++i){
            sum += (*this).m_buffer[i] * mat.m_buffer[i];
        }
        return_value.m_buffer[0] = sum;
        return return_value;
    }
    
    if (mat.ncol() == 1) {
        Accelerated_Matrix return_value((*this).nrow(), mat.ncol());
        size_t i;
// #pragma omp parallel for private(i) num_threads(number_of_threads)
        for(i = 0; i < (*this).nrow(); ++i){
            size_t mat_this_begin = i * (*this).ncol();
            size_t j = 0;
            size_t j_end = (*this).ncol();
            double sum = .0f;
            for (j = 0; j < j_end; j++)
                sum += (*this).m_buffer[mat_this_begin + j] * mat.m_buffer[j];
            return_value.m_buffer[i] = sum;
        }
        return return_value;
    }
    
    Accelerated_Matrix result((*this).nrow(), mat.ncol());
    size_t tsize = 32;
    size_t max_i = (*this).nrow();
    size_t max_j = mat.ncol();
    size_t max_k = (*this).ncol();

    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);
    
    for (int t = 0; t < number_of_threads; ++t) {
        threads.emplace_back([&, t, max_i, max_j, max_k, tsize]() {
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
        );
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
    
}

Accelerated_Matrix Accelerated_Matrix::operator*(double const & other) const {
    Accelerated_Matrix temp = (*this);
    return temp *= other;
}

Accelerated_Matrix& Accelerated_Matrix::operator*=(double const & other) {
    size_t i, shape = m_nrow * m_ncol;
// #pragma omp parallel for private(i) num_threads(number_of_threads)
    for(i = 0 ; i < shape; ++i){
        m_buffer[i] *= other;
    }
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

double Accelerated_Matrix::norm()
{
    double sum = 0.0;
    size_t i;
    size_t shape = m_nrow * m_ncol;
    // #pragma omp parallel for private(i) reduction(+:sum) num_threads(number_of_threads)
    for (i = 0; i< shape; ++i)
    {
        sum += m_buffer[i] * m_buffer[i];
    }
    return sqrt(sum);
}

double* Accelerated_Matrix::data() const { return m_buffer; }

size_t Accelerated_Matrix::nrow() const { return m_nrow; }
size_t Accelerated_Matrix::ncol() const { return m_ncol; }

void Accelerated_Matrix::reset_buffer(size_t nrow, size_t ncol){
    if (m_buffer) {
        cudaFreeHost(m_buffer);
    }
    const size_t nelement = nrow * ncol;
    if (nelement) {
        cudaHostAlloc(&m_buffer, sizeof(double) * nelement, cudaHostAllocMapped);
    } else {
        m_buffer = nullptr;
    }
    m_nrow = nrow;
    m_ncol = ncol;
}

}