#include <_matrix.hpp>

namespace Matrix{

Naive_Matrix::Naive_Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol)
{
    size_t nelement = nrow * ncol;
    reset_buffer(nrow, ncol);
    memset(m_buffer, 0, nelement * sizeof(double));
}

Naive_Matrix::Naive_Matrix(Naive_Matrix const &mat) : m_nrow(mat.m_nrow), m_ncol(mat.m_ncol)
{
    reset_buffer(mat.m_nrow, mat.m_ncol);
    for (size_t i=0; i<m_nrow; ++i){
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*this)(i,j) = mat(i,j);
        }
    }
}

Naive_Matrix::Naive_Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec) : m_nrow(nrow), m_ncol(ncol)
{
    if (vec.size() != nrow * ncol){
        throw std::out_of_range("Naive_Matrix::Naive_Matrix(): vector size differs from matrix size");
    }
        reset_buffer(nrow, ncol);
        (*this) = vec;
}

Naive_Matrix::Naive_Matrix(std::vector<double> const & vec) : m_nrow(vec.size()), m_ncol(1)
{
    reset_buffer(vec.size(), 1);
    (*this) = vec;
}

Naive_Matrix::Naive_Matrix(std::vector<std::vector<double>> const & vec) : m_nrow(vec.size()), m_ncol(vec[0].size())
{
    reset_buffer(m_nrow, m_ncol);
    (*this) = vec;
}

Naive_Matrix::~Naive_Matrix()
{
    reset_buffer(0, 0);
}

Naive_Matrix::Naive_Matrix(Naive_Matrix && other)
  : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
    reset_buffer(0, 0);
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_buffer, other.m_buffer);
}

Naive_Matrix & Naive_Matrix::operator=(Naive_Matrix && other){
        if (this == &other) { return *this; }
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
        return *this;
}

Naive_Matrix & Naive_Matrix::operator=(std::vector<double> const & vec)
{
    if (size() != vec.size()){
        throw std::out_of_range("number of elements mismatch");
    }
    size_t k = 0;
    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*this)(i,j) = vec[k];
            ++k;
        }
    }
    return *this;
}

Naive_Matrix & Naive_Matrix::operator=(std::vector<std::vector<double>> const & vec2d)
{
    if (m_nrow != vec2d.size() || m_ncol != vec2d[0].size()){
        throw std::out_of_range("number of elements mismatch");
    }
    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*this)(i,j) = vec2d[i][j];
        }
    }
    return *this;
}

Naive_Matrix & Naive_Matrix::operator=(Naive_Matrix const & other)
{
    if (this == &other) { return *this; }
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
    }
    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*this)(i,j) = other(i,j);
        }
    }
    return *this;
}

Naive_Matrix Naive_Matrix::operator+(Naive_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }    
    Naive_Matrix temp(m_nrow, m_ncol);
    for(size_t i = 0 ; i < m_nrow; ++i){
        for(size_t j = 0; j < m_ncol; ++j){
            temp(i,j) = (*this)(i,j) + other(i,j);
        }
    }
    return temp;
}

Naive_Matrix Naive_Matrix::operator-(Naive_Matrix const & other){
    if(( nrow() != other.nrow()) || ( ncol() != other.ncol())){
        throw std::out_of_range("Number of elements mismatch.");
    }    
    Naive_Matrix temp(m_nrow, m_ncol);
    for(size_t i = 0 ; i < m_nrow; ++i){
        for(size_t j = 0; j < m_ncol; ++j){
            temp(i,j) = (*this)(i,j) - other(i,j);
        }
    }
    return temp;
}

Naive_Matrix Naive_Matrix::operator-(){
    Naive_Matrix temp(m_nrow, m_ncol);
    for(size_t i = 0 ; i < m_nrow; ++i){
        for(size_t j = 0; j < m_ncol; ++j){
            temp(i,j) = -(*this)(i,j);
        }
    }
    return temp;
}

Naive_Matrix Naive_Matrix::operator*(Naive_Matrix const & mat){

    if( mat.nrow() == 1 && mat.ncol() == 1){
        Naive_Matrix temp(m_nrow, m_ncol);
        for(size_t i = 0 ; i < m_nrow; ++i){
            for(size_t j = 0; j < m_ncol; ++j){
                temp(i,j) = (*this)(i,j) * mat(0,0);
            }
        }
        return temp;
    }

    if( (*this).nrow() == 1 && (*this).ncol() == 1){
        Naive_Matrix temp(mat.nrow(), mat.ncol());
        for(size_t i = 0 ; i < mat.nrow(); ++i){
            for(size_t j = 0; j < mat.ncol(); ++j){
                temp(i,j) = (*this)(0,0) * mat(i,j);
            }
        }
        return temp;
    }

    if ((*this).ncol() == 1 && mat.ncol() == 1 && mat.nrow() != 1) {
        Naive_Matrix return_value(1, 1);
        for (size_t i = 0; i < (*this).nrow(); ++i) {
            return_value(0, 0) += (*this)(i, 0) * mat(i, 0);
        }
        return return_value;
    }

    if ((*this).ncol() != mat.nrow()) {
        throw std::out_of_range("Number of columns in the first matrix must be equal to the number of rows in the second matrix.");
    }

    Naive_Matrix result((*this).nrow(), mat.ncol());

    for (size_t i = 0; i < result.nrow(); ++i) {
        for (size_t j = 0; j < result.ncol(); ++j) {
            for (size_t k = 0; k < (*this).ncol(); ++k) {
                result(i, j) += (*this)(i, k) * mat(k, j);
            }
        }
    }
    return result;

    return result;
}

Naive_Matrix Naive_Matrix::operator*(double const & other){
    Naive_Matrix temp(m_nrow, m_ncol);
    for(size_t i = 0 ; i < m_nrow; ++i){
        for(size_t j = 0; j < m_ncol; ++j){
            temp(i,j) = (*this)(i,j) * other;
        }
    }
    return temp;
}

bool Naive_Matrix::operator==(Naive_Matrix const & mat) const
{
    if (mat.m_ncol != m_ncol || mat.m_nrow != m_nrow) return false;
    for (size_t i = 0; i < mat.m_nrow; ++i)
    {
        for (size_t j = 0; j < mat.m_ncol; ++j)
        {
            if(mat(i, j) != m_buffer[i*m_nrow+j])return false;
        }
    }
    return true;
}

double  Naive_Matrix::operator() (size_t row, size_t col) const
{
    if (row >= m_nrow || col >= m_ncol){
        throw std::out_of_range("Naive_Matrix::operator(): index out of range");
    }

    return m_buffer[index(row, col)];
}

double & Naive_Matrix::operator() (size_t row, size_t col)
{
    if (row >= m_nrow || col >= m_ncol){
        throw std::out_of_range("Naive_Matrix::operator(): index out of range");
    }

    return m_buffer[index(row, col)];
}

double Naive_Matrix::norm()
{
    double sum = 0.0;
    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            sum += (*this)(i,j) * (*this)(i,j);
        }
    }
    return sqrt(sum);
}

double* Naive_Matrix::data() const { return m_buffer; }

size_t Naive_Matrix::nrow() const { return m_nrow; }
size_t Naive_Matrix::ncol() const { return m_ncol; }

void Naive_Matrix::reset_buffer(size_t nrow, size_t ncol){
    if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
    if (nelement) { m_buffer = new double[nelement]; }
    else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
}

std::ostream & operator << (std::ostream & ostr, Naive_Matrix const & mat)
{
    for (size_t i=0; i<mat.nrow(); ++i)
    {
        ostr << std::endl << " ";
        for (size_t j=0; j<mat.ncol(); ++j)
        {
            ostr << " " << std::setw(2) << mat(i, j);
        }
    }

    return ostr;
}

Naive_Matrix multiply_tile(Naive_Matrix const& mat1, Naive_Matrix const& mat2, size_t tsize)
{

    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
            "the number of first matrix column "
            "differs from that of second matrix row");
    }

    Naive_Matrix result(mat1.nrow(), mat2.ncol());
    size_t max_i = mat1.nrow();
    size_t max_j = mat2.ncol();
    size_t max_k = mat1.ncol();

    for (size_t i = 0; i < max_i; i += tsize)
    {
        for (size_t j = 0; j < max_j; j += tsize)
        {
            for (size_t k = 0; k < max_k; k += tsize)
            {
                size_t upper_i = std::min(i + tsize, max_i);
                size_t upper_j = std::min(j + tsize, max_j);
                size_t upper_k = std::min(k + tsize, max_k);
                for (size_t ii = i; ii < upper_i ; ++ii)
                {
                    for (size_t jj = j; jj < upper_j ; ++jj) 
		            {
		    	        double sum = .0;
                        for (size_t kk = k; kk < upper_k; ++kk)
                        {
                            sum += mat1(ii, kk) * mat2(kk, jj);
                        }
			            result(ii, jj) += sum;
                    }
                }
            }
        }
    }

    return result;
}

Naive_Matrix multiply_naive(Naive_Matrix const& mat1, Naive_Matrix const& mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
            "the number of first matrix column "
            "differs from that of second matrix row");
    }

    Naive_Matrix ret(mat1.nrow(), mat2.ncol());

    for (size_t j=0; j<mat1.nrow(); ++j)
    {
        for (size_t k=0; k<ret.ncol(); ++k)
        {
            for (size_t i=0; i<ret.ncol(); ++i)
            {
                ret(i,k) += mat1(i,j) * mat2(j,k);
            }
        }
    }

    return ret;
}

}