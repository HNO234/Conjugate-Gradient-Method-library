#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>

class Matrix {

public:
    Matrix() = default;
    Matrix(size_t nrow, size_t ncol);
    Matrix(const Matrix &mat);
    Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec);
    Matrix(std::vector<double> const & vec);
    Matrix(std::vector<std::vector<double>> const & vec2d);
    ~Matrix();
    Matrix(Matrix && other);
    Matrix & operator=(std::vector<double> const & vec);
    Matrix & operator=(std::vector<std::vector<double>> const & vec2d);
    Matrix & operator=(Matrix const & mat);
    Matrix & operator=(Matrix && other);
    Matrix operator+(Matrix const & other);
    Matrix operator-(Matrix const & other);
    Matrix operator-();
    Matrix operator*(Matrix const & other);
    Matrix operator*(double const & other);
    bool operator==(Matrix const & mat) const;
    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col);
    double norm();
    
    double * data() const;
    size_t nrow() const;
    size_t ncol() const;
    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const
    {
        return std::vector<double>(m_buffer, m_buffer+size());
    }

    std::vector<std::vector<double>> buffer_vector2d() const
    {
        std::vector<std::vector<double>> vec2d;
        for (size_t i = 0; i < m_nrow; i++)
        {
            std::vector<double> vec;
            for (size_t j = 0; j < m_ncol; j++)
            {
                vec.push_back(m_buffer[i * m_ncol + j]);
            }
            vec2d.push_back(vec);
        }
        return vec2d;
    }

private:
    void reset_buffer(size_t nrow, size_t ncol);
    size_t index(size_t row, size_t col) const{return m_ncol*row + col;}
    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double * m_buffer = nullptr;
};

std::ostream & operator << (std::ostream & ostr, Matrix const & mat);

Matrix multiply_tile(Matrix const& mat1, Matrix const& mat2, size_t tsize);
Matrix multiply_naive(Matrix const& mat1, Matrix const& mat2);

#endif
