#ifndef __MATRIX_NAIVE_HPP__
#define __MATRIX_NAIVE_HPP__

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Matrix{

class Naive_Matrix{

public:
    Naive_Matrix() = default;
    Naive_Matrix(size_t nrow, size_t ncol);
    Naive_Matrix(const Naive_Matrix &mat);
    Naive_Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec);
    Naive_Matrix(std::vector<double> const & vec);
    Naive_Matrix(std::vector<std::vector<double>> const & vec2d);
    ~Naive_Matrix();
    Naive_Matrix(Naive_Matrix && other);
    Naive_Matrix & operator=(std::vector<double> const & vec);
    Naive_Matrix & operator=(std::vector<std::vector<double>> const & vec2d);
    Naive_Matrix & operator=(Naive_Matrix const & mat);
    Naive_Matrix & operator=(Naive_Matrix && other);
    Naive_Matrix operator+(Naive_Matrix const & other) const;
    Naive_Matrix& operator+=(Naive_Matrix const & other);
    Naive_Matrix operator-(Naive_Matrix const & other) const;
    Naive_Matrix& operator-=(Naive_Matrix const & other);
    Naive_Matrix operator-() const;
    Naive_Matrix operator*(Naive_Matrix const & other) const;
    Naive_Matrix operator*(double const & other) const;
    Naive_Matrix& operator*=(double const & other);
    bool operator==(Naive_Matrix const & mat) const;
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

std::ostream & operator << (std::ostream & ostr, Naive_Matrix const & mat);

Naive_Matrix multiply_tile(Naive_Matrix const& mat1, Naive_Matrix const& mat2, size_t tsize);
Naive_Matrix multiply_naive(Naive_Matrix const& mat1, Naive_Matrix const& mat2);

}

#endif
