#ifndef __MATRIX_ACC_HPP__
#define __MATRIX_ACC_HPP__

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <thread>

namespace Matrix{
class Accelerated_Matrix{

public:
    Accelerated_Matrix() = default;
    Accelerated_Matrix(size_t nrow, size_t ncol);
    Accelerated_Matrix(const Accelerated_Matrix &mat);
    Accelerated_Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec);
    Accelerated_Matrix(std::vector<double> const & vec);
    Accelerated_Matrix(std::vector<std::vector<double>> const & vec2d);
    ~Accelerated_Matrix();
    Accelerated_Matrix(Accelerated_Matrix && other);
    Accelerated_Matrix & operator=(std::vector<double> const & vec);
    Accelerated_Matrix & operator=(std::vector<std::vector<double>> const & vec2d);
    Accelerated_Matrix & operator=(Accelerated_Matrix const & mat);
    Accelerated_Matrix & operator=(Accelerated_Matrix && other);
    Accelerated_Matrix operator+(Accelerated_Matrix const & other) const;
    Accelerated_Matrix& operator+=(Accelerated_Matrix const & other);
    Accelerated_Matrix operator-(Accelerated_Matrix const & other) const;
    Accelerated_Matrix& operator-=(Accelerated_Matrix const & other);
    Accelerated_Matrix operator-() const;
    Accelerated_Matrix operator*(Accelerated_Matrix const & other) const;
    Accelerated_Matrix operator*(double const & other) const;
    Accelerated_Matrix& operator*=(double const & other);
    bool operator==(Accelerated_Matrix const & mat) const;
    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col);
    int const & get_number_of_threads() const;
    int       & set_number_of_threads();

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
    static int number_of_threads;
};

}

#endif
