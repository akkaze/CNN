#ifndef BLAS_HPP
#define BLAS_HPP

#include "cnn.hpp"

#include <Eigen/Dense>

typedef Eigen::Matrix<real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MyMatrix;
//A(m * k)
//B(k * n)
//C(m * n)
void prod(real_t* A,real_t* B,real_t *C,int m,int n,int k)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);
}

void prodEigen(real_t* A,real_t* B,real_t* C,int m,int n,int k)
{
    MyMatrix MatrixA = Eigen::Map<MyMatrix>(A,m,k);
    MyMatrix MatrixB = Eigen::Map<MyMatrix>(B,k,n);
    MyMatrix MatrixC = MatrixA * MatrixB;
    memcpy(C,MatrixC.data(),sizeof(real_t) * MatrixC.rows() * MatrixC.cols());
}

//A(k * m)
//B(k * n)
//C(m * n)
void transProd(real_t* A,real_t* B,real_t *C,int m,int n,int k)
{
    cblas_gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                m, n, k, 1.0, A, m, B, n, 0.0, C, n);
}

void transProdEigen(real_t* A,real_t* B,real_t* C,int m,int n,int k)
{
    MyMatrix MatrixA = Eigen::Map<MyMatrix>(A,k,m);
    MyMatrix MatrixB = Eigen::Map<MyMatrix>(B,k,n);
    MyMatrix MatrixC = MatrixA.transpose() * MatrixB;
    memcpy(C,MatrixC.data(),sizeof(real_t) * MatrixC.rows() * MatrixC.cols());
}
//A(k * m)
//B(k * n)
//C(m * n)
void transProdPlus(real_t* A,real_t* B,real_t *C,int m,int n,int k)
{
    cblas_gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                m, n, k, 1.0, A, m, B, n, 1.0, C, n);
}
void transProdPlusEigen(real_t* A,real_t* B,real_t* C,int m,int n,int k)
{
    MyMatrix MatrixA = Eigen::Map<MyMatrix>(A,k,m);
    MyMatrix MatrixB = Eigen::Map<MyMatrix>(B,k,n);
    MyMatrix MatrixC = Eigen::Map<MyMatrix>(C,m,n);
    MatrixC = MatrixA.transpose() * MatrixB + MatrixC;
    memcpy(C,MatrixC.data(),sizeof(real_t) * MatrixC.rows() * MatrixC.cols());
}
//A(m * k)
//B(n * k)
//C(m * n)
void prodTrans(real_t* A,real_t* B,real_t *C,int m,int n,int k)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0, A, k, B, k, 0.0, C, n);
}
void prodTransEigen(real_t* A,real_t* B,real_t* C,int m,int n,int k)
{
    MyMatrix MatrixA = Eigen::Map<MyMatrix>(A,m,k);
    MyMatrix MatrixB = Eigen::Map<MyMatrix>(B,n,k);
    MyMatrix MatrixC = MatrixA * MatrixB.transpose();
    memcpy(C,MatrixC.data(),sizeof(real_t) * MatrixC.rows() * MatrixC.cols());
}
//A(m * k)
//B(n * k)
//C(m * n)
void prodTransPlus(real_t* A,real_t* B,real_t *C,int m,int n,int k)
{
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0, A, k, B, k, 1.0, C, n);
}

void prodTransPlusEigen(real_t* A,real_t* B,real_t* C,int m,int n,int k)
{
    MyMatrix MatrixA = Eigen::Map<MyMatrix>(A,m,k);
    MyMatrix MatrixB = Eigen::Map<MyMatrix>(B,n,k);
    MyMatrix MatrixC = Eigen::Map<MyMatrix>(B,m,n);
    MatrixC = MatrixA * MatrixB.transpose() + MatrixC;
    memcpy(C,MatrixC.data(),sizeof(real_t) * MatrixC.rows() * MatrixC.cols());
}
#endif // BLAS_HPP

