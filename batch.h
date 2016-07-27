#ifndef TENSOR_H
#define TENSOR_H

#include "cnn.hpp"

class Batch
{
public:
    Batch();
    Batch(const int& nums,
          const int& rows,
          const int& cols,
          const int& channels)
    {
        nums_ = nums;
        channels_ = channels;
        rows_ = rows;
        cols_ = cols;
        size_ = nums *
                channels *
                rows *
                cols;
        value_ = new real_t[size_];
        memset(value_,0,sizeof(real_t) * size_);

        gradient_ = new real_t[size_];
        memset(gradient_,0,sizeof(real_t) * size_);
    }

    void Init(const int& nums,
               const int& rows,
               const int& cols,
               const int& channels)
    {
        size_ = nums *
                channels *
                rows *
                cols;
        value_ = new real_t[size_];

        memset(value_,0,sizeof(real_t) * size_);

        gradient_ = new real_t[size_];
        memset(gradient_,0,sizeof(real_t) * size_);
    }

    virtual ~Batch()
    {
        if(value_)
            delete[] value_;
        if(gradient_)
            delete[] gradient_;
    }

    //getters of dimensions
    int get_nums() const
    {
        return nums_;
    }

    int get_rows() const
    {
        return rows_;
    }

    int get_cols() const
    {
        return cols_;
    }

    int get_channels() const
    {
        return channels_;
    }

    int get_size() const
    {
        return size_;
    }


    //setters of value and gradient
    real_t* value() const
    {
        return value_;
    }

    real_t* gradient() const
    {
        return gradient_;
    }

    //setters of dimensions
    void set_value(real_t* value)
    {
        value_ = value;
    }


    void set_nums(const int& nums)
    {
        nums_ = nums;
    }

    void set_channels(const int& channels)
    {
        channels_ = channels;
    }

    void set_rows(const int& rows)
    {
        rows_ = rows;
    }

    void set_cols(const int& cols)
    {
        cols_ = cols;
    }

    void set_dimensions(const int& nums,
                       const int& rows,
                       const int& cols,
                       const int& channels)
    {
        nums_ = nums;
        rows_ = rows;
        cols_ = cols;
        channels_ = channels;
    }

    //get the address of value or gradient buuffer of certain offset
    real_t* data_ptr(int num,
                   int col,
                   int row,
                   int channel)
    {
        return (value_ +
                num * cols_ * rows_ * channels_ +
                col * rows_ * channels_ +
                row * channels_ +
                channel);
    }

    real_t* data_ptr(int idx)
    {
        return (value_ + idx);
    }

    real_t* grad_ptr(int num,
                   int col,
                   int row,
                   int channel)
    {
        return (gradient_ +
                num * cols_ * rows_ * channels_ +
                col * rows_ * channels_ +
                row * channels_ +
                channel);
    }

    real_t* grad_ptr(int idx)
    {
        return (gradient_ + idx);
    }

private:
    int nums_;
    int channels_;
    int rows_;
    int cols_;

    int size_;

    real_t* value_;
    real_t* gradient_;
};

#endif // TENSOR_H
