#ifndef NUMERIC_HPP
#define NUMERIC_HPP

#include "cnn.hpp"

void onehot(real_t* labels,
            real_t* onehot_vector,
            const int& nums,
            const int& rows,
            const int& cols,
            const int& num_classes)
{
    for(int num = 0; num < nums; num++)
        for(int row = 0; row < rows; row++)
            for(int col = 0; col < cols; col++)
                {
                     real_t label = *(labels + num * rows * cols
                                    + row * cols
                                    + col);
                     int cls = static_cast<int>(label);
                     *(onehot_vector + num * rows * cols * num_classes
                            + row * cols * num_classes
                            + col * num_classes
                            + cls) = 1;
                }
}


void logsumexp(real_t* value,real_t* result,
               const int& nums,const int& rows,
               const int& cols,const int& channels)
{
    for(int num = 0; num < nums; num++)
        for(int row = 0; row < rows; row++)
            for(int col = 0; col < cols; col++)
            {
                real_t max_value = 0;
                for(int channel = 0; channel < channels; channel++)
                {
                    real_t tmp_value = *(value + num * rows * cols * channels
                                + row * cols * channels
                                + col * channels
                                + channel);
                    if(tmp_value > max_value)
                        max_value = tmp_value;
                }
                real_t sum  = 0;
                for(int channel = 0; channel < channels; channel++)
                {
                    sum += std::exp(*(value + num * rows * cols * channels
                                + row * cols * channels
                                + col * channels
                                + channel) - max_value);
                }
                *(result + num * rows * cols
                            + row * cols
                            + col) = std::log(sum) + max_value;
        }
}


void softmax(real_t* value,real_t* result,
             const int& nums,const int& rows,
             const int& cols,const int& channels)
{
    for(int num = 0; num < nums; num++)
        for(int row = 0; row < rows; row++)
            for(int col = 0; col < cols; col++)
                {
                    real_t max_value = 0;
                    for(int channel = 0; channel < channels; channel++)
                    {
                        real_t tmp_value = *(value + num * rows * cols * channels
                                    + row * cols * channels
                                    + col * channels
                                    + channel);
                        if(tmp_value > max_value)
                            max_value = tmp_value;
                    }
                    real_t sum  = 0;
                    for(int channel = 0; channel < channels; channel++)
                    {
                        sum += std::exp(*(value + num * rows * cols * channels
                                    + row * cols * channels
                                    + col * channels
                                    + channel) - max_value);
                    }
                    for(int channel = 0; channel < channels; channel++)
                    {
                        *(result + num * rows * cols * channels
                                    + row * cols * channels
                                    + col * channels
                                    + channel) =
                        *(result + num * rows * cols * channels
                                    + row * cols * channels
                                    + col * channels
                                    + channel) / sum;
                    }
                }
}

#endif
