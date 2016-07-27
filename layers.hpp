#ifndef LAYERS_HPP
#define LSYERS_HPP
#include "cnn.hpp"

class Param;

class Node
{
private:
    int nums_;
    int channels_;
    int rows_;
    int cols_;

    int size_;

    real_t* value_;
    real_t* gradient_;

    std::vector<Node*> inputs_;

public:
    Node() {}
    Node(const int& nums,
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

    virtual ~Node()
    {
        if(value_)
            delete[] value_;
        if(gradient_)
            delete[] gradient_;
    }
    //getters of dimensions
    const int& nums() const
    {
        return nums_;
    }

    const int& rows() const
    {
        return rows_;
    }

    const int& cols() const
    {
        return cols_;
    }

    const int& channels() const
    {
        return channels_;
    }

    const int& size() const
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

    std::vector<Node*>& inputs()
    {
        return inputs_;
    }

    void set_inputs(const std::vector<Node*>& inputs)
    {
        inputs_ = inputs;
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

    const bool has_learnable_params() const
    {
        return !learnable_params_.empty();
    }

    std::vector<Param*>& learnable_params()
    {
        return learnable_params_;
    }

    virtual void Forward() = 0;
    virtual void Backward() = 0;
    virtual void SetUp() = 0;
private:
    std::vector<Param*> learnable_params_;
};

//base class of operation
class Op : public Node
{
public:
    Op(const std::vector<Node*>& inputs)
    {
        set_inputs(inputs);
    }

    //set input batches,and the size of output batch
    Op(const std::vector<Node*>& inputs,
       int nums,
       int rows,
       int cols,
       int channels) :
        Node(nums,rows,cols,channels)
    {
        set_inputs(inputs);
    }

    ~Op()
    {
        std::vector<Node*> inputs = this->inputs();
        inputs.clear();
    }
};


//not real_tly the layer that read data,note that data is fed to the whole network form the begin
class Data : public Node
{
public:
    //the mean value of the whole image sets
    real_t *mean_;

    Data(real_t* data_mean)
    {
        mean_ = data_mean;
    }
    Data(real_t* data_mean,
         int nums,
         int channels,
         int rows,
         int cols)
        : Node(nums,channels,rows,cols)
    {
        mean_ = data_mean;
    }

    void SetUp()
    {
        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();

        this->Init(nums,rows,cols,channels);
    }
    void Forward()
    {
        if(mean_ != NULL)
        {
            const int& nums = this->nums();
            const int& rows = this->rows();
            const int& cols = this->cols();
            const int& channels = this->channels();
            const int& size = this->size();

            real_t* gradient = this->gradient();

            for(int num = 0; num < nums; num++)
                for(int row = 0; row < rows; row++)
                    for(int col = 0; col < cols; col++)
                        for(int channel = 0; channel < channels; channel++)
                        {
                            real_t* data = data_ptr(num,row,col,num);
                            *data -= *mean_;
                        }
            memset(gradient,0,sizeof(real_t) * size);
        }
    }
    void Backward() {}
};

//base class of all preprocessions
class Preprocess : public Op {};

//The parameter class,note that all paramters are acted as a matrix
class Param
{
private:
    bool fixed_;
    int size_;
    real_t* value_;
    real_t* gradient_;
public:
    Param(const int size,
          const bool fixed = false)
    {
        fixed_ = fixed;
        size_ = size;
        value_ = new real_t[size];
        gradient_ = new real_t[size];
    }

    ~Param()
    {
        if(value_)
            delete[] value_;
        if(gradient_)
            delete[] gradient_;
    }

public:
    const bool& fixed()
    {
        return fixed_;
    }

    const int& size()
    {
        return size_;
    }

    real_t* value()
    {
        return value_;
    }

    real_t* gradient()
    {
        return gradient_;
    }

    real_t* data_ptr(int idx)
    {
        return (value_ + idx);
    }

    real_t* grad_ptr(int idx)
    {
        return (gradient_ + idx);
    }

    void setZeros()
    {
        memset(value_,0,sizeof(real_t) * size_);
    }
    void setRandn(real_t var = -1)
    {
        if(var < 0)
            var = sqrt(2.0 / size_);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<real_t>  dis(0,var);

        for(long idx = 0; idx < size_; idx++)
            *(value_ + idx) = dis(gen);
    }
};


class FC : public Op
{
public:
    FC(const std::vector<Node*>& inputs,const int& num_outputs)
        : Op(inputs)
    {
        this->set_channels(num_outputs);
    }
    void SetUp()
    {
        std::vector<Node*>& inputs = this->inputs();
        this->set_nums(inputs[0]->nums());
        this->set_rows(1);
        this->set_cols(1);

        const int& nums = this->nums();
        const int& channels = this->channels();
        this->Init(nums,1,1,channels);

        W_ = new Param(inputs[0]->channels() *
                inputs[0]->rows() *
                inputs[0]->cols() *
                channels);
        W_->setRandn();
        b_ = new Param(channels);
        b_->setZeros();
        std::vector<Param*>& learnable_params =
                this->learnable_params();
        learnable_params.push_back(W_);
        learnable_params.push_back(b_);
    }

    ~FC()
    {
        if(W_)
            delete W_;
        if(b_)
            delete b_;
    }

    void Forward()
    {
        const std::vector<Node*>& inputs = this->inputs();
        real_t* value = this->value();
        real_t* gradient = this->gradient();

        const int& nums = this->nums();
        const int& channels = this->channels();
        const int& size = this->size();

        int m = inputs[0]->nums();
        int n = channels;
        int k = inputs[0]->channels() *
                inputs[0]->rows() *
                inputs[0]->cols();

        //FC = x * W + b
        //as in the convolution layer,the data will always be in front of
        //weight matrix,since the first diemsnion will always be batchs size
        prod(inputs[0]->value(),W_->value(),value,m,n,k);

        for(int num = 0; num < nums; num++)
        {
            for(int channel = 0; channel < channels; channel++)
            {
                *(this->data_ptr(num,0,0,channel)) +=
                        *(b_->data_ptr(channel));
            }
        }
        memset(gradient,0,sizeof(real_t) * size);
    }

    //the value of bottom layer will not be used in the Backward step
    void Backward()
    {
        std::vector<Node*>& inputs = this->inputs();
        real_t* gradient = this->gradient();

        const int& nums = this->nums();
        const int& channels = this->channels();

        int m = nums;
        int n = inputs[0]->channels() *
                inputs[0]->rows() *
                inputs[0]->cols();
        int k = channels;

        //gradient have dimension m * k
        //weight matrix has dimension n * k
        //bottom gradient have dimension m * n
        prodTransPlus(gradient,W_->value(),inputs[0]->gradient(),m,n,k);


        m = inputs[0]->channels() *
                inputs[0]->rows() *
                inputs[0]->cols();
        k = nums;
        n = channels;
        //bottom data will have dimension k * m
        //gradient will have dimension k * n
        //weight graditn will have dimension m * n
        transProdPlus(inputs[0]->value(),gradient,W_->gradient(),m,n,k);

        //add up to bias gradient
        for(int num = 0; num < nums; num++)
        {
            for(int channel = 0; channel < channels; channel++)
            {
                *(b_->grad_ptr(channels)) +=
                        *(this->grad_ptr(num,0,0,channel));
            }
        }
    }
private:
    //weight
    Param* W_;
    //bias
    Param* b_;
};


//Conv Layer
class Conv : public Op
{
private:
    int window_;
    int stride_;
    int padding_;

    Param* W_;
    Param* b_;

    real_t* imcol_;
    real_t* gradcol_;

public:
    Conv(const std::vector<Node*>& inputs, int num_filters,
         int window = 5,int padding = 2,int stride = 1):
        Op(inputs)
    {
        this->set_channels(num_filters);
        window_ = window;
        stride_ = stride;
        padding_ = padding;
    }

    void SetUp()
    {
        std::vector<Node*>& inputs = this->inputs();
        this->set_nums(inputs[0]->nums());
        this->set_rows((inputs[0]->rows() + 2 * padding_ - window_)
                / stride_ + 1);
        this->set_cols((inputs[0]->cols() + 2 * padding_ - window_)
                / stride_ + 1);
        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();

        this->Init(nums,rows,cols,channels);
        W_ = new Param(window_ * window_ *
                       inputs[0]->channels() *
                channels);
        W_->setRandn();
        //bias will have different values on
        //diffrent channels of the output featuremap
        b_ = new Param(channels);
        b_->setZeros();

        std::vector<Param*>& learnable_params =
                this->learnable_params();
        learnable_params.push_back(W_);
        learnable_params.push_back(b_);

        //imcol will multiply W from the left
        //note the inputs[0]->channels()
        imcol_ = new
                real_t[nums * rows * cols
                * window_ * window_ *
                inputs[0]->channels()];

        gradcol_ = new
                real_t[nums * rows * cols
                * window_ * window_ *
                inputs[0]->channels()];
    }

    ~Conv()
    {
        if(W_)
            delete W_;
        if(b_)
            delete b_;
        if(imcol_)
            delete[] imcol_;
        if(gradcol_)
            delete[] gradcol_;
    }

    void Forward()
    {
        //bottom value are used to compute top value
        std::vector<Node*>& inputs = this->inputs();

        real_t* value = this->value();
        //gradient will be set to zero in each Forward step
        real_t* gradient = this->gradient();

        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();
        //since in each Forward step,we need set gradient all to zero
        const int& size = this->size();

        im2col(inputs[0]->value(),imcol_,
                nums,inputs[0]->channels(),inputs[0]->rows(),inputs[0]->cols(),
                window_,window_,padding_,stride_);

        //notice that batch_size will always in front of all dimensions
        int m = nums * rows * cols;
        //channel will always be the last dimension
        int n = channels;
        //rows of weight matrix
        int k = window_ * window_ *
                inputs[0]->channels();
        //value = col * W + b
        prod(imcol_,W_->value(),value,m,n,k);
        for(int num = 0; num < nums; num++)
            for(int row = 0; row < rows; row++)
                for(int col = 0; col < cols; col++)
                {
                    for(int channel = 0; channel < channels; channel++)
                    {
                        *(this->data_ptr(num,row,col,channel))
                                += *(b_->data_ptr(channel));
                    }
                }
        memset(gradient,0,sizeof(real_t) * size);
    }

    void Backward()
    {
        //top gradient will be computed by bottom gradient
        std::vector<Node*>& inputs = this->inputs();
        real_t* gradient = this->gradient();

        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();


        //imcol will have dimension k * m
        //grad will have dimension k * n
        //W_grad will have dimension m * n
        int m = window_ * window_ *
                inputs[0]->channels();
        int k = nums * rows * cols;
        int n = channels;
        //W_grad += imcol.T * grad
        transProdPlus(imcol_,gradient,W_->gradient(),m,n,k);

        //b_grad
        for(int num = 0; num < nums; num++)
            for(int row = 0; row < rows; row++)
                for(int col = 0; col < cols; col++)
                    for(int channel = 0; channel < channels; channel++)
                    {
                        //just exchange the oprands on the left and right
                        //hand side of += in the Forward step
                        *(b_->grad_ptr(channel)) +=
                                *(this->grad_ptr(num,row,col,channel));
                    }

        //compute gradcol
        //gradient will have dimension m*k
        //W will have dimension n*k
        //gradcol will have dimension m*n
        m = nums * cols * rows;
        k = channels;
        n = window_ * window_ * inputs[0]->channels();

        prodTrans(gradient,W_->value(),gradcol_,m,n,k);
        real_t* imgradient = new
                real_t[nums * inputs[0]->rows()
                * inputs[0]->cols() *
                inputs[0]->channels()];
        col2im(gradcol_,imgradient,
               nums,inputs[0]->channels(),inputs[0]->rows(),inputs[0]->cols(),
                window_,window_,padding_,stride_);
        //add imgradient to gradient
        cblas_axpy(inputs[0]->size(),1.0,imgradient,1,inputs[0]->gradient(),1);

        delete[] imgradient;
    }
};

//Relu Layer
class Relu : public Op
{
private:
    real_t leak_;
public:
    Relu(const std::vector<Node*>& inputs,
         real_t leak) : Op(inputs)
    {
        this->set_channels(inputs[0]->channels());
        leak_ = leak;
    }

    void SetUp()
    {
        const std::vector<Node*>& inputs = this->inputs();
        this->set_nums(inputs[0]->nums());
        this->set_rows(inputs[0]->rows());
        this->set_cols(inputs[0]->cols());

        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();
        this->Init(nums,rows,cols,channels);
    }

    void Forward()
    {
        const std::vector<Node*>& inputs = this->inputs();
        const int& size = this->size();
        for(int idx = 0; idx < size; idx++)
        {
            if(*(inputs[0]->data_ptr(idx)) < 0)
                *(this->data_ptr(idx)) = 0;
            else
                *(this->data_ptr(idx)) =
                    leak_ * *(inputs[0]->data_ptr(idx));
        }

        real_t* gradient = this->gradient();
        memset(gradient,0,sizeof(real_t) * size);
    }

    void Backward()
    {
        const std::vector<Node*>& inputs = this->inputs();
        const int& size = this->size();
        for(int idx = 0; idx < size; idx++)
        {
            if(*(inputs[0]->data_ptr(idx)) < 0)
                *(inputs[0]->grad_ptr(idx)) += 0;
            else
                *(inputs[0]->grad_ptr(idx)) +=
                    leak_ * *(this->grad_ptr(idx));
        }
    }
};

//Label Layer
class Label : public Node
{
public:
    //label will always have channel 1
    Label() {}
    Label(const int& nums,const int& rows,const int& cols ) :
        Node(nums,rows,cols,1) {}
    void SetUp()
    {
        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();

        this->Init(nums,rows,cols,channels);
    }
    void Forward() {}
    void Backward() {}
};
class Loss : public Op
{
public:
    //in the case of fully connected,the rows and cols will both be 1
    Loss(std::vector<Node*>& inputs) :
        Op(inputs)
    {
        this->set_channels(inputs[0]->channels());
    }

    void SetUp()
    {
        std::vector<Node*>& inputs = this->inputs();
        this->set_nums(inputs[0]->nums());
        this->set_rows(inputs[0]->rows());
        this->set_cols(inputs[0]->cols());
    }

    ~Loss()
    {
    }

    real_t& result()
    {
        return result_;
    }

private:
    real_t result_;
};

class Softmax : public Loss
{
public:
    //channels of softmax layer is simply number of classes
    Softmax(std::vector<Node*>& inputs) : Loss(inputs)
    {

    }

    void SetUp()
    {
        std::vector<Node*>& inputs = this->inputs();
        this->set_nums(inputs[0]->nums());
        this->set_rows(inputs[0]->rows());
        this->set_cols(inputs[0]->cols());

        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();
        const int& channels = this->channels();

        this->Init(nums,rows,cols,channels);
    }

    void Forward()
    {
        std::vector<Node*>& inputs = this->inputs();
        const int& channels = this->channels();
        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();

        const int& size = this->size();

        //resluts of this layer is a simply softmax operation of last layer
        real_t* value = this->value();
        softmax(inputs[0]->value(),value,nums,rows,cols,channels);

        real_t* logz = new real_t[nums * rows * cols];
        memset(logz,0,sizeof(real_t) * nums * rows * cols);
        logsumexp(inputs[0]->value(),logz,nums,rows,cols,channels);
        real_t* tmp_val = new real_t[nums * rows * cols];
        memset(tmp_val,0,sizeof(real_t) * nums * rows * cols);
        for(int num = 0; num < nums; num++)
        {
            int label = static_cast<int>(
                        *(inputs[1]->data_ptr(num,0,0,0)));
            *(tmp_val + num) =
                    *(logz + num) -
                    *(inputs[0]->data_ptr(num,0,0,label));
        }
        delete[] logz;
        real_t& result = this->result();
        result = 0;
        for(int idx = 0; idx < size; idx++)
        {
            result += *(tmp_val + idx) /size;
        }
    //    std::cout << '\n';

        delete[] tmp_val;
        real_t* gradient = this->gradient();
        memset(gradient,1,sizeof(real_t) * size);
    }

    void Backward()
    {
        real_t* value = this->value();

        std::vector<Node*>& inputs = this->inputs();
        const int& channels = this->channels();
        const int& nums = this->nums();
        const int& rows = this->rows();
        const int& cols = this->cols();

        const int& size = this->size();

        real_t* onehot_vector = new real_t[size];
        memset(onehot_vector,0,sizeof(real_t) * size);
        //compute the one hot vector,which has the same shape as the input fully connected layer
        onehot(inputs[1]->value(),onehot_vector,nums,rows,cols,channels);

        cblas_axpy(size,-1.,onehot_vector,1,value,1);

        //add up loss pixel by pixel
        for(int num = 0; num < nums; num++)
            for(int row = 0; row < rows; row++)
                for(int col = 0; col < cols; col++)
                    for(int channel = 0; channel < channels; channel++)
                    {
                        *(inputs[0]->grad_ptr(num,row,col,channel)) +=
                                *(this->data_ptr(num,row,col,channel)) / size;
                    }
        delete[] onehot_vector;
    }
};

#endif
