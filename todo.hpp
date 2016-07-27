class Pool : public Op
{
public:
	int* mask_;
	int window_;
	int stride_;
	Pool(Node* input,int window = 2,int stride = 2)		
	{
		inputs_.push_back(input);
		window_ = window;
		stride_ = stride;

		nums_ = input->nums();
		rows_ = (input->rows() - window_) / stride_ + 1;
		cols_ = (input->cols() - window_) / stride_ + 1;
		channels_ = input->channel;

		init();

		mask_ = new bool[size_];
		memset(mask_,
			0,
			sizeof(bool) * size_);
	}

	~Pool()
	{
		delete[] value_;
		delete[] gradient_;
		delete[] mask_;
	}

	void forward()
	{
	
		for(int num = 0; num < nums; num++)
			for(int row = 0;row < rows - window_; row+= stride_)
				for(int col = 0;col < cols - window_;col+=stride_)
				{
					int max_row_idx = 0;
					int max_col_idx = 0;
					Real max_data = 0;
					for(int channel = 0;channel < channels;channel++)
					{
						for(int pool_row = row - window_; win_row < row + window_; win_row++)
							for(int win_col = col - window_; win_col < col + window_; win_col++)
							{
								Real data = *(inputs[0]->value_ + num * rows * cols * channels +
									row * cols * channels+
									col * channels +
									channel);
								if(data > max_data)
								{
									max_data = data;
									max_row_idx = win_row;
									max_col_idx = win_col;
								}

							}
						
					}
					int pool_row = (row - window_) / stride_;
					int pool_col = (col - window_) / stride_;
					*(value + num * rows * cols * channels +
						max_row * cols * channels +
						max_col * channels +
						channel) = max_data;
					*(mask + num * rows * cols * channels +
					max_row * cols * channels +
					max_col * channels +
					channel)  = 1;
				}
	}

	void backward()
	{
		for(int num = 0; num < nums; num++)
			for(int row = window_;row < rows - window_; row+=2 * window_)
				for(int col = window_;col < cols - window_;col+=2 * window_)
				{
					int max_row_idx = 0;
					int max_col_idx = 0;
					for(int channel = 0;channel < channels;channel++)
					{
						for(int pool_row = row - window_; win_row < row + window_; win_row++)
							for(int win_col = col - window_; win_col < col + window_; win_col++)
							{
								bool mask = *(mask_->value_ + num * rows_ * cols_ * channels_ +
									row * cols * channels +
									col * channels +
									channel);
								if(mask)
								{
									max_col_idx = win_col;
									max_row_idx = win_row;
								}

							}
						
					}
					int pool_row = (row - window_) / stride_;
					int pool_col = (col - window_) / stride_;
					*(inputs[0]->gradient() + num * rows * cols * channels +
						max_row * cols * channels +
						max_col * channels +
						channel) = *(gradient + num * rows * cols * channels +
					max_row * cols * channels +
					max_col * channels +
					channel);
				}
	}
};


class ScalarMul : public Op
{
public:
	Real scalar_;

public:
	ScalarMul(Node* input,Real scalar = 1)
	{
		inputs_.push_back(input);
		scalar_ = scalar;

		nums_ = input->nums();
		rows_ = input->rows();
		cols_ = input->cols();
		channels_ = input -> channels_;
		init();
	}

	~ScalarMul()
	{
		delete[] value_;
		delete[] gradient_;
	}


	void forward()
	{
		memset(value,0,size * sizeof(Real));
		cblas_daxpy(size,
			scalar,
			inputs[0]->value(),
			1,
			value,
			1
			);
		memset(gradient,0,size * sizeof(Real));
	}

	void backward()
	{
		cblas_axpy(size,
			scalar_,
			gradient,
			1,
			inputs[0]->gradient(),
			1
			);
	}
};

class ScalarPow : public Op
{
public:
	Real scalar_;
	Real temp_;

	ScalarPow(Node* input,Real scalar)
	{
		inputs_.push_back(input);
		scalar_ = scalar;
		nums_ = input->nums();
		rows_ = input->rows();
		cols_ = input->cols();
		channels_ = input -> channels_;
		init();
	}

	~ScalarPow()
	{
		delete[] value_;
		delete[] gradient_;
	}

	void forward()
	{
		temp_ = pow(inputs[0]->value(),
			scalar_ - 1);
		value_ = temp_ * inputs[0]->value();
		memset(gradient,0,sizeof(Real) * size);
	}

	void backward()
	{
		cblas_daxpy(size,
			scalar_ * temp_,
			gradient,
			1,
			inputs[0]->gradient(),
			1);
	}
};

class Relu : public Op
{
	Real leak_;
	Relu(Node* input,Real leak_)
	{
		inputs_.push_back(input);
		leak_ = leak;
		nums_ = input->nums();
		rows_ = input->rows();
		cols_ = input->cols();
		channels_ = input -> channels_;
		init();
	}

	void forward()
	{
		for(int id = 0; id < size; id++)
		{
			if(*(inputs[0]->value_ + id) < 0)
				*(value + id) = 0;
			else
				*(value + id) = 
					leak_ * (inputs[0]->value() + id);
		}

		memset(gradient,0,sizeof(Real) * size);
	}

	void backward()
	{
		for(int id = 0; id < size; id++)
		{
			if(*(inputs[0]->value() + id) < 0)
				*(inputs[0]->gradient() + id) += 0;
			else
				*(inputs[0]->gradient() + id) += 
					leak_ * *(gradient + id);
		}
	}
};

class DropOut : public Op
{
public:
	Real prob_;
	bool* mask_;
	DropOut(Node* input,
		const Real prob = 0.5)
	: Op(input->nums(),
		input->rows(),
		input->cols(),
		input->channels())
	{
		prob_ = prob;
		mask_ = new bool[size_];
	}

	DropOut()
	{
		delete[] mask_;
	}
	void forward(bool disabled = false)
	{
		if(disabled)
			memcpy(value_,
				inputs[0]->value(),
				size);
		else
		{
			std::random_device rd;
			std::mt19973 gen(rd());
			std::uniform_distribution<Real>  dis(0,1);
			for(int  id = 0; id < size_; id++)
			{
				Real rand = dis(gen);
				if(rand < prob_)
					*(mask + id) = true;
				else
					*(mask + id) = false;
			}
		}
		memset(gradient_,0,sizeof(Real) * size_);
	}
	void backward(bool disabled = false)
	{
		if(disabled)
			cblas_axpy(size_,
				1.0,
				gradient_,
				1,
				inputs_[0]->gradient(),
				1);
		else
		{
			for(int id = 0; id < size_; id++)
			{
				if(*(mask_ + id) == true)
					*(inputs_[0]->gradient() + id) = 
					*(gradient_ + id) / prob_;
			}
		}
	}
};


template <typename Dtype>
class Batch
{
private:
	int nums_;
	int channels_;
	int rows_;
	int cols_;

	Dtype* data_;
public:
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
		return channels;
	}

//setters of dimensions
	void setNums(const int& nums)
	{
		nums_ = nums;
	}

	void setChannels(const int& channels)
	{
		channels_ = channels;
	}

	void setRows(const int& rows)
	{
		rows_ = rows;
	}

	void setCols(const int& cols)
	{
		cols_ = cols; 
	}

//get the address of certain offset
	Dtype* ptr(int num,
		int row,
		int col,
		int channel)
	{
		return (data + num * rows_ * cols_ * channels_+
			row * cols_ * channels_ +
			col * channels_ +
			channel);
	}
};
