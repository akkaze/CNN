#ifndef NET_HPP
#define NET_HPP
#include "cnn.hpp"

class Net
{
public:
    Net() {}
    Net(Loss* objective,Update* updator,Step* step)
    {
        objective_ = objective;
        setUp(updator,step);
        processObjective();
    }
	
    void setUp(Update* updator,Step* step)
    {
        updator_ = updator;
        step_ = step;
        epoch_ = 0;
        iter_ = 0;
    }

    void processObjective()
    {
        topsort();
        for(int i = 0; i< sorted_nodes_.size(); i++)
        {
            if(sorted_nodes_[i]->has_learnable_params())
            {
                std::vector<Param*> learnable_params_this_node =
                    sorted_nodes_[i]->learnable_params();
                for(int param_id = 0; param_id < learnable_params_this_node.size(); param_id++)
                {
                    learnable_params_.push_back(learnable_params_this_node[param_id]);
                }
            }
        }
    }

    //topological sort the computational graph
    void topsort()
    {
        std::unordered_set<Node*> visited;
        topsortImpl(objective_,visited);
    }
	
    void topsortImpl(Node* node,std::unordered_set<Node*>& visited)
    {
        visited.insert(node);
        for(int i = 0; i < node->inputs().size(); i++)
        {
            if(visited.find(node->inputs()[i]) == visited.end())
            {
                topsortImpl(node->inputs()[i],visited);
            }
        }
        sorted_nodes_.push_back(node);
    }

    void setUp()
    {
        for(int i = 0; i< sorted_nodes_.size(); i++)
            sorted_nodes_[i]->SetUp();
    }

    //from the first node to the last node in computational graph
    //execute forward operation one by one
    void forward()
    {
        for(int i = 0; i < sorted_nodes_.size(); i++)
            sorted_nodes_[i]->Forward();
    }
	
    //from the last node to the first node in computational graph
    //execute backward operation one by one
    void backward()
    {
       //reset the loss layer's loss
        real_t* gradient = objective_->gradient();
        const int& size = objective_->size();
        memset(gradient,1,sizeof(real_t) * size);
        //back propgation
        for(int i = sorted_nodes_.size() - 1; i >= 0; i--)
            sorted_nodes_[i]->Backward();
    }

    void print()
    {
        std::cout << "epoch:" << epoch_ + 1 << ',' <<
                     "iter:" << iter_ + 1 << ',' <<
                     "cost:" << result() << std::endl;
    }

    void incr_epoch()
    {
        epoch_ += 1;
    }

    void incr_iter()
    {
        iter_ += 1;
    }

    real_t& result()
    {
        return objective_->result();
    }
	
    //getters
    const int& get_iter()
    {
        return iter_;
    }

    Update* updator()
    {
        return updator_;
    }

    Step* step()
    {
        return step_;
    }

    Data* data()
    {
        return data_;
    }

    Label* label()
    {
        return label_;
    }

    std::vector<Param*>& learnable_params()
    {
        return learnable_params_;
    }

    //setters
    void setData(Data* data)
    {
        data_ = data;
    }

    void setLabel(Label* label)
    {
        label_ = label;
    }

private:
    Data* data_;
    Label* label_;
    Loss* objective_;
    Update* updator_;
    Step* step_;
    int epoch_;
    int iter_;

    std::vector<Node*> sorted_nodes_;
    std::vector<Param*> learnable_params_;
};

class CNN : public Net
{
public:
    CNN() {}
    CNN(Loss* objective,Update* updator,Step* step)
        :Net(objective,updator,step) {}

    void setInput(const int& batch_size,const std::string& record_file)
    {
        batch_size_ = batch_size;
        readDataFromFiles(record_file);
        setUp();
        get_random_indices();
    }

    ~CNN()
    {
        if(data_value_)
            delete[] data_value_;
        if(labels_value_)
            delete[] labels_value_;
    }
    void get_random_indices()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> tmp_idxs(num_data_);
        for(int i = 0; i < num_data_; i++)
            tmp_idxs[i] = i;
        std::shuffle(tmp_idxs.begin(),tmp_idxs.end(),gen);
        int num_batches = num_data_ / batch_size_;
        idxs.resize(num_batches);
        for(int batch_idx = 0; batch_idx < num_batches; batch_idx++)
            idxs[batch_idx].resize(batch_size_);
        for(int batch_idx = 0; batch_idx < num_batches; batch_idx++)
            for(int in_batch_idx = 0; in_batch_idx < batch_size_; in_batch_idx++)
            idxs[batch_idx][in_batch_idx] = tmp_idxs[batch_idx * batch_size_ + in_batch_idx];
        tmp_idxs.clear();
    }
    void train(const int& epochs)
    {
        this->setUp();
        for(int e = 0; e < epochs; e++)
        {
            for(int i = 0;i < idxs.size(); i ++)
            {
                //set data of this batch
                real_t* batch_data = new real_t[batch_size_ * rows_ * cols_ * channels_];
                memcpy(batch_data,data_value_ + i * batch_size_,
                    sizeof(real_t) * batch_size_ * rows_
                    * cols_ * channels_);
                Data* data = this->data();
                data->set_value(batch_data);

                //set labels of this batch
                real_t* batch_labels = new real_t[batch_size_];
                memcpy(batch_labels,labels_value_ + i * batch_size_, sizeof(real_t) * batch_size_);
                Label* label = this->label();
                label->set_value(batch_labels);

                //execute the forward and backward routine
                this->forward();
                this->backward();
				
                Step* step = this->step();
                Update* updator = this->updator();

                const int& iter = this->get_iter();
                //get the current learning rate
                real_t alpha = step->get(iter);
                //update the parameters
                updator->apply(alpha);
                this->print();
                delete[] batch_data;
                delete[] batch_labels;

                this->incr_iter();
            }
            this->incr_epoch();
        }
    }

    void readDataFromFiles(const std::string& record_file)
    {
        //line of record file must be in the format (filename label)
        std::ifstream in(record_file);
        std::unordered_map<std::string,int> img_label_map;
        if(in.is_open())
        {
            while(!in.eof())
            {
                std::string img_path;
                int label;
                in >> img_path >> label;
                if(!img_path.empty())
                {
                    std::pair<std::string,int> img_label =
                        std::make_pair(img_path,label);
                    img_label_map.insert(img_label);
                }
            }
        }
        in.close();

        //read information from the first image
        {
            std::string img_path = img_label_map.begin()->first;
            cv::Mat img = cv::imread(img_path,cv::IMREAD_UNCHANGED);
            rows_ = img.rows;
            cols_ = img.cols;

            channels_ = img.channels();
            label_map_rows_ = 1;
            label_map_cols_ = 1;
        }
        //get the number of images and reserve space for indices
        num_data_ = img_label_map.size();
		
        //allocate space for labels and image data
        data_value_ = new real_t[num_data_ * rows_ * cols_ * channels_];
        labels_value_ = new real_t[num_data_ * label_map_rows_ * label_map_cols_];
        //read image data from file names
        int idx  = 0;
		
        for(std::unordered_map<std::string,int>::const_iterator itr =
            img_label_map.begin(); itr != img_label_map.end(); itr++)
        {
            std::string img_path = itr->first;
            int label = itr->second;
            cv::Mat img = cv::imread(img_path,cv::IMREAD_UNCHANGED);
            //assertions
            assert(rows_ = img.rows);
            assert(cols_ = img.cols);
            assert(channels_ = img.channels());

            //copy data to cnn's buffer
            uchar* orig_data = img.data;
			
            for(int r = 0; r < rows_; r++)
                for(int c = 0; c < cols_; c++)
                    for(int ch = 0; ch < channels_; ch++)
                    {
                        *(data_value_ + idx * rows_ * cols_ * channels_ +
                            r * cols_ * channels_ +
                            c * channels_ +
                            ch) =
                        *(orig_data + r * cols_ * channels_ +
                            c * channels_ +
                            ch);
                    }
            for(int r = 0; r < label_map_rows_; r++)
                for(int c = 0; c < label_map_cols_; c++)
                {
                    *(labels_value_ + idx * label_map_rows_ * label_map_cols_ +
                            r * label_map_cols_ +
                            c) =
                            label;
                }
            idx++;
        }

        //set values of data and label node
        Data* data = this->data();
        Label* label = this->label();
        data->set_dimensions(batch_size_,rows_,cols_,channels_);
        label->set_dimensions(batch_size_,label_map_rows_,label_map_cols_,1);
    }
private:
    real_t* data_value_;
    real_t* labels_value_;
    int num_data_;
    int rows_;
    int cols_;
    int label_map_rows_;
    int label_map_cols_;
    int channels_;
    int batch_size_;
    std::vector<std::vector<int>> idxs;
};
#endif
