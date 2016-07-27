#include "cnn.hpp"

int main()
{
    const std::string& record_file =
            "/home/zak/Source/pydeeplearn/imglist.txt";

    std::vector<Param*> learnable_params;

    Data* data = new Data(NULL);
    std::vector<Node*> inputs1;
    inputs1.push_back(data);
    Conv* conv1 = new Conv(inputs1,20);
    for(int i = 0; i < conv1->learnable_params().size(); i++)
    {
        learnable_params.push_back(conv1->learnable_params()[i]);
    }
    std::vector<Node*> inputs2;
    inputs2.push_back(conv1);
    Relu* relu1 = new Relu(inputs2,1);
    std::vector<Node*> inputs3;
    inputs3.push_back(relu1);
    FC* fc1 = new FC(inputs3,10);
    for(int i = 0; i < fc1->learnable_params().size(); i++)
    {
        learnable_params.push_back(fc1->learnable_params()[i]);
    }

    Label* label = new Label();
    std::vector<Node*> inputs4;
    inputs4.push_back(fc1);
    inputs4.push_back(label);
    Softmax* loss = new Softmax(inputs4);

    RMSprop* updator = new RMSprop();
    InverseDecay* step = new InverseDecay();
    CNN* cnn = new CNN(loss,updator,step);
    updator->set_learnable_params(learnable_params);
    cnn->setData(data);
    cnn->setLabel(label);
    cnn->setInput(40,record_file);
    cnn->train(100);
    delete data;
    delete conv1;
    delete relu1;
    delete fc1;
    delete loss;
    delete label;
    delete updator;
    delete step;
    delete cnn;
    return 0;
}
