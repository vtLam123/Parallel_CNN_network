#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/CPU_conv.h"
#include "src/layer/GPU_conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "Network.h"

void trainModel(int n_epoch, int batch_size)
// int main()
{
    // data
    std::cout << "Loading fashion-mnist data...\n";
    MNIST dataset("./data/mnist/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    std::cout << "Loading model...\n";

    float accuracy = 0.0;

    //  // 2. Host - CPU Network
    // std::cout << "Test: Host - CPU Network" << std::endl;
    Network dnn = createNetwork_CPU();

    dnn.load_parameters("./model/weights_cpu.bin");
    dnn.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
}

int main(int argc, char *argv[])
{

    trainModel(5, 128);

    return 0;
}