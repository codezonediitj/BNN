#include <bnn/core/tensor.hpp>
#include <bnn/io/input.hpp>
#include <gtest/gtest.h>
#include <iostream>

using namespace bnn::core;
using namespace bnn::io;

TEST(IO, LoadMNIST)
{
    TensorCPU<unsigned>* imgs = load_mnist_images<unsigned>("t10k-images-idx3-ubyte");
    unsigned idx1[10] = {329, 519, 241, 684, 546, 741, 410, 300, 301, 656};
    unsigned pixels[10] = {18, 251, 198, 254, 221, 18, 254, 254, 106, 254};
    for(unsigned i = 0; i < 10; i++)
    {
        EXPECT_EQ(imgs->get_data_pointer()[idx1[i]], pixels[i])
        <<"Incorrect pixel value for first MNIST test image.";
    }

    TensorCPU<unsigned>* labels = load_mnist_labels<unsigned>("t10k-labels-idx1-ubyte");
    unsigned idx2[10] = {548, 711, 258, 630, 655, 327, 258, 232, 740, 630};
    unsigned _labels[10] = {3, 5, 2, 9, 8, 0, 2, 8, 4, 9};
    for(unsigned i = 0; i < 10; i++)
    {
        EXPECT_EQ(labels->at(idx2[i]), _labels[i])
        <<"Incorrect value for first MNIST test labels.";
    }
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
