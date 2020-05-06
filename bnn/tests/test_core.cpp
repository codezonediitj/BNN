#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/core/tensor_ops.hpp>

using namespace std;
using namespace bnn::core;
using namespace bnn::utils;

TEST(Core, TensorCPU)
{
    TensorCPU<float>* t_f = new TensorCPU<float>;
    EXPECT_EQ(0, t_f->get_ndims())<<"Default size should be 0";
    vector<unsigned> shape = {3, 3, 3};
    TensorCPU<float>* t = new TensorCPU<float>(shape);
    string msg1 = "The size of tensor must be 3";
    EXPECT_EQ(3, t->get_ndims())<<msg1;
    EXPECT_EQ(true,
    t->get_shape()[0] == 3 && t->get_shape()[1] == 3 && t->get_shape()[2] == 3)
    <<"The shape of tensor should be (3, 3, 3)";
    for(unsigned i = 0; i < 3; i++)
    {
        for(unsigned j = 0; j < 3; j++)
        {
            for(unsigned k = 0; k < 3; k++)
            {
                t->set(5., i, j, k);
                EXPECT_EQ(5., t->at(i, j, k))
                <<"Each element of the tensor should be 5.";
            }
        }
    }
    t->set(3., 1, 2, 2);
    EXPECT_EQ(3., t->at(1, 2, 2));
}

TEST(Core, TensorOpsSum)
{

    vector<unsigned> shape = {2, 9, 16, 100};
    TensorCPU<unsigned>* t = new TensorCPU<unsigned>(shape);
    bnn::core::fill(t, (unsigned)3);
    TensorCPU<unsigned> *s0, *s1, *s2, *s3, *ts0, *ts1, *ts2, *ts3;
    s0 = sum(t, 0), s1 = sum(t, 1), s2 = sum(t, 2), s3 = sum(t, 3);
    ts0 = sum(s0), ts1 = sum(s1), ts2 = sum(s2), ts3 = sum(s3);
    EXPECT_EQ(s0->at(0, 0, 0), 6)<<"Expected sum along axis 0 is 6";
    EXPECT_EQ(s1->at(1, 4, 50), 27)<<"Expected sum along axis 1 is 27";
    EXPECT_EQ(s2->at(0, 4, 99), 48)<<"Expected sum along axis 2 is 48";
    EXPECT_EQ(s3->at(1, 7, 8), 300)<<"Expected sum along axis 3 is 300";
    EXPECT_EQ(ts0->at(0), 86400)<<"Expected sum is 86400";
    EXPECT_EQ(ts1->at(0), 86400)<<"Expected sum is 86400";
    EXPECT_EQ(ts2->at(0), 86400)<<"Expected sum is 86400";
    EXPECT_EQ(ts3->at(0), 86400)<<"Expected sum is 86400";

    delete BNNMemory;
    delete BNNThreads;
}

int main(int ac, char* av[])
{
    if(ac == 2 && strcmp(av[1], "--CI=ON") == 0)
    {
        testing::GTEST_FLAG(filter) = "Core.TensorCPU";
    }
    testing::InitGoogleTest(&ac, av);
    return RUN_ALL_TESTS();
}
