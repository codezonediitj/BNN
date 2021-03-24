#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/core/tensor_ops.hpp>

using namespace std;
using namespace bnn::core;
using namespace bnn::utils;

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

    BNNMemory->free_memory(t);
    BNNMemory->free_memory(s0);
    BNNMemory->free_memory(s1);
    BNNMemory->free_memory(s2);
    BNNMemory->free_memory(s3);
    BNNMemory->free_memory(ts0);
    BNNMemory->free_memory(ts1);
    BNNMemory->free_memory(ts2);
    BNNMemory->free_memory(ts3);

}

TEST(Core, TensorOpsDivide)
{
    vector<unsigned> shape = {2, 9, 16, 100};
    TensorCPU<unsigned>* t = new TensorCPU<unsigned>(shape);
    bnn::core::fill(t, (unsigned)6);
    TensorCPU<unsigned>* z = divide(t, (unsigned)3);
    EXPECT_EQ(z->at(0, 4, 8, 50), 2)<<"Expected quotient is 2.";
}

TEST(Core, TensorOpsOneHot)
{
    vector<unsigned> shape = {1000};
    TensorCPU<unsigned>* labels = new TensorCPU<unsigned>(shape);
    bnn::core::fill(labels, (unsigned)9);
    for(unsigned i = 0; i < 9; i++)
    {
        labels->set(i, i);
    }
    TensorCPU<unsigned>* new_labels = one_hot(labels, (unsigned)1, (unsigned)0, (unsigned)10);
    for(unsigned i = 0; i < 10; i++)
    {
        for(unsigned j = 0; j < 10; j++)
        {
            EXPECT_EQ(new_labels->at(i, j), labels->at(i) == j)
            <<"Expected one hot value is "<<(labels->at(i) == j)
            <<" for "<<i<<", "<<j;
        }
    }
}

TEST(Core, TensorOpsMatmul)
{
    TensorCPU<float> *M, *N, *V;
    vector<unsigned> shapeM = {200, 784}, shapeN = {784, 3000}, shapeV = {784, 1};
    M = new TensorCPU<float>(shapeM);
    N = new TensorCPU<float>(shapeN);
    V = new TensorCPU<float>(shapeV);
    bnn::core::fill(M, (float)2.);
    bnn::core::fill(N, (float)3.);
    bnn::core::fill(V, (float)3.);
    string msg = "The resultant matrix should be filled with 4704.";
    TensorCPU<float>* MN = matmul(M, N);
    EXPECT_EQ(MN->at(0, 0), 4704.)<<msg;
    EXPECT_EQ(MN->at(199, 2999), 4704.)<<msg;
    EXPECT_EQ(MN->at(100, 1500), 4704.)<<msg;
    TensorCPU<float>* MV = matmul(M, V);
    EXPECT_EQ(MN->at(0, 0), 4704.)<<msg;
    EXPECT_EQ(MN->at(199, 0), 4704.)<<msg;
    EXPECT_EQ(MN->at(100, 0), 4704.)<<msg;
}

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

    shape = {100, 28, 28};
    TensorCPU<unsigned>* p = new TensorCPU<unsigned>(shape);
    TensorCPU<unsigned>* q = new TensorCPU<unsigned>(shape);
    vector<unsigned> new_shape = {100, 784};
    unsigned _new_shape[] = {100, 784};
    p->reshape(new_shape);
    q->reshape(_new_shape, (unsigned)2);
    EXPECT_EQ((p->get_shape()[0] == 100) && (p->get_shape()[1], 784), true)
    <<"Expected shape is {100, 784}";
    EXPECT_EQ((q->get_shape()[0] == 100) && (q->get_shape()[1], 784), true)
    <<"Expected shape is {100, 784}";
    EXPECT_EQ((p->get_ndims() == 2) && (q->get_ndims() == 2), true)
    <<"Expected number of dimensions is 2";

    new_shape = {10, 78};
    _new_shape[0] = 10, _new_shape[1] = 78;

    EXPECT_THROW({
        try
        {
            q->reshape(_new_shape, 2);
        }
        catch(const std::runtime_error& e)
        {
            EXPECT_STREQ("The new shape consumes different amount of memory.", e.what());
            throw;
        }
    }, std::runtime_error);
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
