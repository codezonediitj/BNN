#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operations.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/autodiff/graph.hpp>
#include <bnn/autodiff/forward.hpp>
#include <bnn/autodiff/reverse.hpp>
#include <bnn/core/tensor_ops.hpp>

using namespace bnn::core;
using namespace bnn::operators;
using namespace bnn::autodiff;

TEST(Autodiff, BuildGraph)
{
    vector<unsigned> shape = {3, 3, 3};
    TensorCPU<float> x1(shape), x2(shape), x3(shape);
    Operator<float>* expr = bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(&x1, &x2)),
        bnn::operations::exp(&x2)
    );

    GraphNode<float>* graph = build_graph(expr);
    vector<string>* _graph = new vector<string>[4];
    _graph[0] = {"Add_1"};
    _graph[1] = {"Exp_1", "Exp_0"};
    _graph[2] = {"Add_0", "TensorWrapper_0"};
    _graph[3] = {"TensorWrapper_1", "TensorWrapper_2"};
    GraphNode<float>* layer = graph;
    unsigned j = 3;
    while(layer != NULL)
    {
        for(unsigned i = 0; i < layer->len_ops; i++)
        {
            string op = layer->ops[i]->get_name();
            EXPECT_EQ(op, _graph[j][i])
            <<"Expected "<<_graph[j][i]<<" at layer "<<j<<" got, "<<op;
        }
        layer = layer->prev;
        j--;
    }
}

TEST(Autodiff, ComputeValue)
{
    TensorCPU<float> *x1, *x2, *x3;
    unsigned ndims = 3;
    unsigned* shape = new unsigned[ndims];
    shape[0] = 1, shape[1] = 1000, shape[2] = 1000;
    x1 = new TensorCPU<float>(shape, ndims);
    x2 = new TensorCPU<float>(shape, ndims);
    x3 = new TensorCPU<float>(shape, ndims);
    bnn::core::fill<float>(x1, 3.);
    bnn::core::fill<float>(x2, 2.);
    bnn::core::fill<float>(x3, 1.);
    Operator<float>* expr =
    bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(x1, x2)),
        bnn::operations::exp(bnn::operations::add(x2, x3))
    );
    TensorCPU<float>* val =  compute_value(expr);
    float expecval = 168.49870300292969;
    EXPECT_NEAR(expecval, val->at(0, 0, 0), 1.e-6)<<
    "Expected value of the expression is "<<val;

    BNNMemory->free_memory(x1);
    BNNMemory->free_memory(x2);
    BNNMemory->free_memory(x3);
    BNNMemory->free_memory(expr->get_value());
}

TEST(Autodiff, ComputeGradientForward)
{
    TensorCPU<float> *x1, *x2, *x3;
    unsigned ndims = 3;
    unsigned* shape = new unsigned[ndims];
    shape[0] = 1, shape[1] = 1000, shape[2] = 100;
    x1 = new TensorCPU<float>(shape, ndims);
    x2 = new TensorCPU<float>(shape, ndims);
    x3 = new TensorCPU<float>(shape, ndims);
    bnn::core::fill<float>(x1, 3.);
    bnn::core::fill<float>(x2, 2.);
    bnn::core::fill<float>(x3, 1.);
    Operator<float>* expr =
    bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(x1, x2)),
        bnn::operations::exp(bnn::operations::add(x2, x3))
    );
    TensorCPU<float>** vars = new TensorCPU<float>*[2];
    vars[0] = x1, vars[1] = x2;
    TensorCPU<float>** grads =  compute_gradient_forward(expr, vars, 2);
    float gradvals[] = {148.41316223144531, 168.49870300292969};
    EXPECT_NEAR(gradvals[0], grads[0]->at(0, 0, 0), 1.e-6)<<
    "Expected value of graident with respect to x2 is "<<gradvals[0];
    EXPECT_NEAR(gradvals[1], grads[1]->at(0, 0, 0), 1.e-6)<<
    "Expected value of graident with respect to x2 is "<<gradvals[1];

    BNNMemory->free_memory(x1);
    BNNMemory->free_memory(x2);
    BNNMemory->free_memory(x3);
    BNNMemory->free_memory(grads[0]);
    BNNMemory->free_memory(grads[1]);
}

TEST(Autodiff, ComputeGradientReverse)
{
    TensorCPU<float> *x1, *x2, *x3;
    unsigned ndims = 3;
    unsigned* shape = new unsigned[ndims];
    shape[0] = 1, shape[1] = 1000, shape[2] = 100;
    x1 = new TensorCPU<float>(shape, ndims);
    x2 = new TensorCPU<float>(shape, ndims);
    x3 = new TensorCPU<float>(shape, ndims);
    bnn::core::fill<float>(x1, 3.);
    bnn::core::fill<float>(x2, 2.);
    bnn::core::fill<float>(x3, 1.);
    Operator<float>* expr =
    bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(x1, x2)),
        bnn::operations::exp(bnn::operations::add(x2, x3))
    );
    TensorCPU<float>** vars = new TensorCPU<float>*[2];
    vars[0] = x1, vars[1] = x3;
    TensorCPU<float>** grads =  compute_gradient_reverse(expr, vars, 2);
    float gradvals[] = {148.41316223144531, 20.085536956787109};
    EXPECT_NEAR(gradvals[0], grads[0]->at(0, 0, 0), 1.e-6)<<
    "Expected value of graident with respect to x2 is "<<gradvals[0];
    EXPECT_NEAR(gradvals[1], grads[1]->at(0, 0, 0), 1.e-6)<<
    "Expected value of graident with respect to x2 is "<<gradvals[1];

    BNNMemory->free_memory(x1);
    BNNMemory->free_memory(x2);
    BNNMemory->free_memory(x3);
    BNNMemory->free_memory(grads[0]);
    BNNMemory->free_memory(grads[1]);
}

TEST(Autodiff, MatMulComputeGradientReverse)
{
    TensorCPU<float> *x1, *x2, *x3, *x4;
    unsigned ndims = 2;
    unsigned* shape = new unsigned[ndims];
    shape[0] = 3, shape[1] = 3;
    x1 = new TensorCPU<float>(shape, ndims);
    x2 = new TensorCPU<float>(shape, ndims);
    x3 = new TensorCPU<float>(shape, ndims);
    x4 = new TensorCPU<float>(shape, ndims);
    float A[3][3] = {{1, -1, 7}, {-2, 3, 3}, {1, 0, 1}};
    float B[3][3] = {{8, -9, -6}, {1, -3, -4}, {2, 8, -8}};
    float C[3][3] = {{1, 0, 0}, {0, 3, 0}, {0, 0, 1}};
    float D[3][3] = {{1, 2, 1}, {2, 3, 4}, {4, 1, 2}};
    for( int i = 0; i < shape[0]; i++ )
    {
        for( int j = 0; j < shape[1]; j++ )
        {
            x1->set(A[i][j], i, j);
            x2->set(B[i][j], i, j);
            x3->set(C[i][j], i, j);
            x4->set(D[i][j], i, j);
        }
    }
    Operator<float> *x1x2, *x3x4;
    x1x2 = bnn::operations::matmul(x1, x2), x3x4 = bnn::operations::matmul(x3, x4);
    Operator<float>* expr = bnn::operations::matmul(x1x2, x3x4);
    TensorCPU<float>** vars = new TensorCPU<float>*[1];
    vars[0] = x3;
    TensorCPU<float>** grads =  compute_gradient_reverse(expr, vars, 1);
    float gradvals[3][3] = {{96., 216., 168.}, {328., 738., 574.}, {-384., -864., -672.}};
    for( int i = 0; i < shape[0]; i++ )
    {
        for( int j = 0; j < shape[1]; j++ )
        {
            EXPECT_NEAR(gradvals[i][j], grads[0]->at(i, j), 1.e-6)<<
            "Expected value of graident with respect to x3 at ("<<i<<", "<<j<<") is "<<gradvals[i][j];
        }
    }

    delete BNNMemory;
    delete BNNThreads;
}

int main(int ac, char* av[])
{
    if(ac == 2 && strcmp(av[1], "--CI=ON") == 0)
    {
        testing::GTEST_FLAG(filter) = "Autodiff.BuildGraph";
    }
    testing::InitGoogleTest(&ac, av);
    return RUN_ALL_TESTS();
}
