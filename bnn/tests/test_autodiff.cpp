#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operations.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/autodiff/graph.hpp>
#include <bnn/autodiff/forward.hpp>
#include <bnn/core/tensor_ops.hpp>

using namespace bnn::core;
using namespace bnn::operators;
using namespace bnn::operations;
using namespace bnn::autodiff;

TEST(Autodiff, BuildGraphForward)
{
    vector<unsigned> shape = {3, 3, 3};
    TensorCPU<float> x1(shape), x2(shape), x3(shape);
    Operator<float>* expr = bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(&x1, &x2)),
        bnn::operations::exp(&x2)
    );

    ForwardGraphNode<float>* graph = build_graph_forward(expr);
    vector<string>* _graph = new vector<string>[4];
    _graph[0] = {"Add_1"};
    _graph[1] = {"Exp_1", "Exp_0"};
    _graph[2] = {"Add_0", "TensorWrapper_0"};
    _graph[3] = {"TensorWrapper_1", "TensorWrapper_2"};
    ForwardGraphNode<float>* layer = graph;
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

TEST(Autodiff, ComputeGradient)
{
    bnn::core::TensorCPU<float> *x1, *x2, *x3;
    unsigned ndims = 3;
    unsigned* shape = new unsigned[ndims];
    shape[0] = 1, shape[1] = 1000, shape[2] = 1000;
    x1 = new bnn::core::TensorCPU<float>(shape, ndims);
    x2 = new bnn::core::TensorCPU<float>(shape, ndims);
    x3 = new bnn::core::TensorCPU<float>(shape, ndims);
    bnn::core::fill<float>(x1, 3.);
    bnn::core::fill<float>(x2, 2.);
    bnn::core::fill<float>(x3, 1.);
    bnn::operators::Operator<float>* expr =
    bnn::operations::add(
        bnn::operations::exp(bnn::operations::add(x1, x2)),
        bnn::operations::exp(bnn::operations::add(x2, x3))
    );
    bnn::core::TensorCPU<float> *gradx2 =  bnn::autodiff::compute_gradient_forward(expr, x2);
    for(unsigned i = 0; i < gradx2->get_shape()[0]; i++)
    {
        for(unsigned j = 0; j < gradx2->get_shape()[1]; j++)
        {
            for(unsigned k = 0; k < gradx2->get_shape()[2]; k++)
            {
                float gradval = 168.49870300292969;
                EXPECT_NEAR(gradval, gradx2->at(i, j, k), 1.e-6)<<
                "Expected value of graident with respect to x2 is 168.4987";
            }
        }
    }
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
