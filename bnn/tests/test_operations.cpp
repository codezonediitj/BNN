#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operations.hpp>
#include <bnn/operations/operators.hpp>


using namespace bnn::core;
using namespace bnn::operators;
using namespace bnn::operations;

TEST(Operations, Add)
{
    std::vector<unsigned> shape = {1};
    TensorCPU<float> t1(shape), t2(shape);
    t1.set(1., 0);
    t2.set(3., 0);
    Operator<float>* res1 = add(&t1, &t2);
    Operator<float>* res2 = add(res1, &t1);
    Operator<float>* res3 = add(&t2, res2);

    EXPECT_EQ("Add_0", res1->get_name());
    EXPECT_EQ("Add_1", res2->get_name());
    EXPECT_EQ("Add_2", res3->get_name());

    Operator<float>* res1arg1 = res1->get_arg(0);
    Operator<float>* res1arg2 = res1->get_arg(1);
    TensorCPU<float>* ta = res1arg1->get_value();
    TensorCPU<float>* tb = res1arg2->get_value();
    EXPECT_EQ(1., ta->at(0));
    EXPECT_EQ(3., tb->at(0));

    Operator<float>* res2arg1 = res2->get_arg(0);
    Operator<float>* res2arg2 = res2->get_arg(1);
    EXPECT_EQ("Add_0", res2arg1->get_name());
    tb = res2arg2->get_value();
    EXPECT_EQ(1., tb->at(0));

    Operator<float>* res3arg1 = res3->get_arg(0);
    Operator<float>* res3arg2 = res3->get_arg(1);
    EXPECT_EQ("Add_1", res3arg2->get_name());
    ta = res3arg1->get_value();
    EXPECT_EQ(3., ta->at(0));
}

TEST(Operations, Exp)
{
    std::vector<unsigned> shape = {1};
    TensorCPU<float> t1(shape), t2(shape);
    t1.set(1., 0);
    t2.set(3., 0);
    Operator<float>* res1 = exp(&t1);
    Operator<float>* res2 = exp(res1);

    EXPECT_EQ("Exp_0", res1->get_name());
    EXPECT_EQ("Exp_1", res2->get_name());

    Operator<float>* res1arg1 = res1->get_arg();
    TensorCPU<float>* ta = res1arg1->get_value();
    EXPECT_EQ(1., ta->at(0));

    Operator<float>* res2arg1 = res2->get_arg();
    EXPECT_EQ("Exp_0", res2arg1->get_name());
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
