#include<gtest/gtest.h>
#include<string>
#include<stdexcept>
#include<vector>
#include<bnn/core/tensor.hpp>
#include<iostream>

TEST(Core, TensorCPU)
{
    bnn::core::TensorCPU<float> t_f;
    EXPECT_EQ(0, t_f.get_ndims())<<"Default size should be 0";
    std::vector<unsigned> shape = {3, 3, 3};
    bnn::core::TensorCPU<float> t(shape);
    std::string msg1 = "The size of tensor must be 3";
    EXPECT_EQ(3, t.get_ndims())<<msg1;
    EXPECT_EQ(true,
    t.get_shape()[0] == 3 && t.get_shape()[1] == 3 && t.get_shape()[2] == 3)
    <<"The shape of tensor should be (3, 3, 3)";
    for(unsigned i = 0; i < 3; i++)
    {
        for(unsigned j = 0; j < 3; j++)
        {
            for(unsigned k = 0; k < 3; k++)
            {
                t.set(5., i, j, k);
                EXPECT_EQ(5., t.at(i, j, k))
                <<"Each element of the tensor should be 5.";
            }
        }
    }
    t.set(3., 1, 2, 2);
    EXPECT_EQ(3., t.at(1, 2, 2));
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
