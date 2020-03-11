#include<gtest/gtest.h>
#include<string>
#include<vector>
#include<bnn/cuda/core/tensor.hpp>

TEST(Core, TensorCPU)
{
    bnn::cuda::core::TensorGPU<float> t_f;
    EXPECT_EQ(0, t_f.get_ndims(true))<<"Default GPU size should be 0";
    EXPECT_EQ(0, t_f.get_ndims(false))<<"Default CPU size should be 0";
    std::vector<unsigned> shape = {3, 3, 3};
    bnn::cuda::core::TensorGPU<float> t(shape);
    EXPECT_EQ(3, t.get_ndims(true))<<"The GPU dimensions of tensor must be 3";
    EXPECT_EQ(3, t.get_ndims(false))<<"The CPU dimensions of tensor must be 3";
    EXPECT_EQ(true,
    t.get_shape(true)[0] == 3 && t.get_shape(true)[1] == 3 && t.get_shape(true)[2] == 3)
    <<"The GPU shape of tensor should be (3, 3, 3)";
    EXPECT_EQ(true,
    t.get_shape(false)[0] == 3 && t.get_shape(false)[1] == 3 && t.get_shape(false)[2] == 3)
    <<"The CPU shape of tensor should be (3, 3, 3)";
    for(unsigned i = 0; i < 3; i++)
    {
        for(unsigned j = 0; j < 3; j++)
        {
            for(unsigned k = 0; k < 3; k++)
            {
                t.set(true, 5., i, j, k);
                EXPECT_EQ(5., t.at(true, i, j, k))
                <<"Each element of the tensor should be 5.";
            }
        }
    }
    t.set(true, 3., 1, 2, 2);
    EXPECT_EQ(3., t.at(true, 1, 2, 2));
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
