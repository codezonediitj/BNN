#include<gtest/gtest.h>
#include<string>
#include<stdexcept>
#include<vector>
#include<bnn/core/tensor.hpp>
#include <bnn/operations/operations.hpp>
#include <bnn/operations/operators.hpp>

TEST(Core, Add)
{
    using namespace bnn::core;
    using namespace bnn::operators;
    using namespace bnn::operations;

    std::vector<unsigned> shape = {1};
    std::vector<unsigned> _shape = {2, 2};
    TensorCPU<float> t1(shape), t2(shape), t3(_shape);
    EXPECT_THROW({
        try
        {
            add(t1, t3);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Got tensors of different shapes.", e.what());
            throw;
        }
    }, std::logic_error);
    t1.set(1., 0);
    t2.set(3., 0);
    Operator* res1 = add(t1, t2);
    Operator* res2 = add(t1, t2);
    EXPECT_EQ("Add_0", res1->get_name());
    EXPECT_EQ("Add_1", res2->get_name());
    Operator* _twa = res1->get_arg(0);
    Operator* _twb = res2->get_arg(1);
    TensorWrapper<float>* twa;
    if(_twa->is_tensor())
        twa = dynamic_cast<TensorWrapper<float>*>(_twa);
    TensorWrapper<float>* twb;
    if(_twa->is_tensor())
        twb = dynamic_cast<TensorWrapper<float>*>(_twb);
    TensorCPU<float>* ta = twa->get_tensor();
    TensorCPU<float>* tb = twb->get_tensor();
    EXPECT_EQ(1., ta->at(0));
    EXPECT_EQ(3., tb->at(0));
}

int main(int ac, char* av[])
{
  testing::InitGoogleTest(&ac, av);
  return RUN_ALL_TESTS();
}
