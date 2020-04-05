#ifndef BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP

#include<bnn/operations/operators.hpp>
#include<bnn/operations/operations.hpp>

namespace bnn
{
    namespace operations
    {
        template <class data_type>
        bnn::operators::Add*
        add(bnn::core::TensorCPU<data_type>& a,
            bnn::core::TensorCPU<data_type>& b)
        {
            using namespace bnn::operators;
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return add(ta, tb);
        }

        bnn::operators::Add*
        add(bnn::operators::Operator* a,
            bnn::operators::Operator* b)
        {
            using namespace bnn::operators;
            Add* result = new Add(a, b);
            return result;
        }

        #include "bnn/templates/operations_operations.hpp"
    }
}

#endif
