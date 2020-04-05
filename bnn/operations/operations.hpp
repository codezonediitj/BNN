#ifndef BNN_BNN_OPERATIONS_OPERATIONS_HPP
#define BNN_BNN_OPERATIONS_OPERATIONS_HPP

#include<bnn/operations/operators.hpp>
#include<bnn/core/tensor.hpp>

namespace bnn
{
    namespace operations
    {
        template <class data_type>
        bnn::operators::Add*
        add(bnn::core::TensorCPU<data_type>& a,
            bnn::core::TensorCPU<data_type>& b);

        bnn::operators::Add*
        add(bnn::operators::Operator* a,
            bnn::operators::Operator* b);
    }
}

#endif
