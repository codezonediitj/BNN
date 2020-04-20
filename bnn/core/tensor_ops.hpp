#ifndef BNN_BNN_CORE_TENSOR_OPS_HPP
#define BNN_BNN_CORE_TENSOR_OPS_HPP

#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace core
    {

        template <class data_type>
        TensorCPU<data_type>*
        add
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y);

        template <class data_type>
        TensorCPU<data_type>*
        mul
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y);

        template <class data_type>
        TensorCPU<data_type>*
        exp
        (TensorCPU<data_type>* x);

        template <class data_type>
        void
        fill
        (TensorCPU<data_type>* x, data_type val);

    }
}

#endif
