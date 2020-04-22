#ifndef BNN_BNN_CORE_TENSOR_OPS_HPP
#define BNN_BNN_CORE_TENSOR_OPS_HPP

#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace core
    {

        /*
        * Performs point wise addition of two tensors.
        * Returns the result in a new tensor.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>*
        * @param y Tensor<data_type>*
        */
        template <class data_type>
        TensorCPU<data_type>*
        add
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y);

        /*
        * Performs point wise multiplication of two tensors.
        * Returns the result in a new tensor.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>*
        * @param y Tensor<data_type>*
        */
        template <class data_type>
        TensorCPU<data_type>*
        mul
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y);

        /*
        * Performs point wise exponentiation of a tensor.
        * Returns the result in a new tensor.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>*
        */
        template <class data_type>
        TensorCPU<data_type>*
        exp
        (TensorCPU<data_type>* x);

        /*
        * Fills the tensor with a given value.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>* The tensor to be
        *     filled.
        * @param val data_type The value with which
        *     the tensor is to be filled.
        */
        template <class data_type>
        void
        fill
        (TensorCPU<data_type>* x, data_type val);

    }
}

#endif
