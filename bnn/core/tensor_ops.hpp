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

        /*
        * Performs a deep copy from the destination
        * tensor to source tensor.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param des Tensor<data_type>* Destination
        *     tensor.
        * @param src Tensor<data_type>* Source
        *     tensor.
        */
        template <class data_type>
        void
        copy
        (TensorCPU<data_type>* dest, TensorCPU<data_type>* src);

        /*
        * Performs summation of elements of the given
        * Tensor along the given axis.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>* Tensor whose
        *     elements are to be summed up.
        * @param axis int The axis along which summation
        *     is to be done.
        *     By default, performs summation along all
        *     the axis.
        */
        template <class data_type>
        TensorCPU<data_type>*
        sum
        (TensorCPU<data_type>* x, int axis=-1);

        /*
        * Performs division of elements of the given
        * Tensor by a scalar.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>* Tensor whose
        *     elements are to be considered.
        * @param divisor data_type
        */
        template <class data_type>
        TensorCPU<data_type>*
        divide
        (TensorCPU<data_type>* x, data_type divisor);

        /*
        * Returns a one hot tensor with a new axis appended
        * in its shape.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param x Tensor<data_type>* Tensor whose
        *     elements are to be converted to one
        *     hot notation.
        * @param on_value data_type The value which
        *     is to be used for filling in the one
        *     hot tensor when the value in the original
        *     tensor matches with the index in the
        *     range [0, depth) for the new axis.
        * @param off_value data_type The value which
        *     is to be used for filling in the one
        *     hot tensor when the value in the original
        *     tensor doesn't match with the index in the
        *     range [0, depth) for the new axis.
        * @param depth unsigned The size of the axis which
        *     is to be appended.
        */
        template <class data_type>
        TensorCPU<data_type>*
        one_hot
        (TensorCPU<data_type>* x, data_type on_value,
         data_type off_value, unsigned depth);

    }
}

#endif
