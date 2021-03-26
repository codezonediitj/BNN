#ifndef BNN_BNN_OPERATIONS_OPERATIONS_HPP
#define BNN_BNN_OPERATIONS_OPERATIONS_HPP

#include <bnn/operations/operators.hpp>
#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace operations
    {

        using namespace bnn::core;
        using namespace bnn::operators;

        /* @overload
        * Creates a bnn::operators::Add object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Add<data_type>*
        add
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Add object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Add<data_type>*
        add
        (TensorCPU<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Add object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a Pointer to Operator object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Add<data_type>*
        add
        (Operator<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Add object.
        *
        * @param a Pointer to Operator object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Add<data_type>*
        add
        (Operator<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Exp object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        */
        template <class data_type>
        Exp<data_type>*
        exp
        (TensorCPU<data_type>* a);

        /* @overload
        * Creates a bnn::operators::Exp object.
        *
        * @param a Pointer to Operator object.
        */
        template <class data_type>
        Exp<data_type>*
        exp
        (Operator<data_type>* a);

        /* @overload
        * Creates a bnn::operators::Log object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        */
        template <class data_type>
        Log<data_type>*
        log
        (TensorCPU<data_type>* a);

        /* @overload
        * Creates a bnn::operators::Log object.
        *
        * @param a Pointer to Operator object.
        */
        template <class data_type>
        Log<data_type>*
        log
        (Operator<data_type>* a);

        /* @overload
        * Creates a bnn::operators::Rectifier object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        */
        template <class data_type>
        Rectifier<data_type>*
        rectifier
        (TensorCPU<data_type>* a);

        /* @overload
        * Creates a bnn::operators::Rectifier object.
        *
        * @param a Pointer to Operator object.
        */
        template <class data_type>
        Rectifier<data_type>*
        rectifier
        (Operator<data_type>* a);

        /* @overload
        * Creates a bnn::operators::MatMul object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        MatMul<data_type>*
        matmul
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::MatMul object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        MatMul<data_type>*
        matmul
        (TensorCPU<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::MatMul object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a Pointer to Operator object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        MatMul<data_type>*
        matmul
        (Operator<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::MatMul object.
        *
        * @param a Pointer to Operator object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        MatMul<data_type>*
        matmul
        (Operator<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Multiply object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Multiply<data_type>*
        multiply
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Multiply object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Multiply<data_type>*
        multiply
        (TensorCPU<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Multiply object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a Pointer to Operator object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Multiply<data_type>*
        multiply
        (Operator<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Multiply object.
        *
        * @param a Pointer to Operator object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Multiply<data_type>*
        multiply
        (Operator<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Divide object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Divide<data_type>*
        divide
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Divide object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Divide<data_type>*
        divide
        (TensorCPU<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Divide object.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a Pointer to Operator object.
        * @param b TensorCPU object.
        */
        template <class data_type>
        Divide<data_type>*
        divide
        (Operator<data_type>* a, TensorCPU<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Divide object.
        *
        * @param a Pointer to Operator object.
        * @param b Pointer to Operator object.
        */
        template <class data_type>
        Divide<data_type>*
        divide
        (Operator<data_type>* a, Operator<data_type>* b);

        /* @overload
        * Creates a bnn::operators::Sum object along the specified axis.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param a TensorCPU object.
        * @param axis Axis along which the sum is to be performed.
        *             Optional, by default, 0.
        */
        template <class data_type>
        Sum<data_type>*
        sum
        (TensorCPU<data_type>* a, unsigned int axis=0);

        /* @overload
        * Creates a bnn::operators::Sum object along the specified axis.
        *
        * @param a Pointer to Operator object.
        * @param axis Axis along which the sum is to be performed.
        *             Optional, by default, 0.
        */
        template <class data_type>
        Sum<data_type>*
        sum
        (Operator<data_type>* a, unsigned int axis=0);

    }
}

#endif
