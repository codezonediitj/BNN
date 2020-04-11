#ifndef BNN_BNN_OPERATIONS_OPERATIONS_HPP
#define BNN_BNN_OPERATIONS_OPERATIONS_HPP

#include<bnn/operations/operators.hpp>
#include<bnn/core/tensor.hpp>

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
        (TensorCPU<data_type>& a, TensorCPU<data_type>& b);

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
        (TensorCPU<data_type>& a, Operator<data_type>* b);

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
        (Operator<data_type>* a, TensorCPU<data_type>& b);

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
        (TensorCPU<data_type>& a);

        /* @overload
        * Creates a bnn::operators::Exp object.
        *
        * @param a Pointer to Operator object.
        */
        template <class data_type>
        Exp<data_type>*
        exp
        (Operator<data_type>* a);

    }
}

#endif
