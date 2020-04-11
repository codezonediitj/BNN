#ifndef BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP

#include <bnn/operations/operators.hpp>
#include <bnn/operations/operations.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace operations
    {

        using namespace bnn::core;
        using namespace bnn::operators;
        using namespace bnn::utils;

        template <class data_type>
        Add<data_type>*
        add
        (TensorCPU<data_type>& a, TensorCPU<data_type>& b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return add(ta, tb);
        }

        template <class data_type>
        Add<data_type>*
        add
        (TensorCPU<data_type>& a, Operator<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return add(ta, b);
        }

        template <class data_type>
        Add<data_type>*
        add
        (Operator<data_type>* a, TensorCPU<data_type>& b)
        {
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return add(a, tb);
        }

        template <class data_type>
        Add<data_type>*
        add
        (Operator<data_type>* a, Operator<data_type>* b)
        {
            Add<data_type>* result = new Add<data_type>(a, b);
            return result;
        }

        template <class data_type>
        Exp<data_type>*
        exp
        (TensorCPU<data_type>& a)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return exp(ta);
        }

        template <class data_type>
        Exp<data_type>*
        exp
        (Operator<data_type>* a)
        {
            Exp<data_type>* result = new Exp<data_type>(a);
            return result;
        }

        #include "bnn/templates/operations_operations.hpp"

    }
}

#endif
