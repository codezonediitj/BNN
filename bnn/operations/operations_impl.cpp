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
        Add* add(TensorCPU<data_type>& a, TensorCPU<data_type>& b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return add(ta, tb);
        }

        template <class data_type>
        Add* add(TensorCPU<data_type>& a, Operator* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return add(ta, b);
        }

        template <class data_type>
        Add* add(Operator* a, TensorCPU<data_type>& b)
        {
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return add(a, tb);
        }

        Add* add(Operator* a, Operator* b)
        {
            Add* result = new Add(a, b);
            return result;
        }

        template <class data_type>
        Exp* exp(TensorCPU<data_type>& a)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return exp(ta);
        }

        Exp* exp(Operator* a)
        {
            Exp* result = new Exp(a);
            return result;
        }

        #include "bnn/templates/operations_operations.hpp"
    }
}

#endif
