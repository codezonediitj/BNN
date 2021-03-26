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
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b)
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
        (TensorCPU<data_type>* a, Operator<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return add(ta, b);
        }

        template <class data_type>
        Add<data_type>*
        add
        (Operator<data_type>* a, TensorCPU<data_type>* b)
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
        (TensorCPU<data_type>* a)
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

        template <class data_type>
        MatMul<data_type>*
        matmul
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return matmul(ta, tb);
        }

        template <class data_type>
        MatMul<data_type>*
        matmul
        (TensorCPU<data_type>* a, Operator<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return matmul(ta, b);
        }

        template <class data_type>
        MatMul<data_type>*
        matmul
        (Operator<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return matmul(a, tb);
        }

        template <class data_type>
        MatMul<data_type>*
        matmul
        (Operator<data_type>* a, Operator<data_type>* b)
        {
            MatMul<data_type>* result = new MatMul<data_type>(a, b);
            return result;
        }

        template <class data_type>
        Multiply<data_type>*
        multiply
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return multiply(ta, tb);
        }

        template <class data_type>
        Multiply<data_type>*
        multiply
        (TensorCPU<data_type>* a, Operator<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return multiply(ta, b);
        }

        template <class data_type>
        Multiply<data_type>*
        multiply
        (Operator<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return multiply(a, tb);
        }

        template <class data_type>
        Multiply<data_type>*
        multiply
        (Operator<data_type>* a, Operator<data_type>* b)
        {
            Multiply<data_type>* result = new Multiply<data_type>(a, b);
            return result;
        }

        template <class data_type>
        Divide<data_type>*
        divide
        (TensorCPU<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return divide(ta, tb);
        }

        template <class data_type>
        Divide<data_type>*
        divide
        (TensorCPU<data_type>* a, Operator<data_type>* b)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return divide(ta, b);
        }

        template <class data_type>
        Divide<data_type>*
        divide
        (Operator<data_type>* a, TensorCPU<data_type>* b)
        {
            TensorWrapper<data_type>* tb =
            new TensorWrapper<data_type>(b);
            return divide(a, tb);
        }

        template <class data_type>
        Divide<data_type>*
        divide
        (Operator<data_type>* a, Operator<data_type>* b)
        {
            Divide<data_type>* result = new Divide<data_type>(a, b);
            return result;
        }

        template <class data_type>
        Log<data_type>*
        log
        (TensorCPU<data_type>* a)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return log(ta);
        }

        template <class data_type>
        Log<data_type>*
        log
        (Operator<data_type>* a)
        {
            Log<data_type>* result = new Log<data_type>(a);
            return result;
        }

        template <class data_type>
        Rectifier<data_type>*
        rectifier
        (TensorCPU<data_type>* a)
        {
            TensorWrapper<data_type>* ta =
            new TensorWrapper<data_type>(a);
            return rectifier(ta);
        }

        template <class data_type>
        Rectifier<data_type>*
        rectifier
        (Operator<data_type>* a)
        {
            Rectifier<data_type>* result = new Rectifier<data_type>(a);
            return result;
        }

        #include "bnn/templates/operations/operations.hpp"

    }
}

#endif
