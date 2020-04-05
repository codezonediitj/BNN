#ifndef BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP

#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>

namespace bnn
{
    namespace operators
    {
        Operator::Operator
        (std::string _name):
        name(_name)
        {}

        std::string
        Operator::get_name()
        {
            return this->name;
        }

        bool
        Operator::is_tensor()
        {
            return false;
        }

        Operator*
        Operator::get_arg()
        {
            return NULL;
        }

        Operator*
        Operator::get_arg(bool index)
        {
            return NULL;
        }

        UnaryOperator::UnaryOperator
        (std::string _name):
        x(NULL),
        Operator::Operator(_name)
        {}

        UnaryOperator::UnaryOperator
        (Operator* a,
         std::string _name):
        x(a),
        Operator::Operator(_name)
        {}

        Operator*
        UnaryOperator::get_arg()
        {
            return this->x;
        }

        BinaryOperator::BinaryOperator
        (std::string _name):
        x(NULL),
        y(NULL),
        Operator::Operator(_name)
        {}

        BinaryOperator::BinaryOperator
        (Operator* a,
         Operator* b,
         std::string _name):
        x(a),
        y(b),
        Operator::Operator(_name)
        {}

        Operator*
        BinaryOperator::get_arg
        (bool idx)
        {
            return idx ? this->y : this->x;
        }

        template <class data_type>
        unsigned long int
        TensorWrapper<data_type>::_id = 0;

        template <class data_type>
        TensorWrapper<data_type>::TensorWrapper
        ():
        t(NULL),
        Operator::Operator
        ("TensorWrapper")
        {}

        template <class data_type>
        TensorWrapper<data_type>::TensorWrapper
        (bnn::core::TensorCPU<data_type>& _t):
        t(&_t),
        Operator::Operator
        ("TensorWrapper_" + std::to_string(_id++))
        {}

        template <class data_type>
        bnn::core::TensorCPU<data_type>*
        TensorWrapper<data_type>::get_tensor()
        {
            return this->t;
        }

        template <class data_type>
        bool
        TensorWrapper<data_type>::is_tensor()
        {
            return true;
        }

        unsigned long int
        Add::_id = 0;

        Add::Add():
        BinaryOperator::
        BinaryOperator("Add")
        {}

        Add::Add
        (Operator* a,
         Operator* b):
        BinaryOperator::BinaryOperator
        (a, b, "Add_" + std::to_string(_id++))
        {}

        #include "bnn/templates/operations_operators.hpp"

    }
}

#endif
