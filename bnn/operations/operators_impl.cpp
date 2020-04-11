#ifndef BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP

#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>

namespace bnn
{
    namespace operators
    {

        using namespace std;
        using namespace bnn::core;

        template <class data_type>
        Operator<data_type>::
        Operator
        (string _name):
        name(_name),
        value(0),
        gradient(0),
        variable(NULL)
        {
        }

        template <class data_type>
        string
        Operator<data_type>::
        get_name
        ()
        {
            return this->name;
        }

        template <class data_type>
        bool
        Operator<data_type>::
        is_tensor
        ()
        {
            return false;
        }

        template <class data_type>
        Operator<data_type>*
        Operator<data_type>::
        get_arg
        ()
        {
            return NULL;
        }

        template <class data_type>
        Operator<data_type>*
        Operator<data_type>::
        get_arg
        (bool index)
        {
            return NULL;
        }

        template <class data_type>
        data_type
        Operator<data_type>::
        compute_gradient
        ()
        {
            return 0;
        }

        template <class data_type>
        data_type
        Operator<data_type>::
        compute_value
        ()
        {
            return 0;
        }

        template <class data_type>
        data_type
        Operator<data_type>::
        get_value
        ()
        {
            return this->value;
        }

        template <class data_type>
        data_type
        Operator<data_type>::
        get_gradient
        ()
        {
            return this->gradient;
        }

        template <class data_type>
        void
        Operator<data_type>::
        set_value
        (data_type _value)
        {
            this->value = _value;
        }

        template <class data_type>
        Operator<data_type>*
        Operator<data_type>::
        get_variable
        ()
        {
            return this->variable;
        }

        template <class data_type>
        void
        Operator<data_type>::
        set_variable
        (Operator<data_type>* _var)
        {
            this->variable = _var;
        }

        template <class data_type>
        unsigned
        Operator<data_type>::
        num_args
        ()
        {
            return 0;
        }

        template <class data_type>
        UnaryOperator<data_type>::
        UnaryOperator
        (string _name):
        x(NULL),
        Operator<data_type>::Operator(_name)
        {
        }

        template <class data_type>
        UnaryOperator<data_type>::
        UnaryOperator
        (Operator<data_type>* a, string _name):
        x(a),
        Operator<data_type>::Operator(_name)
        {
        }

        template <class data_type>
        Operator<data_type>*
        UnaryOperator<data_type>::
        get_arg
        ()
        {
            return this->x;
        }

        template <class data_type>
        unsigned
        UnaryOperator<data_type>::
        num_args
        ()
        {
            return 1;
        }

        template <class data_type>
        BinaryOperator<data_type>::
        BinaryOperator
        (string _name):
        x(NULL),
        y(NULL),
        Operator<data_type>::Operator(_name)
        {
        }

        template <class data_type>
        BinaryOperator<data_type>::
        BinaryOperator
        (Operator<data_type>* a,
         Operator<data_type>* b,
         string _name):
        x(a),
        y(b),
        Operator<data_type>::Operator(_name)
        {
        }

        template <class data_type>
        Operator<data_type>*
        BinaryOperator<data_type>::
        get_arg
        (bool idx)
        {
            return idx ? this->y : this->x;
        }

        template <class data_type>
        unsigned
        BinaryOperator<data_type>::
        num_args
        ()
        {
            return 2;
        }

        template <class data_type>
        unsigned long int
        TensorWrapper<data_type>::_id = 0;

        template <class data_type>
        TensorWrapper<data_type>::
        TensorWrapper
        ():
        t(NULL),
        Operator<data_type>::Operator
        ("TensorWrapper")
        {
        }

        template <class data_type>
        TensorWrapper<data_type>::
        TensorWrapper
        (TensorCPU<data_type>& _t):
        t(&_t),
        Operator<data_type>::Operator
        ("TensorWrapper_" + to_string(_id++))
        {
        }

        template <class data_type>
        TensorCPU<data_type>*
        TensorWrapper<data_type>::
        get_tensor
        ()
        {
            return this->t;
        }

        template <class data_type>
        bool
        TensorWrapper<data_type>::
        is_tensor
        ()
        {
            return true;
        }

        template <class data_type>
        unsigned
        TensorWrapper<data_type>::
        num_args
        ()
        {
            return 0;
        }

        template <class data_type>
        unsigned long int
        Add<data_type>::_id = 0;

        template <class data_type>
        Add<data_type>::
        Add
        ():
        BinaryOperator<data_type>::BinaryOperator("Add")
        {
        }

        template <class data_type>
        Add<data_type>::
        Add
        (Operator<data_type>* a, Operator<data_type>* b):
        BinaryOperator<data_type>::BinaryOperator
        (a, b, "Add_" + std::to_string(_id++))
        {
        }

        template <class data_type>
        unsigned long int
        Exp<data_type>::_id = 0;

        template <class data_type>
        Exp<data_type>::
        Exp
        ():
        UnaryOperator<data_type>::UnaryOperator("Exp")
        {
        }

        template <class data_type>
        Exp<data_type>::
        Exp
        (Operator<data_type>* a):
        UnaryOperator<data_type>::UnaryOperator
        (a, "Exp_" + std::to_string(_id++))
        {
        }

        #include "bnn/templates/operations_operators.hpp"

    }
}

#endif
