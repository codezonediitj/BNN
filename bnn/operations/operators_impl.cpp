#ifndef BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP

#include <vector>
#include <iostream>
#include <bnn/core/tensor.hpp>
#include <bnn/core/tensor_ops.hpp>
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
        value(NULL),
        gradient(NULL)
        {
        }

        template <class data_type>
        Operator<data_type>::
        Operator
        (TensorCPU<data_type>* _value, string _name):
        name(_name),
        value(_value),
        gradient(NULL)
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
        TensorCPU<data_type>*
        Operator<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            return 0;
        }

        template <class data_type>
        TensorCPU<data_type>*
        Operator<data_type>::
        compute_value
        ()
        {
            return 0;
        }

        template <class data_type>
        TensorCPU<data_type>*
        Operator<data_type>::
        get_value
        ()
        {
            return this->value;
        }

        template <class data_type>
        TensorCPU<data_type>*
        Operator<data_type>::
        get_gradient
        ()
        {
            return this->gradient;
        }

        template <class data_type>
        void
        Operator<data_type>::
        set_gradient
        (TensorCPU<data_type>* _gradient)
        {
            this->gradient = _gradient;
        }


        template <class data_type>
        void
        Operator<data_type>::
        set_value
        (TensorCPU<data_type>* _value)
        {
            this->value = _value;
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
        Operator<data_type>::Operator
        ("TensorWrapper")
        {
        }

        template <class data_type>
        TensorWrapper<data_type>::
        TensorWrapper
        (TensorCPU<data_type>* _t):
        Operator<data_type>::Operator
        (_t, "TensorWrapper_" + to_string(_id++))
        {
        }

        template <class data_type>
        TensorCPU<data_type>*
        TensorWrapper<data_type>::
        compute_value
        ()
        {
            return this->get_value();
        }

        template <class data_type>
        TensorCPU<data_type>*
        TensorWrapper<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            TensorCPU<data_type>* t;
            t = this->get_value();
            vector<unsigned> shape
            (t->get_shape(), t->get_shape() + t->get_ndims());
            TensorCPU<data_type>* grad = new TensorCPU<data_type>(shape);
            bnn::core::fill(grad, (data_type)(var == t));
            return grad;
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
        TensorCPU<data_type>*
        Add<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y =  this->get_arg(1);
            return add(x->get_value(), y->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Add<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y =  this->get_arg(1);
            return add(x->get_gradient(), y->get_gradient());
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

        template <class data_type>
        TensorCPU<data_type>*
        Exp<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return exp(x->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Exp<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return mul(exp(x->get_value()), x->get_gradient());
        }

        #include "bnn/templates/operations/operators.hpp"

    }
}

#endif
