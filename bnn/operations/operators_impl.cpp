#ifndef BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATORS_IMPL_CPP

#include <vector>
#include <bnn/core/tensor.hpp>
#include <bnn/core/tensor_ops.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace operators
    {

        using namespace std;
        using namespace bnn::core;

        string msg = "This is an abstract method";

        template <class data_type>
        Operator<data_type>::
        Operator
        (string _name):
        name(_name),
        value(NULL),
        gradient(NULL)
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Operator<data_type>::
        Operator
        (TensorCPU<data_type>* _value, string _name):
        name(_name),
        value(_value),
        gradient(NULL)
        {
            BNNMemory->push(this);
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
            check(false, msg);
        }

        template <class data_type>
        Operator<data_type>*
        Operator<data_type>::
        get_arg
        (bool index)
        {
            check(false, msg);
        }

        template <class data_type>
        bool
        Operator<data_type>::
        is_variable
        ()
        {
            return variable;
        }

        template <class data_type>
        void
        Operator<data_type>::
        set_variable
        (bool _val)
        {
            this->variable = _val;
        }

        template <class data_type>
        TensorCPU<data_type>*
        Operator<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            check(false, msg);
        }

        template <class data_type>
        void
        Operator<data_type>::
        compute_gradient_reverse
        ()
        {
            check(false, msg);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Operator<data_type>::
        compute_value
        ()
        {
            check(false, msg);
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
        Operator<data_type>::
        ~Operator
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        UnaryOperator<data_type>::
        UnaryOperator
        (string _name):
        x(NULL),
        Operator<data_type>::Operator(_name)
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        UnaryOperator<data_type>::
        UnaryOperator
        (Operator<data_type>* a, string _name):
        x(a),
        Operator<data_type>::Operator(_name)
        {
            BNNMemory->push(this);
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
        UnaryOperator<data_type>::
        ~UnaryOperator
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        BinaryOperator<data_type>::
        BinaryOperator
        (string _name):
        x(NULL),
        y(NULL),
        Operator<data_type>::Operator(_name)
        {
            BNNMemory->push(this);
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
            BNNMemory->push(this);
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
        BinaryOperator<data_type>::
        ~BinaryOperator
        ()
        {
            BNNMemory->invalidate(this);
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
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorWrapper<data_type>::
        TensorWrapper
        (TensorCPU<data_type>* _t):
        Operator<data_type>::Operator
        (_t, "TensorWrapper_" + to_string(_id++))
        {
            BNNMemory->push(this);
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
        void
        TensorWrapper<data_type>::
        compute_gradient_reverse
        ()
        {
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
        TensorWrapper<data_type>::
        ~TensorWrapper
        ()
        {
            BNNMemory->invalidate(this);
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
            BNNMemory->push(this);
        }

        template <class data_type>
        Add<data_type>::
        Add
        (Operator<data_type>* a, Operator<data_type>* b):
        BinaryOperator<data_type>::BinaryOperator
        (a, b, "Add_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
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
        void
        Add<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg1 = this->get_arg(0);
            Operator<data_type>* arg2 = this->get_arg(1);
            TensorCPU<data_type>* dy_dcurr = this->get_gradient();
            if(arg1->is_variable())
            {
                TensorCPU<data_type>* dy_darg1 = new TensorCPU<data_type>
                (dy_dcurr->get_shape(), dy_dcurr->get_ndims());
                bnn::core::copy(dy_darg1, dy_dcurr);
                arg1->set_gradient(dy_darg1);
            }
            if(arg2->is_variable())
            {
                TensorCPU<data_type>* dy_darg2 = new TensorCPU<data_type>
                (dy_dcurr->get_shape(), dy_dcurr->get_ndims());
                bnn::core::copy(dy_darg2, dy_dcurr);
                arg2->set_gradient(dy_darg2);
            }
        }

        template <class data_type>
        Add<data_type>::
        ~Add
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        Multiply<data_type>::_id = 0;

        template <class data_type>
        Multiply<data_type>::
        Multiply
        ():
        BinaryOperator<data_type>::BinaryOperator("Multiply")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Multiply<data_type>::
        Multiply
        (Operator<data_type>* a, Operator<data_type>* b):
        BinaryOperator<data_type>::BinaryOperator
        (a, b, "Multiply_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Multiply<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y = this->get_arg(1);
            return multiply(x->get_value(), y->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Multiply<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y =  this->get_arg(1);
            return add(multiply(x->get_gradient(), y->get_value()),
                       multiply(x->get_value(), y->get_gradient()));
        }

        template <class data_type>
        void
        Multiply<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg1 = this->get_arg(0);
            Operator<data_type>* arg2 = this->get_arg(1);
            TensorCPU<data_type>* dy_dcurr = this->get_gradient();
            if(arg1->is_variable())
            {
                arg1->set_gradient(multiply(dy_dcurr, arg2->get_value()));
            }
            if(arg2->is_variable())
            {
                arg2->set_gradient(multiply(dy_dcurr, arg1->get_value()));
            }
        }

        template <class data_type>
        Multiply<data_type>::
        ~Multiply
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        Divide<data_type>::_id = 0;

        template <class data_type>
        Divide<data_type>::
        Divide
        ():
        BinaryOperator<data_type>::BinaryOperator("Divide")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Divide<data_type>::
        Divide
        (Operator<data_type>* a, Operator<data_type>* b):
        BinaryOperator<data_type>::BinaryOperator
        (a, b, "Divide_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Divide<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y = this->get_arg(1);
            return divide(x->get_value(), y->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Divide<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x, *y;
            x = this->get_arg(0), y = this->get_arg(1);
            TensorCPU<data_type>* left = divide(x->get_gradient(), y->get_value());
            TensorCPU<data_type>* right = divide(multiply(this->get_value(), y->get_gradient()),
                                                 y->get_value());
            return subtract(left, right);
        }

        template <class data_type>
        void
        Divide<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg1 = this->get_arg(0);
            Operator<data_type>* arg2 = this->get_arg(1);
            TensorCPU<data_type>* dy_dcurr = this->get_gradient();
            if(arg1->is_variable())
            {
                arg1->set_gradient(divide(dy_dcurr, arg2->get_value()));
            }
            if(arg2->is_variable())
            {
                TensorCPU<data_type>* neg_one = new TensorCPU<data_type>
                (dy_dcurr->get_shape(), dy_dcurr->get_ndims());
                bnn::core::fill(neg_one, (data_type)-1.0);
                arg2->set_gradient(multiply(dy_dcurr,
                                        multiply(neg_one, divide(this->get_value(),
                                                            arg2->get_value()))));
            }
        }

        template <class data_type>
        Divide<data_type>::
        ~Divide
        ()
        {
            BNNMemory->invalidate(this);
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
            BNNMemory->push(this);
        }

        template <class data_type>
        Exp<data_type>::
        Exp
        (Operator<data_type>* a):
        UnaryOperator<data_type>::UnaryOperator
        (a, "Exp_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
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
            return multiply(this->get_value(), x->get_gradient());
        }

        template <class data_type>
        void
        Exp<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg = this->get_arg();
            if(arg->is_variable())
            {
                TensorCPU<data_type>* dy_dcurr = this->get_gradient();
                TensorCPU<data_type>* dcurr_darg = this->get_value();
                arg->set_gradient(multiply(dy_dcurr, dcurr_darg));
            }
        }

        template <class data_type>
        Exp<data_type>::
        ~Exp
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        Log<data_type>::_id = 0;

        template <class data_type>
        Log<data_type>::
        Log
        ():
        UnaryOperator<data_type>::UnaryOperator("Log")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Log<data_type>::
        Log
        (Operator<data_type>* a):
        UnaryOperator<data_type>::UnaryOperator
        (a, "Log_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Log<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return log(x->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Log<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return divide(x->get_gradient(), this->get_value());
        }

        template <class data_type>
        void
        Log<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg = this->get_arg();
            if(arg->is_variable())
            {
                TensorCPU<data_type>* dy_dcurr = this->get_gradient();
                arg->set_gradient(divide(dy_dcurr, arg->get_value()));
            }
        }

        template <class data_type>
        Log<data_type>::
        ~Log
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        Rectifier<data_type>::_id = 0;

        template <class data_type>
        Rectifier<data_type>::
        Rectifier
        ():
        UnaryOperator<data_type>::UnaryOperator("Rectifier")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Rectifier<data_type>::
        Rectifier
        (Operator<data_type>* a):
        UnaryOperator<data_type>::UnaryOperator
        (a, "Rectifier_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Rectifier<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return rectifier(x->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        Rectifier<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return multiply(heaviside(this->get_value()), x->get_gradient());
        }

        template <class data_type>
        void
        Rectifier<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg = this->get_arg();
            if(arg->is_variable())
            {
                TensorCPU<data_type>* dy_dcurr = this->get_gradient();
                TensorCPU<data_type>* dcurr_darg = this->get_value();
                arg->set_gradient(multiply(dy_dcurr, heaviside(this->get_value())));
            }
        }

        template <class data_type>
        Rectifier<data_type>::
        ~Rectifier
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        MatMul<data_type>::_id = 0;

        template <class data_type>
        MatMul<data_type>::
        MatMul
        ():
        BinaryOperator<data_type>::BinaryOperator("MatMul")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        MatMul<data_type>::
        MatMul
        (Operator<data_type>* m, Operator<data_type>* n):
        BinaryOperator<data_type>::BinaryOperator
        (m, n, "MatMul_" + std::to_string(_id++))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        MatMul<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *m, *n;
            m = this->get_arg((bool)0);
            n = this->get_arg((bool)1);
            return matmul(m->get_value(), n->get_value());
        }

        template <class data_type>
        TensorCPU<data_type>*
        MatMul<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {
            check(false, std::string("MatMul::compute_gradient"));
        }

        template <class data_type>
        void
        MatMul<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg1 = this->get_arg((bool)0);
            Operator<data_type>* arg2 = this->get_arg((bool)1);
            if(arg1->is_variable())
            {
                TensorCPU<data_type>* dy_dcurr = this->get_gradient();
                TensorCPU<data_type>* n_val = arg2->get_value();
                arg1->set_gradient(matmul(dy_dcurr, n_val, false, true));
            }
            if(arg2->is_variable())
            {
                TensorCPU<data_type>* dy_dcurr = this->get_gradient();
                TensorCPU<data_type>* m_val = arg1->get_value();
                arg2->set_gradient(matmul(m_val, dy_dcurr, true, false));
            }
        }

        template <class data_type>
        MatMul<data_type>::
        ~MatMul
        ()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        unsigned long int
        Sum<data_type>::_id = 0;

        template <class data_type>
        Sum<data_type>::
        Sum
        ():
        UnaryOperator<data_type>::UnaryOperator("Sum")
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        Sum<data_type>::
        Sum
        (Operator<data_type>* a, unsigned int axis):
        UnaryOperator<data_type>::UnaryOperator
        (a, "Sum_" + std::to_string(_id++)),
        _axis(axis)
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Sum<data_type>::
        compute_value
        ()
        {
            Operator<data_type> *x;
            x = this->get_arg();
            return sum(x->get_value(), this->_axis);
        }

        template <class data_type>
        TensorCPU<data_type>*
        Sum<data_type>::
        compute_gradient
        (TensorCPU<data_type>* var)
        {

            Operator<data_type>* arg = this->get_arg();
            TensorCPU<data_type>* darg_dvar = arg->get_gradient();
            TensorCPU<data_type>* dy_darg = new TensorCPU<data_type>
            (darg_dvar->get_shape(), darg_dvar->get_ndims());
            bnn::core::copy(dy_darg, darg_dvar);
            arg->set_gradient(dy_darg);
        }

        template <class data_type>
        void
        Sum<data_type>::
        compute_gradient_reverse
        ()
        {
            Operator<data_type>* arg = this->get_arg();
            if(arg->is_variable())
            {
                TensorCPU<data_type>* arg_val = arg->get_value();
                TensorCPU<data_type>* dy_darg = new TensorCPU<data_type>
                (arg_val->get_shape(), arg_val->get_ndims());
                bnn::core::fill(dy_darg, (data_type) 1.0);
                arg->set_gradient(dy_darg);
            }
        }

        template <class data_type>
        Sum<data_type>::
        ~Sum
        ()
        {
            BNNMemory->invalidate(this);
        }

        #include "bnn/templates/operations/operators.hpp"

    }
}

#endif
