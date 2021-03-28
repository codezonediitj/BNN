#ifndef BNN_BNN_OPERATIONS_OPERATORS_HPP
#define BNN_BNN_OPERATIONS_OPERATORS_HPP

#include <string>
#include <bnn/core/tensor.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace operators
    {

        using namespace std;
        using namespace bnn::core;
        using namespace bnn::utils;

        /*
        * This class represents generic
        * Operator class. The documentation
        * of this class applies to all the sub
        * classes.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        */
        template <class data_type>
        class Operator: public BNNBase
        {
            protected:

                //! Identifies operator in an expression.
                std::string name;

            public:

                //! The value of the sub expression
                //! represented by this object. Used
                //! only in automatic differentiation.
                TensorCPU<data_type>* value;

                //! The gradient of the sub expression
                //! represented by this object. Used
                //! only in automatic differentiation.
                TensorCPU<data_type>* gradient;

                bool variable;

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                Operator
                (string _name);

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                Operator
                (TensorCPU<data_type>* value, string _name);

                /*
                * For obtaining name of the operator.
                */
                string
                get_name
                ();

                /*
                * Reads the argument from a UnaryOperator.
                */
                virtual
                Operator<data_type>*
                get_arg
                ();

                /*
                * Reads the argument from a BinaryOperator.
                *
                * @param idx bool to identify which argument
                *    to return, if true/1 then second argument
                *    is returned else first argument is returned.
                */
                virtual
                Operator<data_type>*
                get_arg
                (bool idx);

                /*
                * Computes the gradient of the sub-expression
                * w.r.t the given variable.
                *
                * @param var Tensor<data_type>* The variable.
                */
                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                /*
                * Computes gradient of operator's
                * variable arguments.
                */
                virtual
                void
                compute_gradient_reverse
                ();

                /*
                * Computes the value of the sub-expression
                */
                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                //! Getter method for the member, value.
                TensorCPU<data_type>*
                get_value
                ();

                //! Getter method for the member, gradient.
                TensorCPU<data_type>*
                get_gradient
                ();

                //! Setter method for the member, value.
                void
                set_value
                (TensorCPU<data_type>* value);

                //! Setter method for the member, gradient.
                void
                set_gradient
                (TensorCPU<data_type>* _gradient);

                //! Returns the number of arguments in a operator.
                virtual
                unsigned
                num_args
                ();

                virtual
                bool
                is_variable
                ();

                virtual
                void
                set_variable
                (bool _val);

                //! Destructor
                virtual
                ~Operator
                ();
        };

        template <class data_type>
        class UnaryOperator: public Operator<data_type>
        {
            protected:

                //! The only argument of the operator.
                Operator<data_type>* x;

            public:

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                UnaryOperator
                (std::string _name);

                /*
                * Parametrized constructor.
                *
                * @param Operator* The only argument
                *    to the UnaryOperator.
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                UnaryOperator
                (Operator<data_type>* a, std::string _name);

                //! Getter method for the member, x.
                virtual
                Operator<data_type>*
                get_arg
                ();

                virtual
                unsigned
                num_args();

                virtual
                ~UnaryOperator
                ();
        };

        template <class data_type>
        class BinaryOperator: public Operator<data_type>
        {
            protected:

                //! The first argument of the operator.
                Operator<data_type>* x;

                //! The second argument of the operator.
                Operator<data_type>* y;

            public:

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                BinaryOperator
                (std::string _name);

                /*
                * Parametrized constructor.
                *
                * @param Operator* The first argument
                *    to the BinaryOperator.
                * @param Operator* The second argument
                *    to the BinaryOperator.
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                BinaryOperator
                (Operator<data_type>* a, Operator<data_type>* b,
                 std::string _name);

                /*
                * Getter method for the members, x and y.
                *
                * @param idx bool Returns the member x
                *     if True else returns y.
                */
                virtual
                Operator<data_type>*
                get_arg
                (bool idx);

                virtual
                unsigned
                num_args();

                virtual
                ~BinaryOperator
                ();
        };

        template <class data_type>
        class TensorWrapper: public Operator<data_type>
        {
            protected:

                //! Identity of a TensorWrapper object.
                //! Used in name.
                static unsigned long _id;

            public:

                /*
                * Default constructor.
                */
                TensorWrapper
                ();

                /*
                * Parametrized constructor.
                *
                * @param _t TensorCPU object to be
                *    referred.
                */
                TensorWrapper
                (TensorCPU<data_type>* _t);

                /*
                * Reads pointer to the TensorCPU
                * object wrapped by TensorWrapper.
                */
                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                unsigned
                num_args
                ();

                virtual
                ~TensorWrapper
                ();
        };

        template <class data_type>
        class Add: public BinaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Add
                ();

                Add
                (Operator<data_type>* a, Operator<data_type>* b);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Add
                ();
        };

        template <class data_type>
        class Multiply: public BinaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Multiply
                ();

                Multiply
                (Operator<data_type>* a, Operator<data_type>* b);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Multiply
                ();
        };

        template <class data_type>
        class Divide: public BinaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Divide
                ();

                Divide
                (Operator<data_type>* a, Operator<data_type>* b);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Divide
                ();
        };

        template <class data_type>
        class Exp: public UnaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Exp
                ();

                Exp
                (Operator<data_type>* a);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Exp
                ();
        };

        template <class data_type>
        class Log: public UnaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Log
                ();

                Log
                (Operator<data_type>* a);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Log
                ();
        };

        template <class data_type>
        class Rectifier: public UnaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                Rectifier
                ();

                Rectifier
                (Operator<data_type>* a);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Rectifier
                ();
        };

        template <class data_type>
        class MatMul: public BinaryOperator<data_type>
        {
            protected:

                static unsigned long _id;

            public:

                MatMul
                ();

                MatMul
                (Operator<data_type>* m, Operator<data_type>* n);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~MatMul
                ();
        };

        template <class data_type>
        class Sum: public UnaryOperator<data_type>
        {
            protected:

                static unsigned long _id;
                unsigned int _axis;

            public:

                Sum
                ();

                Sum
                (Operator<data_type>* a, unsigned int axis=0);

                virtual
                TensorCPU<data_type>*
                compute_gradient
                (TensorCPU<data_type>* var);

                virtual
                void
                compute_gradient_reverse
                ();

                virtual
                TensorCPU<data_type>*
                compute_value
                ();

                virtual
                ~Sum
                ();
        };

    }
}

#endif
