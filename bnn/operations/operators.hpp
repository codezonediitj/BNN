#ifndef BNN_BNN_OPERATIONS_OPERATORS_HPP
#define BNN_BNN_OPERATIONS_OPERATORS_HPP

#include<bnn/core/tensor.hpp>
#include<string>

namespace bnn
{
    namespace operators
    {

        using namespace bnn::core;

        /*
        * This class represents generic
        * Operator class.
        */
        template <class data_type>
        class Operator
        {
            protected:

                //! Identifies operator in an expression.
                std::string name;

            public:

                data_type value;

                data_type gradient;

                Operator<data_type>* variable;

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                Operator
                (std::string _name);

                /*
                * For obtaining name of the operator.
                */
                std::string
                get_name
                ();

                /*
                * For checking if the current operator
                * wraps a tensor. The bool value returned
                * acts as a signal of reaching the leaf
                * in the expression tree.
                */
                virtual
                bool
                is_tensor
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

                data_type
                compute_gradient
                ();

                data_type
                compute_value
                ();

                data_type
                get_value
                ();

                data_type
                get_gradient
                ();

                void
                set_value
                (data_type value);

                void
                set_gradient
                (data_type gradient);

                Operator<data_type>*
                get_variable
                ();

                void
                set_variable
                (Operator<data_type>* _var);
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

                virtual
                Operator<data_type>*
                get_arg
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

                virtual Operator<data_type>*
                get_arg
                (bool idx);
        };

        template <class data_type>
        class TensorWrapper: public Operator<data_type>
        {
            protected:

                //! Identity of a TensorWrapper object.
                //! Used in name.
                static unsigned long _id;

                //! Pointer to the TensorCPU object.
                TensorCPU<data_type>* t;

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
                (TensorCPU<data_type>& _t);

                /*
                * Reads pointer to the TensorCPU
                * object wrapped by TensorWrapper.
                */
                TensorCPU<data_type>*
                get_tensor
                ();

                virtual
                bool
                is_tensor
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
        };

    }
}

#endif
