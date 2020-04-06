#ifndef BNN_BNN_OPERATIONS_OPERATORS_HPP
#define BNN_BNN_OPERATIONS_OPERATORS_HPP

#include<bnn/core/tensor.hpp>
#include<string>

namespace bnn
{
    namespace operators
    {
        /*
        * This class represents generic
        * Operator class.
        */
        class Operator
        {
            protected:

                //! Identifies operator in an expression.
                std::string name;

            public:

                /*
                * Parametrized constructor.
                *
                * @param _name std::string which is to be
                *    used to identify the operator.
                */
                Operator(std::string _name);

                /*
                * For obtaining name of the operator.
                */
                std::string get_name();

                /*
                * For checking if the current operator
                * wraps a tensor. The bool value returned
                * acts as a signal of reaching the leaf
                * in the expression tree.
                */
                virtual bool is_tensor();

                /*
                * Reads the argument from a UnaryOperator.
                */
                virtual Operator*
                get_arg();

                /*
                * Reads the argument from a BinaryOperator.
                *
                * @param idx bool to identify which argument
                *    to return, if true/1 then second argument
                *    is returned else first argument is returned.
                */
                virtual Operator*
                get_arg(bool idx);

        };

        class UnaryOperator: public Operator
        {
            protected:

                //! The only argument of the operator.
                Operator* x;

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
                (Operator* a,
                 std::string _name);

                virtual Operator*
                get_arg();

        };

        class BinaryOperator: public Operator
        {
            protected:

                //! The first argument of the operator.
                Operator* x;

                //! The second argument of the operator.
                Operator* y;

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
                (Operator* a,
                 Operator* b,
                 std::string _name);

                virtual Operator*
                get_arg(bool idx);
        };

        template <class data_type>
        class TensorWrapper: public Operator
        {
            protected:

                //! Identity of a TensorWrapper object.
                //! Used in name.
                static unsigned long _id;

                //! Pointer to the TensorCPU object.
                bnn::core::TensorCPU<data_type>* t;

            public:

                /*
                * Default constructor.
                */
                TensorWrapper();

                /*
                * Parametrized constructor.
                *
                * @param _t TensorCPU object to be
                *    referred.
                */
                TensorWrapper
                (bnn::core::TensorCPU<data_type>& _t);

                /*
                * Reads pointer to the TensorCPU
                * object wrapped by TensorWrapper.
                */
                bnn::core::TensorCPU<data_type>*
                get_tensor();

                virtual bool is_tensor();

        };

        class Add: public BinaryOperator
        {
            protected:

                static unsigned long _id;

            public:

                Add();

                Add
                (Operator* a,
                 Operator* b);

        };

        class Exp: public UnaryOperator
        {
            protected:

                static unsigned long _id;

            public:

                Exp();

                Exp(Operator* a);

        };
    }
}

#endif
