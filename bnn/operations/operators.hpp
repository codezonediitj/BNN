#ifndef BNN_BNN_OPERATIONS_OPERATORS_HPP
#define BNN_BNN_OPERATIONS_OPERATORS_HPP

#include<bnn/core/tensor.hpp>
#include<string>

namespace bnn
{
    namespace operators
    {
        class Operator
        {
            protected:

                std::string name;

            public:

                Operator(std::string _name);

                std::string get_name();

                virtual bool is_tensor();

                virtual Operator*
                get_arg();

                virtual Operator*
                get_arg(bool idx);

        };

        class UnaryOperator: public Operator
        {
            protected:

                Operator* x;

            public:

                UnaryOperator
                (std::string _name);

                UnaryOperator
                (Operator* a,
                 std::string _name);

                virtual Operator*
                get_arg();

        };

        class BinaryOperator: public Operator
        {
            protected:

                Operator* x;
                Operator* y;

            public:

                BinaryOperator
                (std::string _name);

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

                static unsigned long _id;

                bnn::core::TensorCPU<data_type>* t;

            public:

                TensorWrapper();

                TensorWrapper
                (bnn::core::TensorCPU<data_type>& _t);

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
    }
}

#endif
