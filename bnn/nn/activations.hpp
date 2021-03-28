#ifndef BNN_BNN_NN_ACTIVATIONS_HPP
#define BNN_BNN_NN_ACTIVATIONS_HPP

#include <bnn/utils/utils.hpp>
#include <bnn/operations/operators.hpp>

namespace bnn
{
    namespace nn
    {

        using namespace bnn::utils;
        using namespace bnn::operators;

        template <class data_type>
        class Activation: public BNNBase
        {

            protected:

                std::string name;

                Operator<data_type>* input;

                Operator<data_type>* output;

            public:

                Activation
                (std::string _name);

                virtual
                void
                implementation
                ();

                virtual
                TensorCPU<data_type>*
                get_input_value
                ();

                virtual
                TensorCPU<data_type>*
                get_output_value
                ();

                virtual
                ~Activation
                ();

        };

    }
}

#endif
