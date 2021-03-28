#ifndef BNN_BNN_NN_LAYERS_HPP
#define BNN_BNN_NN_LAYERS_HPP

#include <bnn/utils/utils.hpp>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>
#include <string>

namespace bnn
{
    namespace nn
    {

        using namespace bnn::utils;
        using namespace bnn::core;
        using namespace bnn::operators;

        template <class data_type>
        class Layer: public BNNBase
        {

            protected:

                std::string name;

                Operator<data_type>* input;

                Operator<data_type>* output;

            public:

                Layer
                (std::string _name);

                virtual
                void
                initialize_parameters
                (void (*initializer)(Operator<data_type>* parameters));

                virtual
                void
                implementation
                ();

                virtual
                Operator<data_type>*
                get_parameters
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
                ~Layer
                ();

        };

    }
}

#endif
