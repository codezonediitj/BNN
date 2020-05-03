#ifndef BNN_BNN_IO_INPUT_HPP
#define BNN_BNN_IO_INPUT_HPP

#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace io
    {

        using namespace bnn::core;

        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_images
        (const string& path);

        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_labels
        (const string& path);

    }
}

#endif
