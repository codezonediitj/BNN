#ifndef BNN_BNN_IO_INPUT_HPP
#define BNN_BNN_IO_INPUT_HPP

#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace io
    {

        using namespace bnn::core;

        /*
        * Loads MNIST image file into bnn::core::TensorCPU.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param path const string& Path to the MNIST image
        *     file.
        */
        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_images
        (const string& path);

        /*
        * Loads MNIST label file into bnn::core::TensorCPU.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param path const string& Path to the MNIST label
        *     file.
        */
        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_labels
        (const string& path);

    }
}

#endif
