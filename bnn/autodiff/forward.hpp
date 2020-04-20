#ifndef BNN_BNN_AUTODIFF_FORWARD_HPP
#define BNN_BNN_AUTODIFF_FORWARD_HPP

#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>
#include <thread>

namespace bnn
{
    namespace autodiff
    {

        using namespace std;
        using namespace bnn::core;
        using namespace bnn::operators;

        template <class data_type>
        TensorCPU<data_type>*
        compute_gradient
        (Operator<data_type>* expr, TensorCPU<data_type>* var);

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient
        (Operator<data_type>* expr, TensorCPU<data_type>** var);

    }
}

#endif
