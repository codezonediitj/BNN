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

        /*
        * For computing gradients using forward mode.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param expr Operator<data_type>* The expression
        *     to be considered for finding the derivative.
        * @param var TensorCPU<data_type>* The variable
        *     with respect to which the gradient is to
        *     be computed.
        */
        template <class data_type>
        TensorCPU<data_type>*
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>* var);

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>** var,
         unsigned num_vars);

    }
}

#endif
