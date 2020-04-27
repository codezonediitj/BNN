#ifndef BNN_BNN_AUTODIFF_REVERSE_HPP
#define BNN_BNN_AUTODIFF_REVERSE_HPP

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
        * Computes the gradient of a expression
        * w.r.t to given variables using reverse
        * mode automatic differentiation.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        * @param expr Operator<data_type>* The epxression
        *     whose derivative is to be computed.
        * @param vars TensorCPU<data_type>** The variables
        *     w.r.t which gradient is to be computed.
        * @param num_vars unsigned The number of variables.
        */
        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_reverse
        (Operator<data_type>* expr, TensorCPU<data_type>** vars,
         unsigned num_vars);

    }
}

#endif
