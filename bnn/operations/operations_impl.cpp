#ifndef BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP
#define BNN_BNN_OPERATIONS_OPERATIONS_IMPL_CPP

#include<bnn/operations/operators.hpp>
#include<bnn/operations/operations.hpp>

namespace bnn
{
    namespace operations
    {
        template <class data_type>
        bnn::operators::Add*
        add(const bnn::core::TensorCPU<data_type>& a,
            const bnn::core::TensorCPU<data_type>& b)
        {
            using namespace bnn::operators;
            Identity* ia = new Identity<data_type>(a);
            Identity* ib = new Identity<data_type>(b);
            return add(ia, ib);
        }

        bnn::operators::Add*
        add(bnn::operators::Operator* a,
            bnn::operators::Operator* b)
        {
            using namespace bnn::operators;
            Add* result = new Add(a, b);
            return result;
        }
    }
}

#endif
