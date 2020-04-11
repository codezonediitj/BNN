#ifndef BNN_BNN_AUTODIFF_GRAPH_HPP
#define BNN_BNN_AUTODIFF_GRAPH_HPP

#include<bnn/operations/operators.hpp>

namespace bnn
{
    namespace autodiff
    {

        using namespace bnn::operators;

        template <class data_type>
        struct ForwardGraphNode
        {
            ForwardGraphNode<data_type> *prev;

            Operator<data_type>* ops;
        };

        template <class data_type>
        ForwardGraphNode<data_type>*
        build_graph_forward
        (Operator<data_type>* expr, Operator<data_type>* var=NULL);

    }
}

#endif
