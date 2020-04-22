#ifndef BNN_BNN_AUTODIFF_GRAPH_HPP
#define BNN_BNN_AUTODIFF_GRAPH_HPP

#include <bnn/operations/operators.hpp>

namespace bnn
{
    namespace autodiff
    {

        using namespace bnn::operators;

        template <class data_type>
        struct ForwardGraphNode
        {
            ForwardGraphNode<data_type>* prev;

            ForwardGraphNode<data_type>* next;

            Operator<data_type>** ops;

            unsigned len_ops;

            ~ForwardGraphNode();
        };

        template <class data_type>
        ForwardGraphNode<data_type>*
        build_graph_forward
        (Operator<data_type>* expr);

    }
}

#endif
