#ifndef BNN_BNN_AUTODIFF_GRAPH_HPP
#define BNN_BNN_AUTODIFF_GRAPH_HPP

#include <bnn/operations/operators.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace autodiff
    {

        using namespace bnn::operators;

        /*
        * Represents a node in graph built
        * for forward mode automatic
        * differentiation.
        */
        template <class data_type>
        struct ForwardGraphNode: public BNNBase
        {
            //! The node previous to the current node.
            ForwardGraphNode<data_type>* prev;

            //! The node next to the current node.
            ForwardGraphNode<data_type>* next;

            //! An array of Operator<data_type>* elements.
            Operator<data_type>** ops;

            //! The length of ops array.
            unsigned len_ops;

            //! Destructor
            virtual
            ~ForwardGraphNode
            ();

            //! Constructor
            ForwardGraphNode
            ();
        };

        /*
        * Build the graph for forward
        * mode automatic differentiation.
        *
        * @param expr Operator<data_type>*
        *     The expression for which
        *     the graph is to be built.
        */
        template <class data_type>
        ForwardGraphNode<data_type>*
        build_graph_forward
        (Operator<data_type>* expr);

    }
}

#endif
