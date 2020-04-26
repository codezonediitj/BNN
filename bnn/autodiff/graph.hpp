#ifndef BNN_BNN_AUTODIFF_GRAPH_HPP
#define BNN_BNN_AUTODIFF_GRAPH_HPP

#include <bnn/operations/operators.hpp>
#include <bnn/utils/utils.hpp>
#include <thread>

namespace bnn
{
    namespace autodiff
    {

        using namespace std;
        using namespace bnn::operators;

        /*
        * Represents a node in graph built
        * for forward mode automatic
        * differentiation.
        */
        template <class data_type>
        struct GraphNode: public BNNBase
        {
            //! The node previous to the current node.
            GraphNode<data_type>* prev;

            //! The node next to the current node.
            GraphNode<data_type>* next;

            //! An array of Operator<data_type>* elements.
            Operator<data_type>** ops;

            //! The length of ops array.
            unsigned len_ops;

            static
            void
            clear_graph
            (GraphNode<data_type>* layer);

            //! Destructor
            virtual
            ~GraphNode
            ();

            //! Constructor
            GraphNode
            ();
        };

        template <class data_type>
        struct op_queue: public BNNBase
        {
            Operator<data_type>* op;

            op_queue<data_type>* next;

            op_queue
            ();

            static
            void
            clear
            (op_queue<data_type>* ptr);
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
        GraphNode<data_type>*
        build_graph
        (Operator<data_type>* expr);

        template <class data_type>
        void
        _rr_scheduler
        (GraphNode<data_type>* layer, op_queue<data_type>* jobs[][2],
         unsigned threads);

        template <class data_type>
        void
        _clear_jobs
        (thread* pool[], op_queue<data_type>* jobs[][2],
         unsigned threads);

    }
}

#endif
