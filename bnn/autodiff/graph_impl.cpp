#ifndef BNN_BNN_AUTODIFF_GRAPH_IMPL_CPP
#define BNN_BNN_AUTODIFF_GRAPH_IMPL_CPP

#include <bnn/autodiff/graph.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace autodiff
    {

        using namespace bnn::operators;

        template <class data_type>
        GraphNode<data_type>::
        ~GraphNode()
        {
            BNNMemory->invalidate(this);
        }

        template <class data_type>
        GraphNode<data_type>::
        GraphNode()
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        void
        GraphNode<data_type>::
        clear_graph
        (GraphNode<data_type>* layer)
        {
            GraphNode<data_type>* curr = layer;
            while(curr != NULL)
            {
                GraphNode<data_type>* prev = curr->prev;
                BNNMemory->free_memory(curr);
                curr = prev;
            }
        }

        template <class data_type>
        inline
        unsigned
        _sum
        (Operator<data_type>** _ops, unsigned len)
        {
            unsigned i = 0, total_args = 0;
            while(i < len)
            {
                total_args += _ops[i]->num_args();
                i += 1;
            }

            return total_args;
        }

        template <class data_type>
        GraphNode<data_type>*
        build_graph
        (Operator<data_type>* expr)
        {
            GraphNode<data_type>* layer =
            new GraphNode<data_type>;
            layer->prev = NULL;
            layer->next = NULL;
            layer->ops = new Operator<data_type>*[1];
            layer->len_ops = 1;
            layer->ops[0] = expr;

            unsigned total_args = _sum<data_type>(layer->ops, layer->len_ops);
            while(total_args > 0)
            {
                GraphNode<data_type>* next_layer =
                new GraphNode<data_type>;
                layer->next = next_layer;
                next_layer->prev = layer;
                next_layer->next = NULL;
                next_layer->ops = new Operator<data_type>*[total_args];
                next_layer->len_ops = total_args;

                Operator<data_type>* op;
                unsigned i = 0, j = 0;
                while(layer->ops[i] != NULL)
                {
                    op = layer->ops[i];
                    if(op->num_args() == 1)
                    {
                        next_layer->ops[j] = op->get_arg();
                        j += 1;
                    }
                    else if(op->num_args() == 2)
                    {
                        next_layer->ops[j] = op->get_arg(0);
                        next_layer->ops[j+1] = op->get_arg(1);
                        j += 2;
                    }

                    i += 1;
                }

                layer = next_layer;
                total_args = _sum<data_type>(layer->ops, layer->len_ops);
            }

            return layer;
        }

        template <class data_type>
        void
        _rr_scheduler
        (GraphNode<data_type>* layer, op_queue<data_type>* jobs[][2],
         unsigned threads)
        {
            for(unsigned i = 0; i < threads; i++)
            {
                jobs[i][0] = new op_queue<data_type>;
                jobs[i][1] = jobs[i][0];
            }

            for(unsigned i = 0; i < layer->len_ops; i++)
            {
                unsigned j = i%layer->len_ops;
                jobs[j][0]->op = layer->ops[i];
                op_queue<data_type>* task = new op_queue<data_type>;
                jobs[j][0]->next = task;
                jobs[j][0] = task;
            }
        }

        template <class data_type>
        void
        _clear_jobs
        (thread* pool, op_queue<data_type>* jobs[][2],
         unsigned threads)
        {
            for(unsigned i = 0; i < threads; i++)
            {
                BNNThreads->free_thread(pool[i]);
                op_queue<data_type>::clear(jobs[i][1]);
            }
        }


        #include "bnn/templates/autodiff/graph.hpp"

    }
}

#endif
