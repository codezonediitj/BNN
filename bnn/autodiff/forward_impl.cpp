#ifndef BNN_BNN_AUTODIFF_FORWARD_IMPL_CPP
#define BNN_BNN_AUTODIFF_FORWARD_IMPL_CPP

#include <thread>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/autodiff/graph.hpp>
#include <bnn/utils/utils.hpp>


namespace bnn
{
    namespace autodiff
    {

        using namespace std;
        using namespace bnn::core;
        using namespace bnn::operators;

        template <class data_type>
        void
        _compute_gradient_forward_jobs
        (op_queue<data_type>* job_head, TensorCPU<data_type>* var)
        {
            op_queue<data_type>* job = job_head;
            while(job->op != NULL)
            {
                TensorCPU<data_type>* val = job->op->compute_value();
                job->op->set_value(val);
                TensorCPU<data_type>* grad = job->op->compute_gradient(var);
                job->op->set_gradient(grad);
                job = job->next;
            }
        }

        template <class data_type>
        void
        _compute_gradient_forward
        (GraphNode<data_type>* layer, TensorCPU<data_type>* var)
        {
            while(layer != NULL)
            {
                if((layer->next != NULL) && layer->next->next != NULL)
                {
                    GraphNode<data_type>* last2last =  layer->next->next;
                    for(unsigned i = 0; i < last2last->len_ops; i++)
                    {
                        BNNMemory->free_memory(last2last->ops[i]->get_gradient());
                        if(last2last->ops[i]->num_args() != 0)
                        {
                            BNNMemory->free_memory(last2last->ops[i]->get_value());
                        }
                    }
                }

                unsigned threads = layer->len_ops;
                thread* pool[threads];
                op_queue<data_type>* jobs[threads][2];

                _rr_scheduler<data_type>(layer, jobs, threads);

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(_compute_gradient_forward_jobs<data_type>, jobs[i][1], var);
                    BNNThreads->push(pool[i]);
                }

                _clear_jobs<data_type>(pool, jobs, threads);

                layer = layer->prev;
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>* var)
        {
            GraphNode<data_type>* layer = build_graph(expr);
            _compute_gradient_forward(layer, var);
            GraphNode<data_type>::clear_graph(layer);

            return expr->get_gradient();
        }

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>** vars,
         unsigned num_vars)
        {
            TensorCPU<data_type>** grads = new TensorCPU<data_type>*[num_vars];
            GraphNode<data_type>* layer = build_graph(expr);
            for(unsigned i = 0; i < num_vars; i++)
            {
                _compute_gradient_forward(layer, vars[i]);
                grads[i] = expr->get_gradient();
            }
            GraphNode<data_type>::clear_graph(layer);

            return grads;
        }

        #include "bnn/templates/autodiff/forward.hpp"

    }
}

#endif
