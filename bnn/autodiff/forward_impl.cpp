#ifndef BNN_BNN_AUTODIFF_FORWARD_IMPL_CPP
#define BNN_BNN_AUTODIFF_FORWARD_IMPL_CPP

#include <thread>
#include <cmath>
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
        using namespace bnn::autodiff;
        using namespace bnn::operators;

        template <class data_type>
        struct op_queue: public BNNBase
        {
            Operator<data_type>* op;

            op_queue<data_type>* next;

            op_queue():
            next(NULL),
            op(NULL)
            {
                BNNMemory->push(this);
            }

            static
            void
            clear(op_queue<data_type>* ptr)
            {
                op_queue<data_type>* curr = ptr;
                op_queue<data_type>* curr_next;
                while(curr != NULL)
                {
                    curr_next = curr->next;
                    BNNMemory->free_memory(curr);
                    curr = curr_next;
                }
            }
        };

        template <class data_type>
        void
        exec_compute_gradient_jobs
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
        (ForwardGraphNode<data_type>* layer, TensorCPU<data_type>* var)
        {
            while(layer != NULL)
            {
                if((layer->next != NULL) && layer->next->next != NULL)
                {
                    ForwardGraphNode<data_type>* last2last =  layer->next->next;
                    for(unsigned i = 0; i < last2last->len_ops; i++)
                    {
                        BNNMemory->free_memory(last2last->ops[i]->get_gradient());
                        if(last2last->ops[i]->num_args() != 0)
                        {
                                BNNMemory->free_memory(last2last->ops[i]->get_value());
                                last2last->ops[i]->set_gradient(NULL);
                        }
                    }
                }

                unsigned threads = layer->len_ops;
                thread* pool[threads];
                op_queue<data_type>* jobs[threads][2];

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

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(exec_compute_gradient_jobs<data_type>, jobs[i][1], var);
                    BNNThreads->push(pool[i]);
                }

                for(unsigned i = 0; i < threads; i++)
                {
                    BNNThreads->free_thread(pool[i]);
                    op_queue<data_type>::clear(jobs[i][1]);
                }

                layer = layer->prev;
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>* var)
        {
            ForwardGraphNode<data_type>* layer = build_graph_forward(expr);
            _compute_gradient_forward(layer, var);
            ForwardGraphNode<data_type>* curr = layer;
            while(curr != NULL)
            {
                ForwardGraphNode<data_type>* prev = curr->prev;
                BNNMemory->free_memory(curr);
                curr = prev;
            }

            return expr->get_gradient();
        }

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_forward
        (Operator<data_type>* expr, TensorCPU<data_type>** vars,
         unsigned num_vars)
        {
            TensorCPU<data_type>** grads = new TensorCPU<data_type>*[num_vars];
            ForwardGraphNode<data_type>* layer = build_graph_forward(expr);
            for(unsigned i = 0; i < num_vars; i++)
            {
                _compute_gradient_forward(layer, vars[i]);
                grads[i] = expr->get_gradient();
            }

            ForwardGraphNode<data_type>* curr = layer;
            while(curr != NULL)
            {
                ForwardGraphNode<data_type>* prev = curr->prev;
                BNNMemory->free_memory(curr);
                curr = prev;
            }

            return grads;
        }

        #include "bnn/templates/autodiff/forward.hpp"

    }
}

#endif
