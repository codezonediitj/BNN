#ifndef BNN_BNN_AUTODIFF_REVERSE_IMPL_CPP
#define BNN_BNN_AUTODIFF_REVERSE_IMPL_CPP

#include <thread>
#include <bnn/core/tensor.hpp>
#include <bnn/core/tensor_ops.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/autodiff/graph.hpp>
#include <bnn/utils/utils.hpp>
#include <unordered_map>
#include <vector>
#include <string>

namespace bnn
{
    namespace autodiff
    {

        using namespace std;
        using namespace bnn::core;
        using namespace bnn::operators;

        template <class data_type>
        void
        _compute_value_jobs
        (op_queue<data_type>* job_head)
        {
            op_queue<data_type>* job = job_head;
            while(job->op != NULL)
            {
                if(job->op->num_args() == 1)
                {
                    job->op->set_variable
                    (job->op->get_arg()->is_variable());
                }
                else if(job->op->num_args() == 2)
                {
                    job->op->set_variable
                    (job->op->get_arg(0)->is_variable() ||
                        job->op->get_arg(1)->is_variable());
                }
                TensorCPU<data_type>* val = job->op->compute_value();
                job->op->set_value(val);
                job = job->next;
            }
        }

        template <class data_type>
        GraphNode<data_type>*
        _compute_value
        (GraphNode<data_type>* layer,
         unordered_map<TensorCPU<data_type>*, unsigned>& var_map)
        {
            GraphNode<data_type>* top;
            while(layer != NULL)
            {
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
                    if(var_map.find(layer->ops[i]->get_value()) != var_map.end())
                    {
                        layer->ops[i]->set_variable(true);
                    }
                    else
                    {
                        layer->ops[i]->set_variable(false);
                    }
                    op_queue<data_type>* task = new op_queue<data_type>;
                    jobs[j][0]->next = task;
                    jobs[j][0] = task;
                }

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(_compute_value_jobs<data_type>, jobs[i][1]);
                    BNNThreads->push(pool[i]);
                }

                _clear_jobs<data_type>(pool, jobs, threads);

                layer = layer->prev;
                if(layer != NULL && layer->prev == NULL)
                {
                    top = layer;
                }
            }

            return top;
        }

        template <class data_type>
        void
        _add_grad
        (Operator<data_type>* x, Operator<data_type>* y, unsigned size)
        {
            TensorCPU<data_type>* grad;
            grad = bnn::core::add(x->get_gradient(), y->get_gradient());
            BNNMemory->free_memory(x->get_gradient());
            BNNMemory->free_memory(y->get_gradient());
            x->set_gradient(grad);
        }

        template <class data_type>
        void
        _final_grad_job
        (Operator<data_type>** tws, unsigned size,
         TensorCPU<data_type>** grads, unsigned idx)
        {
            unsigned nthreads = size/2;
            thread* pool[nthreads];
            unsigned gap = 1;
            while(gap < size)
            {
                for(unsigned i = 0; i < size; i += 2*gap)
                {
                    if(i + gap < size)
                    {
                        pool[i] = new thread(_add_grad<data_type>, tws[i], tws[i+gap], size);
                        BNNThreads->push(pool[i]);
                    }
                }
                for(unsigned i = 0; i < size; i += 2*gap)
                {
                    if(i + gap < size)
                    {
                        BNNThreads->free_thread(pool[i]);
                    }
                }
                gap *= 2;
            }

            grads[idx] = tws[0]->get_gradient();
        }

        template <class data_type>
        void
        _compute_gradient_reverse_jobs
        (op_queue<data_type>* job_head)
        {
            op_queue<data_type>* job = job_head;
            while(job->op != NULL)
            {
                job->op->compute_gradient_reverse();
                job = job->next;
            }
        }

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_reverse
        (Operator<data_type>* expr, TensorCPU<data_type>** vars,
         unsigned num_vars)
        {
            string msg = "Aborting gradient computation due to absence of variables.";
            bnn::utils::check(num_vars != 0, msg);
            GraphNode<data_type>* layer = build_graph(expr);
            GraphNode<data_type>* _layer = layer;
            unordered_map<TensorCPU<data_type>*, unsigned> var_map;
            unordered_map<TensorCPU<data_type>*, vector<Operator<data_type>*>> t2tw;
            for(unsigned i = 0; i < num_vars; i++)
            {
                var_map[vars[i]] = i;
            }
            layer = _compute_value<data_type>(layer, var_map);

            TensorCPU<data_type>* exprgrad = new TensorCPU<data_type>
            (expr->get_value()->get_shape(), expr->get_value()->get_ndims());
            bnn::core::fill(exprgrad, (data_type)1.);
            expr->set_gradient(exprgrad);
            while(layer != NULL)
            {
                unsigned threads = 0;
                for(unsigned i = 0; i < layer->len_ops; i++)
                {
                    if(layer->ops[i]->is_variable())
                    {
                        threads++;
                    }
                }
                thread* pool[threads];
                op_queue<data_type>* jobs[threads][2];

                for(unsigned i = 0; i < threads; i++)
                {
                    jobs[i][0] = new op_queue<data_type>;
                    jobs[i][1] = jobs[i][0];
                }

                unsigned var_idx = 0;
                for(unsigned i = 0; i < layer->len_ops; i++)
                {
                    if(layer->ops[i]->is_variable())
                    {
                        unsigned j = var_idx%layer->len_ops;
                        jobs[j][0]->op = layer->ops[i];
                        if(var_map.find(layer->ops[i]->get_value()) != var_map.end())
                        {
                            t2tw[layer->ops[i]->get_value()].push_back(layer->ops[i]);
                        }
                        op_queue<data_type>* task = new op_queue<data_type>;
                        jobs[j][0]->next = task;
                        jobs[j][0] = task;
                        var_idx += 1;
                    }
                }

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(_compute_gradient_reverse_jobs<data_type>, jobs[i][1]);
                    BNNThreads->push(pool[i]);
                }

                _clear_jobs<data_type>(pool, jobs, threads);

                if(layer != _layer && layer->next != NULL)
                {
                    for(unsigned i = 0; i < layer->len_ops; i++)
                    {
                        BNNMemory->free_memory(layer->ops[i]->get_gradient());
                        BNNMemory->free_memory(layer->ops[i]->get_value());
                    }
                }

                layer = layer->next;
            }

            TensorCPU<data_type>** grads = new TensorCPU<data_type>*[num_vars];
            GraphNode<data_type>::clear_graph(_layer);

            thread* _pool[t2tw.size()];
            Operator<data_type>** arr[t2tw.size()];

            unsigned i = 0;
            for(auto& it: t2tw)
            {
                arr[i] = new Operator<data_type>*[it.second.size()];
                copy(it.second.begin(), it.second.end(), arr[i]);
                _pool[i] = new thread(_final_grad_job<data_type>,
                                    arr[i], it.second.size(), grads, var_map[it.first]);
                BNNThreads->push(_pool[i]);
                i++;
            }
            for(i = 0; i < t2tw.size(); i++)
            {
                BNNThreads->free_thread(_pool[i]);
                delete arr[i];
            }

            return grads;
        }

        #include "bnn/templates/autodiff/reverse.hpp"

    }
}

#endif
