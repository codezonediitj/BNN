#ifndef BNN_BNN_AUTODIFF_REVERSE_IMPL_CPP
#define BNN_BNN_AUTODIFF_REVERSE_IMPL_CPP

#include <thread>
#include <bnn/core/tensor.hpp>
#include <bnn/operations/operators.hpp>
#include <bnn/autodiff/graph.hpp>
#include <bnn/utils/utils.hpp>
#include <unordered_map>

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
                TensorCPU<data_type>* val = job->op->compute_value();
                job->op->set_value(val);
                job = job->next;
            }
        }

        template <class data_type>
        void
        _compute_value
        (GraphNode<data_type>* layer)
        {
            while(layer != NULL)
            {
                unsigned threads = layer->len_ops;
                thread* pool[threads];
                op_queue<data_type>* jobs[threads][2];

                _rr_scheduler<data_type>(layer, jobs, threads);

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(_compute_value_jobs<data_type>, jobs[i][1]);
                    BNNThreads->push(pool[i]);
                }

                _clear_jobs<data_type>(pool, jobs, threads);

                layer = layer->prev;
            }
        }

        template <class data_type>
        void
        _compute_gradient_reverse_jobs
        (op_queue<data_type>* job_head,
         unordered_map<data_type>& var_map)
        {
            op_queue<data_type>* job = job_head;
            while(job->op != NULL)
            {
                job->op->compute_gradient_reverse(var_map);
                job = job->next;
            }
        }

        template <class data_type>
        TensorCPU<data_type>**
        compute_gradient_reverse
        (Operator<data_type>* expr, TensorCPU<data_type>** vars,
         unsigned num_vars)
        {
            GraphNode<data_type>* layer = build_graph(expr);
            GraphNode<data_type>* _layer = layer;
            _compute_value<data_type>(layer);
            unordered_map<TensorCPU<data_type>*, unsigned> var_map;
            unordered_map<TensorCPU<data_type>*, TensorWrapper<data_type>*> t2tw;
            for(unsigned i = 0; i < num_vars; i++)
            {
                var_map[vars[i]] = i;
            }

            TensorCPU<data_type>* exprgrad = new TensorCPU<data_type>;
            bnn::core::fill(exprgrad, (data_type)1.);
            expr->set_gradient(exprgrad);
            layer = layer->next;
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
                    if(var_map.contains(layer->ops[i]))
                    {
                        t2tw[layer->ops[i]] = layer->ops[i]->get_value();
                    }
                    op_queue<data_type>* task = new op_queue<data_type>;
                    jobs[j][0]->next = task;
                    jobs[j][0] = task;
                }

                for(unsigned i = 0; i < threads; i++)
                {
                    pool[i] = new thread(_compute_gradient_reverse_jobs<data_type>,
                                         jobs[i][1], var_map);
                    BNNThreads->push(pool[i]);
                }

                _clear_jobs<data_type>(pool, jobs, threads);

                if(layer != _layer)
                {
                    for(unsigned i = 0; i < layer->len_ops; i++)
                    {
                        BNNMemory->free_memory(layer->ops[i]->get_gradient());
                        BNNMemory->free_memory(layer->ops[i]->get_value());
                    }
                }

                layer = layer->prev;
            }

            TensorCPU<data_type>** grads = new TensorCPU<data_type>*[num_vars];
            TensorCPU<data_type>* null_ptr = NULL;
            std::fill(grads, grads + num_vars, null_ptr);
            GraphNode<data_type>::clear_graph(_layer);

            for(auto& it: t2tw)
            {
                unsigned idx = var_map[it.first];
                if(grads[idx] == NULL)
                {
                    grads[idx] = it.second->get_gradient();
                }
                else
                {
                    TensorCPU<float>* new_grad =
                    bnn::core::add(grads[idx],  it.second->get_gradient());
                    BNNMemory->free_memory(grads[idx]);
                    grads[idx] = new_grad;
                }
            }

            return grads;
        }

    }
}

#endif
