#ifndef BNN_BNN_AUTODIFF_GRAPH_IMPL_CPP
#define BNN_BNN_AUTODIFF_GRAPH_IMPL_CPP

#include<bnn/autodiff/graph.hpp>

namespace bnn
{
    namespace autodiff
    {

        using namespace bnn::operators;

        template <class data_type>
        struct ForwardGraphNode
        {
            ForwardGraphNode<data_type> *prev;

            Operator<data_type>** ops;

            ~ForwardGraphNode()
            {
                unsigned i = 0;
                while(this->ops[i] != NULL)
                {
                    delete ops[i];
                    i += 1;
                }
            }
        };

        template <class data_type>
        inline
        unsigned
        _sum
        (Operator<data_type>** _ops)
        {
            unsigned i = 0, total_args = 0;
            while(_ops[i] != NULL)
            {
                total_args += _ops[i]->num_args();
                i++;
            }

            return total_args;
        }

        template <class data_type>
        ForwardGraphNode<data_type>*
        build_graph_forward
        (Operator<data_type>* expr, Operator<data_type>* var=NULL)
        {
            ForwardGraphNode<data_type>* layer =
            new ForwardGraphNode<data_type>;
            layer->prev = NULL;
            layer->ops = new Operator<data_type>*[2];
            layer->ops[0] = expr, layer->ops[1] = NULL;
            layer->ops[0]->set_variable(var);

            unsigned total_args = _sum<data_type>(layer->ops);
            while(total_args > 0)
            {
                ForwardGraphNode<data_type>* next_layer =
                new ForwardGraphNode<data_type>;
                next_layer->prev = layer;
                next_layer->ops = new Operator*[total_args+1];
                next_layer->ops[total_args] = NULL;

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
                total_args = _sum<data_type>(layer->ops);
            }

            return layer;
        }

    }
}

#endif
