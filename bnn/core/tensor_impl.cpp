#ifndef BNN_BNN_CORE_TENSOR_CPP
#define BNN_BNN_CORE_TENSOR_CPP

#include<bnn/core/tensor.hpp>
#include<bnn/utils/utils.hpp>

namespace bnn
{
    namespace core
    {
        template <class data_type>
        data_type*
        TensorCPU<data_type>::_reserve_space_cpu
        (std::vector<unsigned>& shape)
        {
            unsigned total_space = 1;
            for(unsigned i = 0; i < shape.size(); i++)
            {
                total_space *= shape.at(i);
            }
            data_type* pointer = new data_type[total_space];
            return pointer;
        }

        template <class data_type>
        unsigned*
        TensorCPU<data_type>::_init_shape_cpu
        (std::vector<unsigned>& shape)
        {
            unsigned* _shape = new unsigned[shape.size()];
            for(unsigned i = 0; i < shape.size(); i++)
            {
                _shape[i] = shape.at(i);
            }
            return _shape;
        }

        template <class data_type>
        TensorCPU<data_type>::TensorCPU():
        data_cpu(NULL),
        ndims_cpu(0),
        shape_cpu(NULL)
        {
        }

        template <class data_type>
        TensorCPU<data_type>::TensorCPU
        (std::vector<unsigned>& shape):
        data_cpu(_reserve_space_cpu(shape)),
        ndims_cpu(shape.size()),
        shape_cpu(_init_shape_cpu(shape))
        {
        }

        template <class data_type>
        unsigned
        TensorCPU<data_type>::_compute_index
        (va_list& args, unsigned ndims)
        {
            unsigned* indices = new unsigned[ndims];
            for(unsigned i = 0; i < ndims; i++)
            {
                indices[i] = va_arg(args, unsigned);
                bnn::utils::check(indices[i] < this->shape_cpu[i],
                                  "Index out of range.");
            }

            unsigned prods = 1, index = 0;
            for(unsigned i = ndims - 1; i >= 0; i--)
            {
                index += prods*indices[i];
                prods *= this->shape_cpu[i];
            }
        }

        template <class data_type>
        data_type
        TensorCPU<data_type>::at(...)
        {
            va_list args;
            va_start(args, this->ndims_cpu);
            unsigned index = this->_compute_index(args);
            return this->data_cpu[index];
        }

        template <class data_type>
        void
        TensorCPU<data_type>::set(data_type value, ...)
        {
            va_list args;
            va_start(args, this->ndims_cpu);
            unsigned index = this->_compute_index(args);
            this->data_cpu[index] = value;
        }

        template <class data_type>
        unsigned*
        TensorCPU<data_type>::get_shape()
        {
            return this->shape_cpu;
        }

        template <class data_type>
        unsigned
        TensorCPU<data_type>::get_ndims()
        {
            return this->ndims_cpu;
        }

        template <class data_type>
        TensorCPU<data_type>::~TensorCPU()
        {
            delete [] this->shape_cpu;
            delete [] this->data_cpu;
        }

        #include "bnn/templates/core_tensor.hpp"
    }
}

#endif
