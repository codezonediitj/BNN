#ifndef BNN_BNN_CORE_TENSOR_CPP
#define BNN_BNN_CORE_TENSOR_CPP

#include <cstring>
#include <bnn/core/tensor.hpp>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace core
    {

        template <class data_type>
        data_type*
        TensorCPU<data_type>::
        _reserve_space_cpu
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
        data_type*
        TensorCPU<data_type>::
        _reserve_space_cpu
        (unsigned* shape, unsigned ndims)
        {
            unsigned total_space = 1;
            for(unsigned i = 0; i < ndims; i++)
            {
                total_space *= shape[i];
            }
            data_type* pointer = new data_type[total_space];
            return pointer;
        }

        template <class data_type>
        unsigned*
        TensorCPU<data_type>::
        _init_shape_cpu
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
        unsigned*
        TensorCPU<data_type>::
        _init_shape_cpu
        (unsigned* _shape, unsigned _ndims)
        {
            unsigned* shape = new unsigned[_ndims];
            for(unsigned i = 0; i < _ndims; i++)
            {
                shape[i] = _shape[i];
            }
            return shape;
        }

        template <class data_type>
        TensorCPU<data_type>::
        TensorCPU
        ():
        data_cpu(NULL),
        ndims_cpu(0),
        shape_cpu(NULL)
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>::
        TensorCPU
        (std::vector<unsigned>& shape):
        data_cpu(_reserve_space_cpu(shape)),
        ndims_cpu(shape.size()),
        shape_cpu(_init_shape_cpu(shape))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        TensorCPU<data_type>::
        TensorCPU
        (unsigned* _shape, unsigned _ndims):
        data_cpu(_reserve_space_cpu(_shape, _ndims)),
        ndims_cpu(_ndims),
        shape_cpu(_init_shape_cpu(_shape, _ndims))
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        data_type
        TensorCPU<data_type>::
        at
        (unsigned s, ...)
        {
            va_list args;
            va_start(args, s);
            unsigned* indices = new unsigned[this->ndims_cpu];
            indices[0] = s;
            for(unsigned i = 1; i < this->ndims_cpu; i++)
            {
                indices[i] = va_arg(args, unsigned);
                bnn::utils::check(indices[i] < this->shape_cpu[i],
                                  "Index out of range.");
            }

            unsigned prods = 1, index = 0;
            for(int i = this->ndims_cpu - 1; i >= 0; i--)
            {
                index += prods*indices[i];
                prods *= this->shape_cpu[i];
            }
            return this->data_cpu[index];
        }

        template <class data_type>
        void
        TensorCPU<data_type>::
        set
        (data_type value, ...)
        {
            va_list args;
            va_start(args, value);
            unsigned* indices = new unsigned[this->ndims_cpu];
            for(unsigned i = 0; i < this->ndims_cpu; i++)
            {
                indices[i] = va_arg(args, unsigned);
                bnn::utils::check(indices[i] < this->shape_cpu[i],
                                  "Index out of range.");
            }

            unsigned prods = 1, index = 0;
            for(int i = this->ndims_cpu - 1; i >= 0; i--)
            {
                index += prods*indices[i];
                prods *= this->shape_cpu[i];
            }
            this->data_cpu[index] = value;
        }

        template <class data_type>
        unsigned*
        TensorCPU<data_type>::
        get_shape
        ()
        {
            return this->shape_cpu;
        }

        template <class data_type>
        unsigned
        TensorCPU<data_type>::
        get_ndims
        ()
        {
            return this->ndims_cpu;
        }

        template <class data_type>
        data_type*
        TensorCPU<data_type>::
        get_data_pointer
        ()
        {
            return this->data_cpu;
        }

        template <class data_type>
        void
        TensorCPU<data_type>::
        reshape
        (vector<unsigned>& shape)
        {
            unsigned size = _calc_size(this->shape_cpu, this->ndims_cpu);
            unsigned* new_shape = new unsigned[shape.size()];
            copy(shape.begin(), shape.end(), new_shape);
            this->reshape(new_shape, shape.size());
        }

        template <class data_type>
        void
        TensorCPU<data_type>::
        reshape
        (unsigned* shape, unsigned ndims)
        {
            unsigned size = _calc_size(this->shape_cpu, this->ndims_cpu);
            unsigned* new_shape = new unsigned[ndims];
            for(unsigned i = 0; i < ndims; i++)
            {
                new_shape[i] = shape[i];
            }
            unsigned _size = _calc_size(new_shape, ndims);
            if(size != _size)
            {
                delete new_shape;
                string msg = "The new shape consumes different amount of memory.";
                check(false, msg);
            }
            this->shape_cpu = new_shape;
            this->ndims_cpu = ndims;
        }

        template <class data_type>
        TensorCPU<data_type>::
        ~TensorCPU
        ()
        {
            if(this->shape_cpu != NULL)
                delete [] this->shape_cpu;
            if(this->data_cpu != NULL)
                delete [] this->data_cpu;
            BNNMemory->invalidate(this);
        }

        #include "bnn/templates/core/tensor.hpp"

    }
}

#endif
