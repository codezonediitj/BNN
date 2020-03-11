#ifndef BNN_BNN_CUDA_CORE_TENSOR_IMPL_HPP
#define BNN_BNN_CUDA_CORE_TENSOR_IMPL_HPP

#include <bnn/cuda/core/tensor.hpp>
#include <bnn/utils/cuda_wrappers.hpp>

namespace bnn
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type>
            data_type*
            TensorGPU<data_type>::_reserve_space_gpu
            (std::vector<unsigned>& shape)
            {
                unsigned total_space = 1;
                for(unsigned i = 0; i < shape.size(); i++)
                {
                    total_space *= shape.at(i);
                }
                data_type* pointer;
                cuda_malloc(pointer, total_space*sizeof(data_type));
            }

            template <class data_type>
            unsigned*
            TensorGPU<data_type>::_init_shape_gpu
            (std::vector<unsigned>& shape)
            {
                unsigned* _shape;
                cuda_malloc(shape, shape.size()*sizeof(unsigned));
                for(unsigned i = 0; i < shape.size(); i++)
                {
                    _shape[i] = shape.at(i);
                }
                return _shape;
            }

            template <class data_type>
            TensorGPU<data_type>::TensorGPU():
            TensorCPU<data_type>::TensorCPU(),
            data_gpu(NULL),
            ndims_gpu(0),
            shape_gpu(NULL)
            {
            }

            template <class data_type>
            TensorGPU<data_type>::TensorGPU
            (std::vector<unsigned>& shape):
            TensorCPU<data_type>::TensorCPU(shape),
            data_gpu(_reserve_space_gpu(shape)),
            ndims_gpu(shape.size()),
            shape_gpu(_init_shape_gpu(shape))
            {
            }

            template <class data_type>
            data_type
            TensorGPU<data_type>::at(bool gpu, ...)
            {
                unsigned ndims = gpu ? this->ndims_gpu : this->ndims_cpu;
                va_list args;
                va_start(args, ndims);
                unsigned index = this->TensorCPU::_compute_index(args, ndims);
                return gpu ? this->data_gpu[index] : this->data_cpu[index];
            }

            template <class data_type>
            void
            TensorGPU<data_type>::set(bool gpu, data_type value, ...)
            {
                unsigned ndims = gpu ? this->ndims_gpu : this->ndims_cpu;
                va_list args;
                va_start(args, ndims);
                unsigned index = this->TensorCPU::_compute_index(args, ndims);
                if(gpu)
                    this->data_gpu[index] = value;
                else
                    this->data_cpu[index] = value;
            }

            template <class data_type>
            unsigned*
            TensorGPU<data_type>::get_shape(bool gpu)
            {
                return gpu ? this->shape_gpu : this->shape_gpu;
            }

            template <class data_type>
            unsigned
            TensorGPU<data_type>::get_ndims(bool gpu)
            {
                return gpu ? this->ndims_gpu : this->ndims_cpu;
            }

            template <class data_type>
            TensorGPU<data_type>::~TensorGPU()
            {
                cuda_free(this->shape_gpu);
                cuda_free(this->data_gpu);
            }

        }
    }
}

#endif
