#ifndef BNN_BNN_CUDA_CORE_TENSOR_IMPL_HPP
#define BNN_BNN_CUDA_CORE_TENSOR_IMPL_HPP

#include <bnn/cuda/core/tensor.hpp>
#include <bnn/cuda/utils/cuda_wrappers.hpp>
#include <bnn/utils/utils.hpp>
#include<iostream>

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
                bnn::cuda::utils::cuda_malloc((void**)(&pointer),
                                              total_space*sizeof(data_type));
                return pointer;
            }

            template <class data_type>
            unsigned*
            TensorGPU<data_type>::_init_shape_gpu
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
            TensorGPU<data_type>::TensorGPU():
            bnn::core::TensorCPU<data_type>::TensorCPU(),
            data_gpu(NULL),
            ndims_gpu(0),
            shape_gpu(NULL)
            {
            }

            template <class data_type>
            TensorGPU<data_type>::TensorGPU
            (std::vector<unsigned>& shape):
            bnn::core::TensorCPU<data_type>::TensorCPU(shape),
            data_gpu(_reserve_space_gpu(shape)),
            ndims_gpu(shape.size()),
            shape_gpu(_init_shape_gpu(shape))
            {
            }

            template <class data_type>
            unsigned*
            TensorGPU<data_type>::get_shape(bool gpu)
            {
                return gpu ? this->shape_gpu :
                             this->bnn::core::TensorCPU<data_type>
                             ::get_shape();
            }

            template <class data_type>
            unsigned
            TensorGPU<data_type>::get_ndims(bool gpu)
            {
                return gpu ? this->ndims_gpu :
                             this->bnn::core::TensorCPU<data_type>
                             ::get_ndims();
            }

            template <class data_type>
            data_type*
            TensorGPU<data_type>::get_data_pointer(bool gpu)
            {
                return gpu ? this->data_gpu :
                             this->bnn::core::TensorCPU<data_type>
                             ::get_data_pointer();
            }

            template <class data_type_vector>
            void
            TensorGPU<data_type_vector>::copy_to_host()
            {
                bnn::cuda::utils::cuda_memcpy(
                this->get_data_pointer(false),
                this->get_data_pointer(true),
                this->size_gpu*sizeof(data_type_vector),
                bnn::cuda::utils::DeviceToHost);
            }

            template <class data_type_vector>
            void
            TensorGPU<data_type_vector>::copy_to_device()
            {
                bnn::cuda::utils::cuda_memcpy(
                this->get_data_pointer(true),
                this->get_data_pointer(false),
                this->size_gpu*sizeof(data_type_vector),
                bnn::cuda::utils::HostToDevice);
            }

            template <class data_type>
            TensorGPU<data_type>::~TensorGPU()
            {
                if(this->shape_gpu != NULL)
                    bnn::cuda::utils::cuda_free((void*)this->shape_gpu);
                if(this->data_gpu != NULL)
                    bnn::cuda::utils::cuda_free((void*)this->data_gpu);
            }

            #include "bnn/templates/cuda_core_tensor.hpp"

        }
    }
}

#endif
