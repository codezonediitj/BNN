#ifndef BNN_BNN_UTILS_CUDA_WRAPPERS_IMPL_HPP
#define BNN_BNN_UTILS_CUDA_WRAPPERS_IMPL_HPP

#include<bnn/cuda/utils/cuda_wrappers.hpp>

namespace bnn
{
    namespace cuda
    {
        namespace utils
        {
            void cuda_malloc(void** ptr, unsigned num_bytes)
            {
                cudaMalloc(ptr, num_bytes);
            }

            void cuda_memcpy
            (void* ptr_1, void* ptr_2, unsigned num_bytes, direction d)
            {
                if(d == HostToDevice)
                {
                    cudaMemcpy(ptr_1, ptr_2, num_bytes, cudaMemcpyHostToDevice);
                }
                else if(d == DeviceToHost)
                {
                    cudaMemcpy(ptr_1, ptr_2, num_bytes, cudaMemcpyDeviceToHost);
                }
            }

            void cuda_event_create(cuda_event_t* event_ptr)
            {
                cudaEventCreate(event_ptr);
            }

            void cuda_event_record(cuda_event_t event)
            {
                cudaEventRecord(event);
            }

            void cuda_event_synchronize(cuda_event_t event)
            {
                cudaEventSynchronize(event);
            }

            void cuda_free(void* ptr)
            {
                cudaFree(ptr);
            }

        } // namespace cuda
    } // namespace utils
} // namespace adaboost

#endif
