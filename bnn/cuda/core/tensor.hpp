#ifndef BNN_BNN_CUDA_CORE_TENSOR_HPP
#define BNN_BNN_CUDA_CORE_TENSOR_HPP

#include<vector>
#include<bnn/core/tensor.hpp>

namespace bnn
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type>
            class TensorGPU: public bnn::core::TensorCPU<data_type>
            {
                private:

                    unsigned* shape_gpu;

                    unsigned ndims_gpu;

                    data_type* data_gpu;

                    static data_type*
                    _reserve_space_gpu
                    (std::vector<unsigned>& shape);

                    static unsigned*
                    _init_shape_gpu
                    (std::vector<unsigned>& shape);

                public:

                    TensorGPU();

                    TensorGPU(std::vector<unsigned>& shape);

                    unsigned* get_shape(bool gpu);

                    unsigned get_ndims(bool gpu);

                    data_type* get_data_pointer(bool gpu);

                     /*
                    * Copies the data from GPU to CPU.
                    */
                    void copy_to_host();

                    /*
                    * Copies the  data from CPU to GPU.
                    */
                    void copy_to_device();

                    ~TensorGPU();

            };
        }
    }
}

#endif
