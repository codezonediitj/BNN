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
            class TensorGPU: public TensorCPU
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

                    data_type at(bool gpu, ...);

                    void set(bool gpu, data_type value, ...);

                    unsigned* get_shape(bool gpu);

                    unsigned get_ndims(bool gpu);

                    ~TensorGPU();

            };
        }
    }
}

#endif
