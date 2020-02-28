#ifndef BNN_BNN_CUDA_CORE_TENSOR_HPP
#define BNN_BNN_CUDA_CORE_TENSOR_HPP

#include<vector>

namespace bnn
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type>
            class TensorGPU
            {
                private:

                    unsigned* shapes_gpu;

                    unsigned ndims_gpu;

                    data_type* data_gpu;

                    static _reserve_space_gpu
                    (std::vector<unsigned> shape);

                public:

                    TensorGPU();

                    TensorGPU(std::vector<unsigned> shape);

                    data_type at(...);

                    void set(data_type value, ...);

                    void fill(data_type value);

                    ~TensorGPU();
            };
        }
    }
}

#endif
