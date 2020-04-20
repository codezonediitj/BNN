#ifndef BNN_BNN_CUDA_CORE_TENSOR_HPP
#define BNN_BNN_CUDA_CORE_TENSOR_HPP

#include <vector>
#include <bnn/core/tensor.hpp>

namespace bnn
{
    namespace cuda
    {
        namespace core
        {

            using namespace std;
            using namespace bnn::core;

            /*
            * This class represents GPU version of
            * bnn::core::TensorCPU
            *
            * @tparam data_type Data type of the elements
            *     supported by C++.
            */
            template <class data_type>
            class TensorGPU: public TensorCPU<data_type>
            {
                private:

                    //! The shape of the tensor.
                    unsigned* shape_gpu;

                    //! The number of dimensions in the tensor.
                    unsigned ndims_gpu;

                    //! Array for storing data internally in tensors(on GPU).
                    data_type* data_gpu;

                    /*
                    * For reserving space in GPU memory accoring to a given shape.
                    * Used in initializer list of parameterized constructors.
                    * Returns a new pointer.
                    *
                    * @param shape The shape vector for which the space
                    *     is to be reserved.
                    */
                    static
                    data_type*
                    _reserve_space_gpu
                    (std::vector<unsigned>& shape);

                    /*
                    * For reserving space in CPU memory for storing shape.
                    * Used in initializer list of parameterized constructors.
                    * Returns a new pointer.
                    *
                    * @param shape The shape vector which is to be stored in array.
                    */
                    static
                    unsigned*
                    _init_shape_gpu
                    (std::vector<unsigned>& shape);

                public:

                    /*
                    * Default constructor.
                    * Sets all the pointers to NULL and
                    * integers to 0.
                    */
                    TensorGPU
                    ();

                    /*
                    * Prameterized constructor.
                    *
                    * @param shape std::vector which is to be used
                    *    for initialisation.
                    */
                    TensorGPU
                    (vector<unsigned>& shape);

                    /*
                    * Used for obtaining the pointer to the shape array
                    * of the tensor.
                    *
                    * @param gpu If set to true then shape of tensor
                    *    on GPU memory will be returned, otherwise,
                    *    shape of tensor on CPU memory will be returned.
                    */
                    unsigned*
                    get_shape
                    (bool gpu);

                    /*
                    * Used for obtaining the number of dimensions in the tensor.
                    *
                    * @param gpu If set to true then number of dimensions
                    *    in tensor on GPU memory will be returned, otherwise,
                    *    number of dimensions in tensor on CPU memory will be returned.
                    */
                    unsigned
                    get_ndims
                    (bool gpu);

                    data_type*
                    get_data_pointer
                    (bool gpu);

                     /*
                    * Copies the data from GPU to CPU.
                    */
                    void
                    copy_to_host
                    ();

                    /*
                    * Copies the data from CPU to GPU.
                    */
                    void
                    copy_to_device
                    ();

                    /*
                    * Used for freeing GPU memory.
                    */
                    ~TensorGPU
                    ();
            };

        }
    }
}

#endif
