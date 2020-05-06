#ifndef BNN_BNN_CORE_TENSOR_HPP
#define BNN_BNN_CORE_TENSOR_HPP

#include <vector>
#include <cstdarg>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace core
    {

        using namespace std;
        using namespace bnn::utils;

        /*
        * This class represents tensors used for storing data
        * in multiple dimensions using internal 1D layout,
        * implemented using C++ arrays.
        *
        * @tparam data_type Data type of the elements
        *     supported by C++.
        */
        template <class data_type>
        class TensorCPU: public BNNBase
        {
            private:

                //! The shape of the tensor.
                unsigned* shape_cpu;

                //! The number of dimensions in the tensor.
                unsigned ndims_cpu;

                //! Array for storing data internally in tensors.
                data_type* data_cpu;

                /*
                * For reserving space in CPU memory accoring to a given shape.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param shape The shape std::vector according to which the space
                *     is to be reserved.
                */
                static
                data_type*
                _reserve_space_cpu
                (vector<unsigned>& shape);

                /* @overload
                * For reserving space in CPU memory accoring to a given shape
                * and number of dimensions.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param shape unsigned* array according to which the space
                *     is to be reserved.
                * @param ndims unsigned Number dimensions in the tensor.
                */
                static
                data_type*
                _reserve_space_cpu
                (unsigned* shape, unsigned ndims);

                /*
                * For reserving space in CPU memory for storing shape.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param shape The shape std::vector which is to be stored in array.
                */
                static
                unsigned*
                _init_shape_cpu
                (vector<unsigned>& shape);

                /* @overload
                * For reserving space in CPU memory for storing shape.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param shape unsigned* The shape of the tensor.
                * @param ndims unsigned Number dimensions in the tensor.
                */
                static
                unsigned*
                _init_shape_cpu
                (unsigned* shape, unsigned ndims);

            public:

                /*
                * Default constructor.
                * Sets all the pointers to NULL and
                * integers to 0.
                */
                TensorCPU
                ();

                /* @overload
                * Prameterized constructor.
                *
                * @param shape std::vector The shape
                *    std::vector which is to be used
                *    for initialisation.
                */
                TensorCPU
                (vector<unsigned>& shape);

                /* @overload
                * Prameterized constructor.
                *
                * @param shape unsigned* The shape of the tensor.
                * @param ndims unsigned Number dimensions in the tensor.
                */
                TensorCPU
                (unsigned* shape, unsigned ndims);

                /*
                * Used for accessing the element of the tensor
                * at given indices.
                *
                * @param ... Accepts variable number of integer
                *    arugments as indices.
                */
                data_type
                at
                (unsigned s, ...);

                /*
                * Used for setting the element of the tensor
                * at given indices with the given value.
                *
                * @param value The value which is to be
                *    used for storing at given indices.
                * @param ... Accepts variable number of integer
                *    arugments as indices.
                */
                void
                set
                (data_type value, ...);

                /*
                * Used for obtaining the pointer to the shape array
                * of the tensor.
                */
                unsigned*
                get_shape
                ();

                /*
                * Used for obtaining the number of dimensions in the tensor.
                */
                unsigned
                get_ndims
                ();

                data_type*
                get_data_pointer
                ();

                void
                reshape
                (vector<unsigned>& shape);

                void
                reshape
                (unsigned* shape, unsigned ndims);

                /*
                * Used for freeing CPU memory.
                */
                ~TensorCPU
                ();
        };

    }
}

#endif
