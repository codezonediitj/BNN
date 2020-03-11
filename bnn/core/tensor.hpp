#ifndef BNN_BNN_CORE_TENSOR_HPP
#define BNN_BNN_CORE_TENSOR_HPP

#include<vector>
#include<cstdarg>

namespace bnn
{
    namespace core
    {
        template <class data_type>
        class TensorCPU
        {
            private:

                unsigned* shape_cpu;

                unsigned ndims_cpu;

                data_type* data_cpu;

                static data_type*
                _reserve_space_cpu
                (std::vector<unsigned>& shape);

                static unsigned*
                _init_shape_cpu
                (std::vector<unsigned>& shape);

            public:

                TensorCPU();

                TensorCPU(std::vector<unsigned>& shape);

                static unsigned
                _compute_index(va_list& args, unsigned ndims);

                data_type at(...);

                void set(data_type value, ...);

                unsigned* get_shape();

                unsigned get_ndims();

                ~TensorCPU();
        };
    }
}

#endif
