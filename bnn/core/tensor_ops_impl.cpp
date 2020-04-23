#ifndef BNN_BNN_CORE_TENSOR_OPS_IMPL_CPP
#define BNN_BNN_CORE_TENSOR_OPS_IMPL_CPP

#include <thread>
#include <string>
#include <cmath>
#include <bnn/utils/utils.hpp>
#include <bnn/core/tensor_ops.hpp>

namespace bnn
{
    namespace core
    {

        using namespace std;
        using namespace bnn::utils;

        thread* null_ptr = NULL;

        unsigned
        _calc_size
        (unsigned* shape, unsigned ndims)
        {
            unsigned size = 1;
            for(unsigned i = 0; i < ndims; i++)
            {
                size *= shape[i];
            }
            return size;
        }

        void
        _check_dimensions
        (unsigned* shapex, unsigned* shapey,
         unsigned ndimsx, unsigned ndimsy)
        {
            string msg = "Tensors should be of same dimensions, " +
                         to_string(ndimsx) + "!=" + to_string(ndimsy);
            check(ndimsx == ndimsy, msg);
            for(unsigned i = 0; i < ndimsx; i++)
            {
                msg = "Tensors should be of same shape, " +
                      to_string(shapex[i]) + "!=" + to_string(shapey[i]);
                check(shapex[i] == shapey[i], msg);
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        unary_op
        (TensorCPU<data_type>* x,  void (*thread_job)
         (data_type*, data_type*,
          unsigned, unsigned))
        {
            TensorCPU<data_type>* z = new TensorCPU<data_type>
                                       (x->get_shape(), x->get_ndims());
            data_type *xd = x->get_data_pointer();
            data_type *zd = z->get_data_pointer();
            unsigned size = _calc_size(x->get_shape(), x->get_ndims());
            unsigned nthreads = thread::hardware_concurrency();
            if(size/nthreads > 0)
            {
                thread* pool[nthreads];
                std::fill(pool, pool + nthreads, null_ptr);
                unsigned sizet = size/nthreads;
                unsigned idx = 0, i = 0;
                while(idx < size)
                {
                    unsigned ti = i%nthreads;
                    if(pool[ti] != NULL)
                    {
                        BNNThreads->free_thread(pool[ti]);
                        pool[ti] = NULL;
                    }
                    pool[ti] = new thread(thread_job, xd, zd,
                                          idx, min(idx + sizet, size));
                    BNNThreads->push(pool[ti]);
                    idx += sizet;
                    i++;
                }
                for(unsigned i = 0; i < nthreads; i++)
                {
                    if(pool[i] != NULL)
                    {
                        BNNThreads->free_thread(pool[i]);
                    }
                }
            }
            else
            {
                thread_job(xd, zd, 0, size);
            }

            return z;
        }

        template <class data_type>
        TensorCPU<data_type>*
        binary_op
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y,
         void (*thread_job)(data_type*,
         data_type*, data_type*,
         unsigned, unsigned))
        {
            _check_dimensions(x->get_shape(), y->get_shape(),
                              x->get_ndims(), y->get_ndims());
            TensorCPU<data_type>* z = new TensorCPU<data_type>
                                       (x->get_shape(), x->get_ndims());
            data_type *xd = x->get_data_pointer();
            data_type *yd = y->get_data_pointer();
            data_type *zd = z->get_data_pointer();
            unsigned size = _calc_size(x->get_shape(), x->get_ndims());
            unsigned nthreads = thread::hardware_concurrency();
            if(size/nthreads > 0)
            {
                thread* pool[nthreads];
                std::fill(pool, pool + nthreads, null_ptr);
                unsigned sizet = size/nthreads;
                unsigned idx = 0, i = 0;
                while(idx < size)
                {
                    unsigned ti = idx%nthreads;
                    if(pool[ti] != NULL)
                    {
                        BNNThreads->free_thread(pool[ti]);
                        pool[ti] = NULL;
                    }
                    pool[ti] = new thread(thread_job, xd, yd, zd,
                                          idx, min(idx + sizet, size));
                    BNNThreads->push(pool[ti]);
                    idx += sizet;
                    i++;
                }
                for(i = 0; i < nthreads; i++)
                {
                    if(pool[i] != NULL)
                    {
                        BNNThreads->free_thread(pool[i]);
                    }
                }
            }
            else
            {
                thread_job(xd, yd, zd, 0, size);
            }

            return z;
        }

        template <class data_type>
        void
        _add_job
        (data_type* xd, data_type* yd,
         data_type* zd, unsigned start,
         unsigned end)
        {
            for(unsigned i = start; i < end; i++)
            {
                zd[i] = xd[i] + yd[i];
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        add
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y)
        {
            return binary_op(x, y, &_add_job<data_type>);
        }

        template <class data_type>
        void
        _mul_job
        (data_type* xd, data_type* yd,
         data_type* zd, unsigned start,
         unsigned end)
        {
            for(unsigned i = start; i < end; i++)
            {
                zd[i] = xd[i]*yd[i];
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        mul
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y)
        {
            return binary_op(x, y, &_mul_job<data_type>);
        }

        template <class data_type>
        void
        _exp_job
        (data_type* xd, data_type* zd,
         unsigned start, unsigned end)
        {
            for(unsigned i = start; i < end; i++)
            {
                zd[i] = std::exp(xd[i]);
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        exp
        (TensorCPU<data_type>* x)
        {
            return unary_op(x, &_exp_job<data_type>);
        }

        template <class data_type>
        void
        _fill_job
        (data_type* xd, data_type val,
         unsigned start, unsigned end)
        {
            for(unsigned i = start; i < end; i++)
            {
                xd[i] = val;
            }
        }

        template <class data_type>
        void
        fill
        (TensorCPU<data_type>* x, data_type val)
        {
            data_type *xd = x->get_data_pointer();
            unsigned size = _calc_size(x->get_shape(), x->get_ndims());
            unsigned nthreads = thread::hardware_concurrency();
            if(size/nthreads > 0)
            {
                thread* pool[nthreads];
                std::fill(pool, pool + nthreads, null_ptr);
                unsigned sizet = size/nthreads;
                unsigned idx = 0, i = 0;
                while(idx < size)
                {
                    unsigned ti = i%nthreads;
                    if(pool[ti] != NULL)
                    {
                        BNNThreads->free_thread(pool[ti]);
                        pool[ti] = NULL;
                    }
                    pool[ti] = new thread(_fill_job<data_type>, xd, val,
                                          idx, min(idx + sizet, size));
                    BNNThreads->push(pool[ti]);
                    idx += sizet;
                    i++;
                }
                for(i = 0; i < nthreads; i++)
                {
                    if(pool[i] != NULL)
                    {
                        BNNThreads->free_thread(pool[i]);
                    }
                }
            }
            else
            {
                _fill_job<data_type>(xd, val, 0, size);
            }
        }

        #include "bnn/templates/core/tensor_ops.hpp"

    }
}

#endif
