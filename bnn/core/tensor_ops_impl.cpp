#ifndef BNN_BNN_CORE_TENSOR_OPS_IMPL_CPP
#define BNN_BNN_CORE_TENSOR_OPS_IMPL_CPP

#include <thread>
#include <cstring>
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

        void
        _check_dimensions
        (unsigned* shapex, unsigned* shapey,
         unsigned ndimsx, unsigned ndimsy)
        {
            string msg = "Tensors should be of same dimensions, " +
                         to_string(ndimsx) + " != " + to_string(ndimsy);
            check(ndimsx == ndimsy, msg);
            for(unsigned i = 0; i < ndimsx; i++)
            {
                msg = "Tensors should be of same shape, " +
                      to_string(shapex[i]) + " != " + to_string(shapey[i]);
                check(shapex[i] == shapey[i], msg);
            }
        }

        template <class data_type>
        struct Args
        {
            data_type *xd;
        };

        template <class data_type>
        struct UnaryArgs: Args<data_type>
        {
            data_type *zd;
        };

        template <class data_type>
        struct BinaryArgs: Args<data_type>
        {
            data_type *yd, *zd;
        };

        template <class data_type>
        struct FillArgs: Args<data_type>
        {
            data_type val;
        };

        template <class data_type>
        void
        op
        (TensorCPU<data_type>* x, Args<data_type>* args,
         void (*thread_job)(Args<data_type>*, unsigned, unsigned))
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
                args->xd = xd;
                while(idx < size)
                {
                    unsigned ti = i%nthreads;
                    if(pool[ti] != NULL)
                    {
                        BNNThreads->free_thread(pool[ti]);
                        pool[ti] = NULL;
                    }
                    unsigned start = idx, end = min(idx + sizet, size);
                    pool[ti] = new thread(thread_job, args, start, end);
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
                unsigned start = 0, end = size;
                thread_job(args, start, end);
            }
        }

        template <class data_type>
        void
        _add_job
        (Args<data_type>* _args, unsigned start,
         unsigned end)
        {
            BinaryArgs<data_type>* args = reinterpret_cast<BinaryArgs<data_type>*>(_args);
            for(unsigned i = start; i < end; i++)
            {
                args->zd[i] = args->xd[i] + args->yd[i];
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        add
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y)
        {
            TensorCPU<data_type>* z = new TensorCPU<data_type>
                                       (x->get_shape(), x->get_ndims());
            BinaryArgs<data_type> args;
            args.yd = y->get_data_pointer(), args.zd = z->get_data_pointer();
            op(x, &args, &_add_job<data_type>);
            return z;
        }

        template <class data_type>
        void
        _mul_job
        (Args<data_type>* _args, unsigned start,
         unsigned end)
        {
            BinaryArgs<data_type>* args = reinterpret_cast<BinaryArgs<data_type>*>(_args);
            for(unsigned i = start; i < end; i++)
            {
                args->zd[i] = args->xd[i]*args->yd[i];
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        mul
        (TensorCPU<data_type>* x, TensorCPU<data_type>* y)
        {
            TensorCPU<data_type>* z = new TensorCPU<data_type>
                                       (x->get_shape(), x->get_ndims());
            BinaryArgs<data_type> args;
            args.yd = y->get_data_pointer(), args.zd = z->get_data_pointer();
            op(x, &args, &_mul_job<data_type>);
            return z;
        }

        template <class data_type>
        void
        _exp_job
        (Args<data_type>* _args, unsigned start,
         unsigned end)
        {
            UnaryArgs<data_type>* args = reinterpret_cast<UnaryArgs<data_type>*>(_args);
            for(unsigned i = start; i < end; i++)
            {
                args->zd[i] = std::exp(args->xd[i]);
            }
        }

        template <class data_type>
        TensorCPU<data_type>*
        exp
        (TensorCPU<data_type>* x)
        {
            TensorCPU<data_type>* z = new TensorCPU<data_type>
                                       (x->get_shape(), x->get_ndims());
            UnaryArgs<data_type> args;
            args.zd = z->get_data_pointer();
            op(x, &args, &_exp_job<data_type>);
            return z;
        }

        template <class data_type>
        void
        _fill_job
        (Args<data_type>* _args, unsigned start,
         unsigned end)
        {
            FillArgs<data_type>* args = reinterpret_cast<FillArgs<data_type>*>(_args);
            for(unsigned i = start; i < end; i++)
            {
                args->xd[i] = args->val;
            }
        }

        template <class data_type>
        void
        fill
        (TensorCPU<data_type>* x, data_type val)
        {
            FillArgs<data_type> args;
            args.val = val;
            op(x, &args, &_fill_job<data_type>);
        }

        template <class data_type>
        void
        copy
        (TensorCPU<data_type>* dest, TensorCPU<data_type>* src)
        {
            _check_dimensions(dest->get_shape(), src->get_shape(),
                              dest->get_ndims(), src->get_ndims());
            unsigned size = _calc_size(dest->get_shape(), dest->get_ndims());
            memcpy(dest->get_data_pointer(), src->get_data_pointer(), sizeof(data_type)*size);
        }

        #include "bnn/templates/core/tensor_ops.hpp"

    }
}

#endif
