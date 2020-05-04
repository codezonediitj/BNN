#ifndef BNN_BNN_IO_INPUT_IMPL_CPP
#define BNN_BNN_IO_INPUT_IMPL_CPP

#include <bnn/core/tensor.hpp>
#include <bnn/io/input.hpp>
#include <fstream>
#include <bnn/utils/utils.hpp>

namespace bnn
{
    namespace io
    {

        using namespace bnn::core;

        auto is_little_endian = []()
        {
            int num = 1;
            return *(char*)&num == 1;
        };

        inline
        unsigned
        _reverse_int
        (unsigned i)
        {
            if(is_little_endian())
            {
                unsigned char c1, c2, c3, c4;

                c1 = i & 255;
                c2 = (i >> 8) & 255;
                c3 = (i >> 16) & 255;
                c4 = (i >> 24) & 255;

                return ((unsigned)c1 << 24) +
                    ((unsigned)c2 << 16) +
                    ((unsigned)c3 << 8) + c4;
            }

            return i;
        }

        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_images
        (const string& path)
        {
            string msg = "Error while reading " + path;
            ifstream* in = new ifstream(path, std::ios::binary);
            if(!in)
            {
                check(false, msg);
            }
            unsigned magic_number;
            in->read(reinterpret_cast<char*>(&magic_number), 4);
            magic_number = _reverse_int(magic_number);
            string invalid_image_file = "Invalid MNIST image file!";
            check(magic_number == 2051, invalid_image_file);
            unsigned shape[3];
            in->read(reinterpret_cast<char*>(shape), 4);
            in->read(reinterpret_cast<char*>(shape + 1), 4);
            in->read(reinterpret_cast<char*>(shape + 2), 4);
            shape[0] = _reverse_int(shape[0]);
            shape[1] = _reverse_int(shape[1]);
            shape[2] = _reverse_int(shape[2]);;
            TensorCPU<data_type>* images = new TensorCPU<data_type>(shape, 3);
            unsigned size = _calc_size(shape, 3);
            data_type* imgs = images->get_data_pointer();
            for(unsigned ptr = 0; ptr < size; ptr++)
            {

                if(!(in->read(reinterpret_cast<char*>(imgs + ptr), 1)))
                {
                    check(false, msg);
                }
            }
            delete in;
            return images;
        }

        template <class data_type>
        TensorCPU<data_type>*
        load_mnist_labels
        (const string& path)
        {
            string msg = "Error while reading " + path;
            ifstream* in = new ifstream(path, std::ios::binary);
            if(!in)
            {
                check(false, msg);
            }
            unsigned magic_number;
            in->read(reinterpret_cast<char*>(&magic_number), 4);
            magic_number = _reverse_int(magic_number);
            string invalid_label_file = "Invalid MNIST label file!";
            check(magic_number == 2049, invalid_label_file);
            unsigned shape[1];
            in->read(reinterpret_cast<char*>(shape), 4);
            shape[0] = _reverse_int(shape[0]);
            TensorCPU<data_type>* labels = new TensorCPU<data_type>(shape, 1);
            unsigned size = _calc_size(shape, 1);
            data_type* _labels = labels->get_data_pointer();
            for(unsigned ptr = 0; ptr < size; ptr++)
            {
                if(!in->read(reinterpret_cast<char*>(_labels + ptr), 1))
                {
                    check(false, msg);
                }
            }
            delete in;
            return labels;
        }

        #include "bnn/templates/io/input.hpp"

    }
}

#endif
