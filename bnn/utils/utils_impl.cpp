#ifndef BNN_UTILS_UTILS_IMPL_CPP
#define BNN_UTILS_UTILS_IMPL_CPP

#include <bnn/utils/utils.hpp>
#include <string>
#include <stdexcept>

namespace bnn
{
    namespace utils
    {

        using namespace std;

        void
        check
        (bool exp, string msg)
        {
            if(!exp)
                throw std::logic_error(msg);
        }

    } // namespace utils
} // namspace adaboost

#endif
