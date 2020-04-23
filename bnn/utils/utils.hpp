#ifndef BNN_UTILS_UTILS_HPP
#define BNN_UTILS_UTILS_HPP

#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace bnn
{
    namespace utils
    {

        using namespace std;

        void
        check
        (bool exp, string msg);

        class BNNBase
        {
            public:

                virtual
                ~BNNBase
                ();
        };

        struct ObjectStack
        {
            ObjectStack* prev;

            BNNBase* ptr;

            ObjectStack
            (ObjectStack* _prev, BNNBase* ptr);
        };

        struct ThreadStack
        {
            ThreadStack* prev;

            thread* ptr;

            ThreadStack
            (ThreadStack* _prev, thread* ptr);
        };

        class MemoryManager
        {
            private:

                ObjectStack* top;

                unordered_map<BNNBase*, bool> objects;

                ObjectStack*
                pop
                ();

            public:

                MemoryManager
                ();

                void
                push
                (BNNBase* obj);

                void
                free_memory
                (BNNBase* obj);

                void
                invalidate
                (BNNBase* obj);

                ~MemoryManager
                ();
        };

        class ThreadManager
        {
            private:

                ThreadStack* top;

                unordered_map<thread*, bool> threads;

                ThreadStack*
                pop
                ();

            public:

                ThreadManager
                ();

                void
                push
                (thread* t);

                void
                free_thread
                (thread* t);

                void
                stop_thread
                (thread* t);

                void
                invalidate
                (thread* t);

                ~ThreadManager
                ();
        };

        extern MemoryManager* BNNMemory;

        extern ThreadManager* BNNThreads;

    } // namespace utils
} // namspace bnn

#endif
