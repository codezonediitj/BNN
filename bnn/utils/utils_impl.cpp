#ifndef BNN_UTILS_UTILS_IMPL_CPP
#define BNN_UTILS_UTILS_IMPL_CPP

#include <bnn/utils/utils.hpp>
#include <string>
#include <stdexcept>
#include <thread>

namespace bnn
{
    namespace utils
    {

        using namespace std;

        BNNBase::
        ~BNNBase
        ()
        {
        }

        ObjectStack::
        ObjectStack
        (ObjectStack* _prev, BNNBase* obj):
        prev(_prev),
        ptr(obj)
        {
        }

        ThreadStack::
        ThreadStack
        (ThreadStack* _prev, thread* t):
        prev(_prev),
        ptr(t)
        {
        }

        MemoryManager::
        MemoryManager
        ():
        top(NULL)
        {
        }

        void
        MemoryManager::
        push
        (BNNBase* obj)
        {
            ObjectStack* node = new ObjectStack(top, obj);
            this->objects[obj] = true;
            this->top = node;
        }

        void
        MemoryManager::
        free_memory
        (BNNBase* obj)
        {
            if(this->objects[obj])
            {
                delete obj;
                this->invalidate(obj);
            }
        }

        void
        MemoryManager::
        invalidate
        (BNNBase* obj)
        {
            this->objects[obj] = false;
        }

        ObjectStack*
        MemoryManager::
        pop
        ()
        {
            ObjectStack* ret = this->top;
            if(ret != NULL)
            {
                this->top = ret->prev;
            }
            return ret;
        }

        MemoryManager::
        ~MemoryManager
        ()
        {
            ObjectStack* curr = this->pop();
            while(curr != NULL)
            {
                this->free_memory(curr->ptr);
                delete curr;
                curr = this->pop();
            }
        }

        ThreadManager::
        ThreadManager
        ():
        top(NULL)
        {
        }

        void
        ThreadManager::
        push
        (thread* t)
        {
            ThreadStack* node = new ThreadStack(top, t);
            this->threads[t] = true;
            this->top = node;
        }

        void
        ThreadManager::
        free_thread
        (thread* t)
        {
            if(this->threads[t])
            {
                if(t->joinable())
                {
                    t->join();
                }
                delete t;
                this->invalidate(t);
            }
        }

        void
        ThreadManager::
        stop_thread
        (thread* t)
        {
            if(this->threads[t])
            {
                if(t->joinable())
                {
                    t->detach();
                }
                delete t;
                this->invalidate(t);
            }
        }

        void
        ThreadManager::
        invalidate
        (thread* t)
        {
            this->threads[t] = false;
        }

        ThreadStack*
        ThreadManager::
        pop
        ()
        {
            ThreadStack* ret = this->top;
            if(ret != NULL)
            {
                this->top = ret->prev;
            }
            return ret;
        }

        ThreadManager::
        ~ThreadManager
        ()
        {
            ThreadStack* curr = this->pop();
            while(curr != NULL)
            {
                if(this->threads[curr->ptr])
                {
                    this->stop_thread(curr->ptr);
                }
                delete curr;
                curr = this->pop();
            }
        }

        MemoryManager* BNNMemory = new MemoryManager;

        ThreadManager* BNNThreads = new ThreadManager;

        void
        check
        (bool exp, string msg)
        {
            if(!exp)
            {
                delete BNNThreads;
                delete BNNMemory;
                throw std::logic_error(msg);
            }
        }

        template <class data_type>
        op_queue<data_type>::
        op_queue
        ():
        next(NULL),
        op(NULL)
        {
            BNNMemory->push(this);
        }

        template <class data_type>
        op_queue<data_type>::
        void
        clear
        (op_queue<data_type>* ptr)
        {
            op_queue<data_type>* curr = ptr;
            op_queue<data_type>* curr_next;
            while(curr != NULL)
            {
                curr_next = curr->next;
                BNNMemory->free_memory(curr);
                curr = curr_next;
            }
        }

    } // namespace utils
} // namspace adaboost

#endif
