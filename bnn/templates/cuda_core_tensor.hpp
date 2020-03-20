#ifndef BNN_BNN_TEMPLATES_CUDA_CORE_TENSOR_HPP
#define BNN_BNN_TEMPLATES_CUDA_CORE_TENSOR_HPP

template class TensorGPU<bool>;
template class TensorGPU<short>;
template class TensorGPU<unsigned short>;
template class TensorGPU<int>;
template class TensorGPU<unsigned int>;
template class TensorGPU<long>;
template class TensorGPU<unsigned long>;
template class TensorGPU<long long>;
template class TensorGPU<unsigned long long>;
template class TensorGPU<float>;
template class TensorGPU<double>;
template class TensorGPU<long double>;

#endif
