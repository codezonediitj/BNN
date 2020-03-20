#ifndef BNN_BNN_TEMPLATES_CORE_TENSOR_HPP
#define BNN_BNN_TEMPLATES_CORE_TENSOR_HPP

template class TensorCPU<bool>;
template class TensorCPU<short>;
template class TensorCPU<unsigned short>;
template class TensorCPU<int>;
template class TensorCPU<unsigned int>;
template class TensorCPU<long>;
template class TensorCPU<unsigned long>;
template class TensorCPU<long long>;
template class TensorCPU<unsigned long long>;
template class TensorCPU<float>;
template class TensorCPU<double>;
template class TensorCPU<long double>;

#endif
