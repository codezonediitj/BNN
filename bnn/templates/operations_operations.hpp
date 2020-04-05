#ifndef BNN_BNN_TEMPLATES_OPERATIONS_OPERATIONS_HPP
#define BNN_BNN_TEMPLATES_OPERATIONS_OPERATIONS_HPP

template bnn::operators::Add* add(
bnn::core::TensorCPU<bool>& a,
bnn::core::TensorCPU<bool>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<short>& a,
bnn::core::TensorCPU<short>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<unsigned short>& a,
bnn::core::TensorCPU<unsigned short>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<int>& a,
bnn::core::TensorCPU<int>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<unsigned int>& a,
bnn::core::TensorCPU<unsigned int>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<long>& a,
bnn::core::TensorCPU<long>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<unsigned long>& a,
bnn::core::TensorCPU<unsigned long>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<long long>& a,
bnn::core::TensorCPU<long long>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<unsigned long long>& a,
bnn::core::TensorCPU<unsigned long long>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<float>& a,
bnn::core::TensorCPU<float>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<double>& a,
bnn::core::TensorCPU<double>& b);
template bnn::operators::Add* add(
bnn::core::TensorCPU<long double>& a,
bnn::core::TensorCPU<long double>& b);

#endif
