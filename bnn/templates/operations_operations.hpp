#ifndef BNN_BNN_TEMPLATES_OPERATIONS_OPERATIONS_HPP
#define BNN_BNN_TEMPLATES_OPERATIONS_OPERATIONS_HPP

template Add* add(
TensorCPU<bool>& a,
TensorCPU<bool>& b);
template Add* add(
TensorCPU<short>& a,
TensorCPU<short>& b);
template Add* add(
TensorCPU<unsigned short>& a,
TensorCPU<unsigned short>& b);
template Add* add(
TensorCPU<int>& a,
TensorCPU<int>& b);
template Add* add(
TensorCPU<unsigned int>& a,
TensorCPU<unsigned int>& b);
template Add* add(
TensorCPU<long>& a,
TensorCPU<long>& b);
template Add* add(
TensorCPU<unsigned long>& a,
TensorCPU<unsigned long>& b);
template Add* add(
TensorCPU<long long>& a,
TensorCPU<long long>& b);
template Add* add(
TensorCPU<unsigned long long>& a,
TensorCPU<unsigned long long>& b);
template Add* add(
TensorCPU<float>& a,
TensorCPU<float>& b);
template Add* add(
TensorCPU<double>& a,
TensorCPU<double>& b);
template Add* add(
TensorCPU<long double>& a,
TensorCPU<long double>& b);
template Add* add(
TensorCPU<bool>& a,
Operator* b);
template Add* add(
TensorCPU<short>& a,
Operator* b);
template Add* add(
TensorCPU<unsigned short>& a,
Operator* b);
template Add* add(
TensorCPU<int>& a,
Operator* b);
template Add* add(
TensorCPU<unsigned int>& a,
Operator* b);
template Add* add(
TensorCPU<long>& a,
Operator* b);
template Add* add(
TensorCPU<unsigned long>& a,
Operator* b);
template Add* add(
TensorCPU<long long>& a,
Operator* b);
template Add* add(
TensorCPU<unsigned long long>& a,
Operator* b);
template Add* add(
TensorCPU<float>& a,
Operator* b);
template Add* add(
TensorCPU<double>& a,
Operator* b);
template Add* add(
TensorCPU<long double>& a,
Operator* b);
template Add* add(
Operator* a,
TensorCPU<bool>& b);
template Add* add(
Operator* a,
TensorCPU<short>& b);
template Add* add(
Operator* a,
TensorCPU<unsigned short>& b);
template Add* add(
Operator* a,
TensorCPU<int>& b);
template Add* add(
Operator* a,
TensorCPU<unsigned int>& b);
template Add* add(
Operator* a,
TensorCPU<long>& b);
template Add* add(
Operator* a,
TensorCPU<unsigned long>& b);
template Add* add(
Operator* a,
TensorCPU<long long>& b);
template Add* add(
Operator* a,
TensorCPU<unsigned long long>& b);
template Add* add(
Operator* a,
TensorCPU<float>& b);
template Add* add(
Operator* a,
TensorCPU<double>& b);
template Add* add(
Operator* a,
TensorCPU<long double>& b);
template Exp*
exp(
TensorCPU<bool>& a);
template Exp*
exp(
TensorCPU<short>& a);
template Exp*
exp(
TensorCPU<unsigned short>& a);
template Exp*
exp(
TensorCPU<int>& a);
template Exp*
exp(
TensorCPU<unsigned int>& a);
template Exp*
exp(
TensorCPU<long>& a);
template Exp*
exp(
TensorCPU<unsigned long>& a);
template Exp*
exp(
TensorCPU<long long>& a);
template Exp*
exp(
TensorCPU<unsigned long long>& a);
template Exp*
exp(
TensorCPU<float>& a);
template Exp*
exp(
TensorCPU<double>& a);
template Exp*
exp(
TensorCPU<long double>& a);

#endif
