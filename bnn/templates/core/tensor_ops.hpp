#ifndef BNN_BNN_TEMPLATES_CORE_TENSOR_OPS_HPP
#define BNN_BNN_TEMPLATES_CORE_TENSOR_OPS_HPP

template TensorCPU<bool>* add<bool>(TensorCPU<bool>* x, TensorCPU<bool>* y);
template TensorCPU<short>* add<short>(TensorCPU<short>* x, TensorCPU<short>* y);
template TensorCPU<unsigned short>* add<unsigned short>(TensorCPU<unsigned short>* x, TensorCPU<unsigned short>* y);
template TensorCPU<int>* add<int>(TensorCPU<int>* x, TensorCPU<int>* y);
template TensorCPU<unsigned int>* add<unsigned int>(TensorCPU<unsigned int>* x, TensorCPU<unsigned int>* y);
template TensorCPU<long>* add<long>(TensorCPU<long>* x, TensorCPU<long>* y);
template TensorCPU<unsigned long>* add<unsigned long>(TensorCPU<unsigned long>* x, TensorCPU<unsigned long>* y);
template TensorCPU<long long>* add<long long>(TensorCPU<long long>* x, TensorCPU<long long>* y);
template TensorCPU<unsigned long long>* add<unsigned long long>(TensorCPU<unsigned long long>* x, TensorCPU<unsigned long long>* y);
template TensorCPU<float>* add<float>(TensorCPU<float>* x, TensorCPU<float>* y);
template TensorCPU<double>* add<double>(TensorCPU<double>* x, TensorCPU<double>* y);
template TensorCPU<long double>* add<long double>(TensorCPU<long double>* x, TensorCPU<long double>* y);
template TensorCPU<bool>* mul<bool>(TensorCPU<bool>* x, TensorCPU<bool>* y);
template TensorCPU<short>* mul<short>(TensorCPU<short>* x, TensorCPU<short>* y);
template TensorCPU<unsigned short>* mul<unsigned short>(TensorCPU<unsigned short>* x, TensorCPU<unsigned short>* y);
template TensorCPU<int>* mul<int>(TensorCPU<int>* x, TensorCPU<int>* y);
template TensorCPU<unsigned int>* mul<unsigned int>(TensorCPU<unsigned int>* x, TensorCPU<unsigned int>* y);
template TensorCPU<long>* mul<long>(TensorCPU<long>* x, TensorCPU<long>* y);
template TensorCPU<unsigned long>* mul<unsigned long>(TensorCPU<unsigned long>* x, TensorCPU<unsigned long>* y);
template TensorCPU<long long>* mul<long long>(TensorCPU<long long>* x, TensorCPU<long long>* y);
template TensorCPU<unsigned long long>* mul<unsigned long long>(TensorCPU<unsigned long long>* x, TensorCPU<unsigned long long>* y);
template TensorCPU<float>* mul<float>(TensorCPU<float>* x, TensorCPU<float>* y);
template TensorCPU<double>* mul<double>(TensorCPU<double>* x, TensorCPU<double>* y);
template TensorCPU<long double>* mul<long double>(TensorCPU<long double>* x, TensorCPU<long double>* y);
template TensorCPU<bool>* exp<bool>(TensorCPU<bool>* x);
template TensorCPU<short>* exp<short>(TensorCPU<short>* x);
template TensorCPU<unsigned short>* exp<unsigned short>(TensorCPU<unsigned short>* x);
template TensorCPU<int>* exp<int>(TensorCPU<int>* x);
template TensorCPU<unsigned int>* exp<unsigned int>(TensorCPU<unsigned int>* x);
template TensorCPU<long>* exp<long>(TensorCPU<long>* x);
template TensorCPU<unsigned long>* exp<unsigned long>(TensorCPU<unsigned long>* x);
template TensorCPU<long long>* exp<long long>(TensorCPU<long long>* x);
template TensorCPU<unsigned long long>* exp<unsigned long long>(TensorCPU<unsigned long long>* x);
template TensorCPU<float>* exp<float>(TensorCPU<float>* x);
template TensorCPU<double>* exp<double>(TensorCPU<double>* x);
template TensorCPU<long double>* exp<long double>(TensorCPU<long double>* x);
template void fill<bool>(TensorCPU<bool>* x, bool val);
template void fill<short>(TensorCPU<short>* x, short val);
template void fill<unsigned short>(TensorCPU<unsigned short>* x, unsigned short val);
template void fill<int>(TensorCPU<int>* x, int val);
template void fill<unsigned int>(TensorCPU<unsigned int>* x, unsigned int val);
template void fill<long>(TensorCPU<long>* x, long val);
template void fill<unsigned long>(TensorCPU<unsigned long>* x, unsigned long val);
template void fill<long long>(TensorCPU<long long>* x, long long val);
template void fill<unsigned long long>(TensorCPU<unsigned long long>* x, unsigned long long val);
template void fill<float>(TensorCPU<float>* x, float val);
template void fill<double>(TensorCPU<double>* x, double val);
template void fill<long double>(TensorCPU<long double>* x, long double val);

#endif
