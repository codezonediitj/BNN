#ifndef BNN_BNN_TEMPLATES_AUTODIFF_FORWARD_HPP
#define BNN_BNN_TEMPLATES_AUTODIFF_FORWARD_HPP

template TensorCPU<bool>* compute_gradient_forward<bool>(Operator<bool>* expr, TensorCPU<bool>* var);
template TensorCPU<short>* compute_gradient_forward<short>(Operator<short>* expr, TensorCPU<short>* var);
template TensorCPU<unsigned short>* compute_gradient_forward<unsigned short>(Operator<unsigned short>* expr, TensorCPU<unsigned short>* var);
template TensorCPU<int>* compute_gradient_forward<int>(Operator<int>* expr, TensorCPU<int>* var);
template TensorCPU<unsigned int>* compute_gradient_forward<unsigned int>(Operator<unsigned int>* expr, TensorCPU<unsigned int>* var);
template TensorCPU<long>* compute_gradient_forward<long>(Operator<long>* expr, TensorCPU<long>* var);
template TensorCPU<unsigned long>* compute_gradient_forward<unsigned long>(Operator<unsigned long>* expr, TensorCPU<unsigned long>* var);
template TensorCPU<long long>* compute_gradient_forward<long long>(Operator<long long>* expr, TensorCPU<long long>* var);
template TensorCPU<unsigned long long>* compute_gradient_forward<unsigned long long>(Operator<unsigned long long>* expr, TensorCPU<unsigned long long>* var);
template TensorCPU<float>* compute_gradient_forward<float>(Operator<float>* expr, TensorCPU<float>* var);
template TensorCPU<double>* compute_gradient_forward<double>(Operator<double>* expr, TensorCPU<double>* var);
template TensorCPU<long double>* compute_gradient_forward<long double>(Operator<long double>* expr, TensorCPU<long double>* var);

#endif
