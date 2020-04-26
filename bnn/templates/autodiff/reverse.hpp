#ifndef BNN_BNN_TEMPLATES_AUTODIFF_REVERSE_HPP
#define BNN_BNN_TEMPLATES_AUTODIFF_REVERSE_HPP

template TensorCPU<bool>** compute_gradient_reverse<bool>(Operator<bool>* expr, TensorCPU<bool>** vars, unsigned num_vars);
template TensorCPU<short>** compute_gradient_reverse<short>(Operator<short>* expr, TensorCPU<short>** vars, unsigned num_vars);
template TensorCPU<unsigned short>** compute_gradient_reverse<unsigned short>(Operator<unsigned short>* expr, TensorCPU<unsigned short>** vars, unsigned num_vars);
template TensorCPU<int>** compute_gradient_reverse<int>(Operator<int>* expr, TensorCPU<int>** vars, unsigned num_vars);
template TensorCPU<unsigned int>** compute_gradient_reverse<unsigned int>(Operator<unsigned int>* expr, TensorCPU<unsigned int>** vars, unsigned num_vars);
template TensorCPU<long>** compute_gradient_reverse<long>(Operator<long>* expr, TensorCPU<long>** vars, unsigned num_vars);
template TensorCPU<unsigned long>** compute_gradient_reverse<unsigned long>(Operator<unsigned long>* expr, TensorCPU<unsigned long>** vars, unsigned num_vars);
template TensorCPU<long long>** compute_gradient_reverse<long long>(Operator<long long>* expr, TensorCPU<long long>** vars, unsigned num_vars);
template TensorCPU<unsigned long long>** compute_gradient_reverse<unsigned long long>(Operator<unsigned long long>* expr, TensorCPU<unsigned long long>** vars, unsigned num_vars);
template TensorCPU<float>** compute_gradient_reverse<float>(Operator<float>* expr, TensorCPU<float>** vars, unsigned num_vars);
template TensorCPU<double>** compute_gradient_reverse<double>(Operator<double>* expr, TensorCPU<double>** vars, unsigned num_vars);
template TensorCPU<long double>** compute_gradient_reverse<long double>(Operator<long double>* expr, TensorCPU<long double>** vars, unsigned num_vars);

#endif
