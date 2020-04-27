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
template TensorCPU<bool>** compute_gradient_forward<bool>(Operator<bool>* expr, TensorCPU<bool>** vars, unsigned num_vars);
template TensorCPU<short>** compute_gradient_forward<short>(Operator<short>* expr, TensorCPU<short>** vars, unsigned num_vars);
template TensorCPU<unsigned short>** compute_gradient_forward<unsigned short>(Operator<unsigned short>* expr, TensorCPU<unsigned short>** vars, unsigned num_vars);
template TensorCPU<int>** compute_gradient_forward<int>(Operator<int>* expr, TensorCPU<int>** vars, unsigned num_vars);
template TensorCPU<unsigned int>** compute_gradient_forward<unsigned int>(Operator<unsigned int>* expr, TensorCPU<unsigned int>** vars, unsigned num_vars);
template TensorCPU<long>** compute_gradient_forward<long>(Operator<long>* expr, TensorCPU<long>** vars, unsigned num_vars);
template TensorCPU<unsigned long>** compute_gradient_forward<unsigned long>(Operator<unsigned long>* expr, TensorCPU<unsigned long>** vars, unsigned num_vars);
template TensorCPU<long long>** compute_gradient_forward<long long>(Operator<long long>* expr, TensorCPU<long long>** vars, unsigned num_vars);
template TensorCPU<unsigned long long>** compute_gradient_forward<unsigned long long>(Operator<unsigned long long>* expr, TensorCPU<unsigned long long>** vars, unsigned num_vars);
template TensorCPU<float>** compute_gradient_forward<float>(Operator<float>* expr, TensorCPU<float>** vars, unsigned num_vars);
template TensorCPU<double>** compute_gradient_forward<double>(Operator<double>* expr, TensorCPU<double>** vars, unsigned num_vars);
template TensorCPU<long double>** compute_gradient_forward<long double>(Operator<long double>* expr, TensorCPU<long double>** vars, unsigned num_vars);
template TensorCPU<bool>* compute_value<bool>(Operator<bool>* expr);
template TensorCPU<short>* compute_value<short>(Operator<short>* expr);
template TensorCPU<unsigned short>* compute_value<unsigned short>(Operator<unsigned short>* expr);
template TensorCPU<int>* compute_value<int>(Operator<int>* expr);
template TensorCPU<unsigned int>* compute_value<unsigned int>(Operator<unsigned int>* expr);
template TensorCPU<long>* compute_value<long>(Operator<long>* expr);
template TensorCPU<unsigned long>* compute_value<unsigned long>(Operator<unsigned long>* expr);
template TensorCPU<long long>* compute_value<long long>(Operator<long long>* expr);
template TensorCPU<unsigned long long>* compute_value<unsigned long long>(Operator<unsigned long long>* expr);
template TensorCPU<float>* compute_value<float>(Operator<float>* expr);
template TensorCPU<double>* compute_value<double>(Operator<double>* expr);
template TensorCPU<long double>* compute_value<long double>(Operator<long double>* expr);

#endif
