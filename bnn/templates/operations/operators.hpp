#ifndef BNN_BNN_TEMPLATES_OPERATIONS_OPERATORS_HPP
#define BNN_BNN_TEMPLATES_OPERATIONS_OPERATORS_HPP

template class Operator<bool>;
template class Operator<short>;
template class Operator<unsigned short>;
template class Operator<int>;
template class Operator<unsigned int>;
template class Operator<long>;
template class Operator<unsigned long>;
template class Operator<long long>;
template class Operator<unsigned long long>;
template class Operator<float>;
template class Operator<double>;
template class Operator<long double>;
template class UnaryOperator<bool>;
template class UnaryOperator<short>;
template class UnaryOperator<unsigned short>;
template class UnaryOperator<int>;
template class UnaryOperator<unsigned int>;
template class UnaryOperator<long>;
template class UnaryOperator<unsigned long>;
template class UnaryOperator<long long>;
template class UnaryOperator<unsigned long long>;
template class UnaryOperator<float>;
template class UnaryOperator<double>;
template class UnaryOperator<long double>;
template class BinaryOperator<bool>;
template class BinaryOperator<short>;
template class BinaryOperator<unsigned short>;
template class BinaryOperator<int>;
template class BinaryOperator<unsigned int>;
template class BinaryOperator<long>;
template class BinaryOperator<unsigned long>;
template class BinaryOperator<long long>;
template class BinaryOperator<unsigned long long>;
template class BinaryOperator<float>;
template class BinaryOperator<double>;
template class BinaryOperator<long double>;
template class TensorWrapper<bool>;
template class TensorWrapper<short>;
template class TensorWrapper<unsigned short>;
template class TensorWrapper<int>;
template class TensorWrapper<unsigned int>;
template class TensorWrapper<long>;
template class TensorWrapper<unsigned long>;
template class TensorWrapper<long long>;
template class TensorWrapper<unsigned long long>;
template class TensorWrapper<float>;
template class TensorWrapper<double>;
template class TensorWrapper<long double>;
template class Add<bool>;
template class Add<short>;
template class Add<unsigned short>;
template class Add<int>;
template class Add<unsigned int>;
template class Add<long>;
template class Add<unsigned long>;
template class Add<long long>;
template class Add<unsigned long long>;
template class Add<float>;
template class Add<double>;
template class Add<long double>;
template class Exp<bool>;
template class Exp<short>;
template class Exp<unsigned short>;
template class Exp<int>;
template class Exp<unsigned int>;
template class Exp<long>;
template class Exp<unsigned long>;
template class Exp<long long>;
template class Exp<unsigned long long>;
template class Exp<float>;
template class Exp<double>;
template class Exp<long double>;
template class MatMul<bool>;
template class MatMul<short>;
template class MatMul<unsigned short>;
template class MatMul<int>;
template class MatMul<unsigned int>;
template class MatMul<long>;
template class MatMul<unsigned long>;
template class MatMul<long long>;
template class MatMul<unsigned long long>;
template class MatMul<float>;
template class MatMul<double>;
template class MatMul<long double>;

#endif
