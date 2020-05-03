#ifndef BNN_BNN_TEMPLATES_IO_INPUT_HPP
#define BNN_BNN_TEMPLATES_IO_INPUT_HPP

template TensorCPU<bool>* load_mnist_images<bool>(const string& path);
template TensorCPU<short>* load_mnist_images<short>(const string& path);
template TensorCPU<unsigned short>* load_mnist_images<unsigned short>(const string& path);
template TensorCPU<int>* load_mnist_images<int>(const string& path);
template TensorCPU<unsigned int>* load_mnist_images<unsigned int>(const string& path);
template TensorCPU<long>* load_mnist_images<long>(const string& path);
template TensorCPU<unsigned long>* load_mnist_images<unsigned long>(const string& path);
template TensorCPU<long long>* load_mnist_images<long long>(const string& path);
template TensorCPU<unsigned long long>* load_mnist_images<unsigned long long>(const string& path);
template TensorCPU<float>* load_mnist_images<float>(const string& path);
template TensorCPU<double>* load_mnist_images<double>(const string& path);
template TensorCPU<long double>* load_mnist_images<long double>(const string& path);
template TensorCPU<bool>* load_mnist_labels<bool>(const string& path);
template TensorCPU<short>* load_mnist_labels<short>(const string& path);
template TensorCPU<unsigned short>* load_mnist_labels<unsigned short>(const string& path);
template TensorCPU<int>* load_mnist_labels<int>(const string& path);
template TensorCPU<unsigned int>* load_mnist_labels<unsigned int>(const string& path);
template TensorCPU<long>* load_mnist_labels<long>(const string& path);
template TensorCPU<unsigned long>* load_mnist_labels<unsigned long>(const string& path);
template TensorCPU<long long>* load_mnist_labels<long long>(const string& path);
template TensorCPU<unsigned long long>* load_mnist_labels<unsigned long long>(const string& path);
template TensorCPU<float>* load_mnist_labels<float>(const string& path);
template TensorCPU<double>* load_mnist_labels<double>(const string& path);
template TensorCPU<long double>* load_mnist_labels<long double>(const string& path);

#endif
