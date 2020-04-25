#ifndef BNN_BNN_TEMPLATES_AUTODIFF_GRAPH_HPP
#define BNN_BNN_TEMPLATES_AUTODIFF_GRAPH_HPP

template struct GraphNode<bool>;
template struct GraphNode<short>;
template struct GraphNode<unsigned short>;
template struct GraphNode<int>;
template struct GraphNode<unsigned int>;
template struct GraphNode<long>;
template struct GraphNode<unsigned long>;
template struct GraphNode<long long>;
template struct GraphNode<unsigned long long>;
template struct GraphNode<float>;
template struct GraphNode<double>;
template struct GraphNode<long double>;
template GraphNode<bool>* build_graph<bool>(Operator<bool>* expr);
template GraphNode<short>* build_graph<short>(Operator<short>* expr);
template GraphNode<unsigned short>* build_graph<unsigned short>(Operator<unsigned short>* expr);
template GraphNode<int>* build_graph<int>(Operator<int>* expr);
template GraphNode<unsigned int>* build_graph<unsigned int>(Operator<unsigned int>* expr);
template GraphNode<long>* build_graph<long>(Operator<long>* expr);
template GraphNode<unsigned long>* build_graph<unsigned long>(Operator<unsigned long>* expr);
template GraphNode<long long>* build_graph<long long>(Operator<long long>* expr);
template GraphNode<unsigned long long>* build_graph<unsigned long long>(Operator<unsigned long long>* expr);
template GraphNode<float>* build_graph<float>(Operator<float>* expr);
template GraphNode<double>* build_graph<double>(Operator<double>* expr);
template GraphNode<long double>* build_graph<long double>(Operator<long double>* expr);

#endif
