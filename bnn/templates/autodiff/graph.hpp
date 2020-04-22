#ifndef BNN_BNN_TEMPLATES_AUTODIFF_GRAPH_HPP
#define BNN_BNN_TEMPLATES_AUTODIFF_GRAPH_HPP

template struct ForwardGraphNode<bool>;
template struct ForwardGraphNode<short>;
template struct ForwardGraphNode<unsigned short>;
template struct ForwardGraphNode<int>;
template struct ForwardGraphNode<unsigned int>;
template struct ForwardGraphNode<long>;
template struct ForwardGraphNode<unsigned long>;
template struct ForwardGraphNode<long long>;
template struct ForwardGraphNode<unsigned long long>;
template struct ForwardGraphNode<float>;
template struct ForwardGraphNode<double>;
template struct ForwardGraphNode<long double>;
template ForwardGraphNode<bool>* build_graph_forward<bool>(Operator<bool>* expr);
template ForwardGraphNode<short>* build_graph_forward<short>(Operator<short>* expr);
template ForwardGraphNode<unsigned short>* build_graph_forward<unsigned short>(Operator<unsigned short>* expr);
template ForwardGraphNode<int>* build_graph_forward<int>(Operator<int>* expr);
template ForwardGraphNode<unsigned int>* build_graph_forward<unsigned int>(Operator<unsigned int>* expr);
template ForwardGraphNode<long>* build_graph_forward<long>(Operator<long>* expr);
template ForwardGraphNode<unsigned long>* build_graph_forward<unsigned long>(Operator<unsigned long>* expr);
template ForwardGraphNode<long long>* build_graph_forward<long long>(Operator<long long>* expr);
template ForwardGraphNode<unsigned long long>* build_graph_forward<unsigned long long>(Operator<unsigned long long>* expr);
template ForwardGraphNode<float>* build_graph_forward<float>(Operator<float>* expr);
template ForwardGraphNode<double>* build_graph_forward<double>(Operator<double>* expr);
template ForwardGraphNode<long double>* build_graph_forward<long double>(Operator<long double>* expr);

#endif
