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
template void _rr_scheduler<bool>(GraphNode<bool>* layer, op_queue<bool>* jobs[][2], unsigned threads);
template void _rr_scheduler<short>(GraphNode<short>* layer, op_queue<short>* jobs[][2], unsigned threads);
template void _rr_scheduler<unsigned short>(GraphNode<unsigned short>* layer, op_queue<unsigned short>* jobs[][2], unsigned threads);
template void _rr_scheduler<int>(GraphNode<int>* layer, op_queue<int>* jobs[][2], unsigned threads);
template void _rr_scheduler<unsigned int>(GraphNode<unsigned int>* layer, op_queue<unsigned int>* jobs[][2], unsigned threads);
template void _rr_scheduler<long>(GraphNode<long>* layer, op_queue<long>* jobs[][2], unsigned threads);
template void _rr_scheduler<unsigned long>(GraphNode<unsigned long>* layer, op_queue<unsigned long>* jobs[][2], unsigned threads);
template void _rr_scheduler<long long>(GraphNode<long long>* layer, op_queue<long long>* jobs[][2], unsigned threads);
template void _rr_scheduler<unsigned long long>(GraphNode<unsigned long long>* layer, op_queue<unsigned long long>* jobs[][2], unsigned threads);
template void _rr_scheduler<float>(GraphNode<float>* layer, op_queue<float>* jobs[][2], unsigned threads);
template void _rr_scheduler<double>(GraphNode<double>* layer, op_queue<double>* jobs[][2], unsigned threads);
template void _rr_scheduler<long double>(GraphNode<long double>* layer, op_queue<long double>* jobs[][2], unsigned threads);
template void _clear_jobs<bool>(thread* pool[], op_queue<bool>* jobs[][2],unsigned threads);
template void _clear_jobs<short>(thread* pool[], op_queue<short>* jobs[][2],unsigned threads);
template void _clear_jobs<unsigned short>(thread* pool[], op_queue<unsigned short>* jobs[][2],unsigned threads);
template void _clear_jobs<int>(thread* pool[], op_queue<int>* jobs[][2],unsigned threads);
template void _clear_jobs<unsigned int>(thread* pool[], op_queue<unsigned int>* jobs[][2],unsigned threads);
template void _clear_jobs<long>(thread* pool[], op_queue<long>* jobs[][2],unsigned threads);
template void _clear_jobs<unsigned long>(thread* pool[], op_queue<unsigned long>* jobs[][2],unsigned threads);
template void _clear_jobs<long long>(thread* pool[], op_queue<long long>* jobs[][2],unsigned threads);
template void _clear_jobs<unsigned long long>(thread* pool[], op_queue<unsigned long long>* jobs[][2],unsigned threads);
template void _clear_jobs<float>(thread* pool[], op_queue<float>* jobs[][2],unsigned threads);
template void _clear_jobs<double>(thread* pool[], op_queue<double>* jobs[][2],unsigned threads);
template void _clear_jobs<long double>(thread* pool[], op_queue<long double>* jobs[][2],unsigned threads);

#endif
