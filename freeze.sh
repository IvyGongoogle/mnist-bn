python freeze_graph.py \
    --input_graph output/graph_def_bn.pb \
    --input_checkpoint ./checkpoints_bn/mnist-conv-slim \
    --output_graph output/mnist_bn.pb \
    --output_node_names fc1/Relu

# python freeze_graph.py \
#     --input_graph output/graph_def_bn_layer.pb \
#     --input_checkpoint ./checkpoints_bn_layer/mnist-conv-slim \
#     --output_graph output/mnist_bn_layer.pb \
#     --output_node_names fc1/Relu
