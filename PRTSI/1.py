from graphviz import Digraph

# 创建一个有向图
dot = Digraph()

# 添加节点
dot.node('input', 'Input\n(batch_size, input_channels, sequence_len)')
dot.node('conv1', 'Conv1D\n(kernel_size=5, out_channels=64)')
dot.node('bn1', 'BatchNorm1D\n(64)')
dot.node('relu1', 'ReLU')
dot.node('maxpool1', 'MaxPool1D\n(kernel_size=2)')
dot.node('dropout1', 'Dropout\n(rate=0.5)')
dot.node('conv2', 'Conv1D\n(kernel_size=8, out_channels=128)')
dot.node('bn2', 'BatchNorm1D\n(128)')
dot.node('relu2', 'ReLU')
dot.node('maxpool2', 'MaxPool1D\n(kernel_size=2)')
dot.node('conv3', 'Conv1D\n(kernel_size=8, out_channels=256)')
dot.node('bn3', 'BatchNorm1D\n(256)')
dot.node('relu3', 'ReLU')
dot.node('maxpool3', 'MaxPool1D\n(kernel_size=2)')
dot.node('aap', 'AdaptiveAvgPool1D\n(features_len)')
dot.node('flatten', 'Flatten')
dot.node('output', 'Output\n(batch_size, 256 * features_len)')

# 添加边
dot.edges([
    ('input_conv1', 'conv1_bn1'),
    ('conv1_bn1', 'bn1_relu1'),
    ('bn1_relu1', 'relu1_maxpool1'),
    ('relu1_maxpool1', 'maxpool1_dropout1')
])


# 渲染图形
dot.render('cnn_model', format='png', view=True)