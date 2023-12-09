import numpy as np
from net.layers import Layer, Conv
# from net.layers.activation import Activation
from net.activations.tanh import Tanh
# from ot import sliced_wasserstein_distance

class ConvResBlock(Layer):
    def __init__(self, kernel_size, Activ=Tanh, **kwargs):
        super(ConvResBlock, self).__init__(**kwargs)
        # self.output_size = output_size
        self.kernel_size = kernel_size
        self.filters = self.input_shape[-1]
        self.conv1 = Conv(self.filters, self.kernel_size, padding = 'same', input_shape=self.input_shape, output_shape=self.output_shape, trainable=self.trainable)
        self.conv1.on_input_shape()
        self.conv2 = Conv(self.filters, self.kernel_size, padding = 'same', input_shape=self.input_shape, output_shape=self.output_shape, trainable=self.trainable)
        self.conv2.on_input_shape()
        self.padded_input_shape = (*tuple(np.add(self.input_shape[:2], (2,2))), self.input_shape[-1])
        self.activ1 = Activ(input_shape=self.input_shape, output_shape=self.input_shape)
        self.activ2 = Activ(input_shape=self.input_shape, output_shape=self.input_shape)
                
    def forward(self, input):
        # print(input)
        residual = input[0]
        self.input = input[0]
        out1 = self.conv1.forward(input)
        # transport=np.mean(np.abs(out1 - self.input) ** 2)
        out2 = self.activ1.forward(out1)
        # transport+=np.mean(np.abs(out2 - out1) ** 2)
        out3 = self.conv2.forward(out2)
        # transport+=np.mean(np.abs(out3 - out2) ** 2)
        out4 = out3 + residual
        out5 = self.activ2.forward(out4)
        transport = np.mean(np.abs(out5 - out1)**2)
        # transport=np.mean(np.abs(out5 - out1) ** 2)
        # print('out5: ', out5, out5.shape, '\n\n')
        return out5, transport
    
    def initialize(self, initializer):
        self.conv2.initialize(initializer)
        self.conv1.initialize(initializer)
    
    def update(self, update):
        update = self.activ2.update(update)
        update = self.conv2.update(update)
        update = self.activ1.update(update)
        update = self.conv1.update(update)
