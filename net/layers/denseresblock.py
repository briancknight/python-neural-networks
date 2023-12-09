import numpy as np
from net.layers import Layer, Dense
from net.activations.tanh import Tanh

class DenseResBlock(Layer):
    def __init__(self, output_size, Activ=Tanh, **kwargs):
        super(DenseResBlock, self).__init__(input_shape = (1, output_size), output_shape=(1, output_size), **kwargs)
        self.output_size = output_size
        self.dense1 = Dense(self.output_size, input_shape=self.input_shape, trainable=self.trainable)
        self.dense2 = Dense(self.output_size, input_shape=self.input_shape, trainable=self.trainable)

        self.activ1 = Activ(input_shape=self.input_shape, output_shape=self.output_shape)
        self.activ2 = Activ(input_shape=self.input_shape, output_shape=self.output_shape)
                
    def forward(self, input):
        # print(input)
        residual = input[0]
        self.input = input[0]
        out1 = self.dense1.forward(input, flag=True)
        # transport=np.mean(np.abs(out1 - self.input) ** 2)
        out2 = self.activ1.forward(out1)
        # transport+=np.mean(np.abs(out2 - out1) ** 2)
        out3 = self.dense2.forward(out2, flag=True)
        # transport+=np.mean(np.abs(out3 - out2) ** 2)
        out4 = out3 + residual
        out5 = self.activ2.forward(out4)
        transport=np.mean(np.abs(out5 - out1) ** 2)
        # print('out5: ', out5, out5.shape, '\n\n')
        return out5, transport
    
    def initialize(self, initializer):
        self.dense2.initialize(initializer)
        self.dense1.initialize(initializer)
    
    
    def update(self, update):
        update = self.activ2.update(update)
        update = self.dense2.update(update)
        update = self.activ1.update(update)
        update = self.dense1.update(update)