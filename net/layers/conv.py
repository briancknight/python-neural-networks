import numpy as np
from scipy import signal
from net.layers.layer import Layer


class Conv(Layer):
    def __init__(self, filters, kernel_size, padding = None, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        if self.padding == 'same':
            if kernel_size != (3,3):
                raise RuntimeError('kernel_size is not (3,3). General kernel size still requires some more thought, sorry')
            self.old_input_shape = self.input_shape
            self.input_shape = (*tuple(np.add(self.input_shape[:2], (2,2))), self.input_shape[-1])
            self.pad_shape = (1, 1)
        else:
            self.pad_shape = (0,0)
        self.kernel_size = kernel_size
        self.filters = filters
        self.channels = None
        self.kernels_shape = None

    def on_input_shape(self):
        self.channels = self.input_shape[-1]
        self.output_shape = (*tuple(np.subtract(self.input_shape[:2], self.kernel_size) + 1), self.filters)
        self.kernels_shape = (self.filters, *self.kernel_size, self.channels)

    def initialize(self, initializer):
        self.kernels = initializer.get(*self.kernels_shape)
        return [self.kernels_shape]

    def forward(self, input):
        self.input = np.pad(input, (self.pad_shape, self.pad_shape, (0,0)))
        return np.reshape([
            np.sum(
                signal.correlate2d(self.input[:,:,i], self.kernels[k,:,:,i], mode='valid')
                for i in range(self.channels)
            )
            for k in range(self.filters)
        ], self.output_shape)

    def backward(self, output_gradient):
        kernels_gradient = np.reshape([
            [
                signal.correlate2d(self.input[:,:,i], output_gradient[:,:,k], mode='valid')
                for i in range(self.channels)
            ]
            for k in range(self.filters)
        ], self.kernels_shape)

        if self.padding == 'same':
            input_gradient = np.reshape(np.sum([
                [
                    signal.convolve2d(output_gradient[:,:,k], self.kernels[k,:,:,i], mode='same')
                    for i in range(self.channels)
                ]
                for k in range(self.filters)
            ], axis=0), self.old_input_shape)            
        else:
            input_gradient = np.reshape(np.sum([
                [
                    signal.convolve2d(output_gradient[:,:,k], self.kernels[k,:,:,i], mode='full')
                    for i in range(self.channels)
                ]
                for k in range(self.filters)
            ], axis=0), self.input_shape)

        return input_gradient, [kernels_gradient]

    def update(self, updates):
        self.kernels += updates[0]