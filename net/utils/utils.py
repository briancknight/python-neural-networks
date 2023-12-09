import numpy as np
from net.optimizers import Optimizer
from net.layers import DenseResBlock, ConvResBlock

def create_model(network, initializer, OptimizerBaseClass, optimizerArgs={}):
    # set input_shape & output_shape
    
    # for i, layer in network:
    #     if isinstance(layer, DenseResBlock):
    #         network.insert(i+1, DeadLayer)
    #         network.insert(i+2, DeadLayer)
    #         network.insert(i+3, DeadLayer)
            
    for i, layer in enumerate(network):

        if not layer.input_shape:
            layer.input_shape = network[i - 1].output_shape
        layer.on_input_shape()
        if not layer.output_shape:
            layer.output_shape = layer.input_shape

    # initialize layers & create one optimizer per layer
    layer_shapes = [(layer.input_shape, layer.output_shape) for layer in network]
    initializer.set_layer_shapes(layer_shapes)
    optimizers = []
    for i, layer in enumerate(network):
        initializer.set_layer_index(i)
        if isinstance(layer, DenseResBlock):
            denseresblock_optimize(layer, initializer, optimizers, OptimizerBaseClass, optimizerArgs)
        if isinstance(layer, ConvResBlock):
            convresblock_optimize(layer, initializer, optimizers, OptimizerBaseClass, optimizerArgs)
        else:
            param_shapes = layer.initialize(initializer)
            optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes) if layer.trainable else None)

    # return list of (layer, optimizer)
    return list(zip(network, optimizers))

def summary(model):
    for layer, _ in model:
        print(layer.input_shape, '\t', layer.output_shape)

def forward(model, input):
    output = input
    transport=0
    for layer, _ in model:
        if isinstance(layer, DenseResBlock) or isinstance(layer, ConvResBlock):
            (output, transport) = layer.forward(output)
        else:
            output = layer.forward(output)
    return output, transport

def backward(model, output):
    error = output
    for layer, optimizer in reversed(model):
        if isinstance(layer, DenseResBlock):
            denseresblock_backward(layer, optimizer, error)
        if isinstance(layer, ConvResBlock):
            convresblock_backward(layer, optimizer, error)
        else:
            error, gradients = layer.backward(error)
            if layer.trainable:
                optimizer.set_gradients(gradients)
    return error

def update(model, iteration):
    for layer, optimizer in model:
        if isinstance(layer, DenseResBlock):
            denseresblock_update(layer, optimizer, iteration)
        if isinstance(layer, ConvResBlock):
            convresblock_update(layer, optimizer, iteration)
        else:
            if layer.trainable:
                layer.update(optimizer.get_gradients(iteration))

def train(model, loss, x_train, y_train, epochs, batch=1, lmbda = 1):
    train_set_size = len(x_train)
    for epoch in range(1, epochs + 1):
        error = 0
        for x, y in zip(x_train, y_train):
            output, transport = forward(model, x)
            error += loss.call(y, output) + lmbda * transport
            backward(model, loss.prime(y, output))
            if epoch % batch == 0:
                update(model, epoch)
        error /= train_set_size
        print('%d/%d, error=%f' % (epoch, epochs, error))

def test(model, loss, x_test, y_test, lmbda = 1):
    error = 0
    for x, y in zip(x_test, y_test):
        (output, transport) = forward(model, x)
        error += loss.call(y, output[0]) + lmbda * transport
    error /= len(x_test)
    return error


def denseresblock_optimize(layer, initializer, optimizers, OptimizerBaseClass, optimizerArgs):

    param_shapes1 = layer.dense1.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+1)
    param_shapes2 = layer.activ1.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+2)
    param_shapes3 = layer.dense2.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+3)
    param_shapes4 = layer.activ2.initialize(initializer)
    optimizers.append(
        [Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes1), None,
        Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes3), None]
    )
    
def convresblock_optimize(layer, initializer, optimizers, OptimizerBaseClass, optimizerArgs):

    param_shapes1 = layer.conv1.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+1)
    param_shapes2 = layer.activ1.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+2)
    param_shapes3 = layer.conv2.initialize(initializer)
    # optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes))
    # initializer.set_layer_index(i+3)
    param_shapes4 = layer.activ2.initialize(initializer)
    optimizers.append(
        [Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes1), None,
        Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes3), None]
    )
    
def denseresblock_backward(layer, optimizer, error):
    in_error = error
    error, gradients = layer.activ2.backward(error)
    # optimizer[0].set_gradients(gradients)
    error, gradients = layer.dense2.backward(error,flag=True)
    optimizer[2].set_gradients(gradients)
    error, gradients = layer.activ1.backward(error)
    # optimizer[2].set_gradients(gradients)
    error, gradients = layer.dense1.backward(error,flag=True)
    optimizer[0].set_gradients(gradients)
    error += in_error
    
def convresblock_backward(layer, optimizer, error):
    in_error = error
    error, gradients = layer.activ2.backward(error)
    # error -= np.ones(error.shape)
    # optimizer[0].set_gradients(gradients)
    error, gradients = layer.conv2.backward(error)
    optimizer[2].set_gradients(gradients)
    error, gradients = layer.activ1.backward(error)
    # optimizer[2].set_gradients(gradients)
    error, gradients = layer.conv1.backward(error)
    optimizer[0].set_gradients(gradients)
    error += in_error
    
def denseresblock_update(layer, optimizer, iteration):
    layer.dense1.update(optimizer[0].get_gradients(iteration))
    # layer.activ1.update(optimizer[1](iteration))
    layer.dense2.update(optimizer[2].get_gradients(iteration))
    # layer.activ1.update(optimizer[3](iteration))
    
def convresblock_update(layer, optimizer, iteration):
    layer.conv1.update(optimizer[0].get_gradients(iteration))
    # layer.activ1.update(optimizer[1](iteration))
    layer.conv2.update(optimizer[2].get_gradients(iteration))
    # layer.activ1.update(optimizer[3](iteration))