import numpy as np
from mytorch.nn.functional import *

class Activation(object):

    """
    Interface for activation functions (non-linearities).
    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):
        a = np.ones(x.shape)
        out = a*x
        self.autograd_engine.add_operation(inputs=[a,x],
                          output=out,
                          gradients_to_update=[None,None],
                          backward_operation=mul_backward)
        return out

class Sigmoid(Activation):
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):
        a = -1 * np.ones(x.shape)
        out1 = np.multiply(a, x)
        self.autograd_engine.add_operation(inputs=[a, x],
                                           output=out1,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        out2 = np.exp(out1)
        self.autograd_engine.add_operation(inputs=[out1],
                                           output=out2,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)
        b = np.ones(out2.shape)
        out3 = b + out2
        self.autograd_engine.add_operation(inputs=[b, out2],
                                           output=out3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        c = np.ones(out3.shape)
        out4 = np.divide(c, out3)
        self.autograd_engine.add_operation(inputs=[c, out3],
                                           output=out4,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)

        return out4

class Tanh(Activation):
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):
        # -x
        a = -1 * np.ones(x.shape)
        out1 = np.multiply(a, x)
        self.autograd_engine.add_operation(inputs=[a, x],
                                           output=out1,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        # e^-x
        out2 = np.exp(out1)
        self.autograd_engine.add_operation(inputs=[out1],
                                           output=out2,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)
        # e^x
        out3 = np.exp(x)
        self.autograd_engine.add_operation(inputs=[x],
                                           output=out3,
                                           gradients_to_update=[None],
                                           backward_operation=exp_backward)

        # e^x + e^-x
        out4 = out3 - out2
        self.autograd_engine.add_operation(inputs=[out3, out2],
                                           output=out4,
                                           gradients_to_update=[None, None],
                                           backward_operation=sub_backward)

        out5 = out3 + out2
        self.autograd_engine.add_operation(inputs=[out3, out2],
                                           output=out5,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)

        out6 = np.divide(out4, out5)
        self.autograd_engine.add_operation(inputs=[out4, out5],
                                           output=out6,
                                           gradients_to_update=[None, None],
                                           backward_operation=div_backward)
        return out6

class ReLU(Activation):
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):

        mask = x>=0
        out1 = mask*x
        self.autograd_engine.add_operation(inputs=[mask, x],
                          output=out1,
                          gradients_to_update=[None, None],
                          backward_operation=mul_backward)
        return out1