import numpy as np
from mytorch.nn.functional import matmul_backward, add_backward, sub_backward, mul_backward, div_backward, SoftmaxCrossEntropy_backward

class MSELoss:
    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine
        self.loss_val = None

    def __call__(self, y, y_hat):
        self.forward(y, y_hat)

    # TODO: Use your working MSELoss forward and add operations to autograd_engine.
    def forward(self, y, y_hat):
        """
            This class is similar to the wrapper functions for the activations
            that you wrote in functional.py with a couple of key differences:
                1. Notice that instead of passing the autograd object to the forward
                    method, we are instead saving it as a class attribute whenever
                    an MSELoss() object is defined. This is so that we can directly 
                    call the backward() operation on the loss as follows:
                        >>> mse_loss = MSELoss(autograd_object)
                        >>> mse_loss(y, y_hat)
                        >>> mse_loss.backward()

                2. Notice that the class has an attribute called self.loss_val. 
                    You must save the calculated loss value in this variable and 
                    the forward() function is not expected to return any value.
                    This is so that we do not explicitly pass the divergence to 
                    the autograd engine's backward method. Rather, calling backward()
                    on the MSELoss object will take care of that for you.

            Args:
                - y (np.ndarray) : the ground truth,
                - y_hat (np.ndarray) : the output computed by the network,

            Returns:
                - No return required
        """
        #TODO: Use the primitive operations to calculate the MSE Loss
        #      Remember to use add_operation to record these operations in
        #      the autograd engine after each operation
        N = np.array(y.shape[0])
        diff = y - y_hat
        self.autograd_engine.add_operation(
            inputs=[y, y_hat],
            output=diff,
            gradients_to_update=[None, None],
            backward_operation=sub_backward
        )
        squared_diff = diff * diff
        self.autograd_engine.add_operation(
            inputs=[diff, diff],
            output=squared_diff,
            gradients_to_update=[None, None],
            backward_operation=mul_backward
        )

        N = np.broadcast_to(N, squared_diff.shape)
        mse = squared_diff / N

        self.autograd_engine.add_operation(
            inputs=[squared_diff, N],
            output=mse,
            gradients_to_update=[None, None],
            backward_operation=div_backward
        )
        self.loss_val = mse
        return self.loss_val

    def backward(self):
        # You can call autograd's backward here or in the mlp.
        self.autograd_engine.backward(self.loss_val)

# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy:
    def __init__(self, autograd_engine):
        self.loss_val = None
        self.y_grad_placeholder = None
        self.autograd_engine = autograd_engine

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

    def forward(self, y, y_hat):
        """
            Refer to the comments in MSELoss
        """
        max_a = np.max(y_hat)
        max_a = np.ones_like(y_hat) * max_a

        # self.autograd_engine.add_operation(
        #     inputs=[y_hat],
        #     output=max_a,
        #     gradients_to_update=[None],
        #     backward_operation=max_backward
        # )

        ones = np.ones(y_hat.shape)
        a = max_a * ones
        # self.autograd_engine.add_operation(
        #     inputs=[max_a, ones],
        #     output=a,
        #     gradients_to_update=[None, None],
        #     backward_operation=mul_backward
        # )

        sub = y_hat - a
        # self.autograd_engine.add_operation(
        #     inputs=[y_hat, a],
        #     output=sub,
        #     gradients_to_update=[None, None],
        #     backward_operation=sub_backward
        # )

        exp = np.exp(sub)
        # self.autograd_engine.add_operation(
        #     inputs=[sub],
        #     output=exp,
        #     gradients_to_update=[None],
        #     backward_operation=exp_backward
        # )

        exp_sum = exp.sum(1, keepdims=True)
        exp_sum = np.ones_like(exp) * exp_sum
        # self.autograd_engine.add_operation(
        #     inputs=[exp, np.asarray(1)],
        #     output=exp_sum,
        #     gradients_to_update=[None, None],
        #     backward_operation=sum_backward
        # )

        self.softmax = exp / exp_sum
        # self.autograd_engine.add_operation(
        #     inputs=[exp, exp_sum],
        #     output=self.softmax,
        #     gradients_to_update=[None, None],
        #     backward_operation=div_backward
        # )

        log_exp = np.log(exp_sum)
        # self.autograd_engine.add_operation(
        #     inputs=[exp_sum],
        #     output=log_exp,
        #     gradients_to_update=[None],
        #     backward_operation=log_backward
        # )

        inner_sub = sub - log_exp
        # self.autograd_engine.add_operation(
        #     inputs=[sub, log_exp],
        #     output=inner_sub,
        #     gradients_to_update=[None, None],
        #     backward_operation=sub_backward
        # )

        mul = y * inner_sub
        # self.autograd_engine.add_operation(
        #     inputs=[y, inner_sub],
        #     output=mul,
        #     gradients_to_update=[None, None],
        #     backward_operation=mul_backward
        # )

        # this is not being run in the backward causing 0s
        crossentropy = mul.sum(1)
        # self.autograd_engine.add_operation(
        #     inputs=[mul, np.asarray(1)],
        #     output=crossentropy,
        #     gradients_to_update=[None, None],
        #     backward_operation=sum_backward
        # )

        crossentropy = crossentropy * -1
        # self.autograd_engine.add_operation(
        #     inputs=[crossentropy, np.ones_like(crossentropy)*-1],
        #     output=crossentropy,
        #     gradients_to_update=[None, None],
        #     backward_operation=mul_backward
        # )

        self.autograd_engine.add_operation(
            inputs=[y, y_hat],
            output=crossentropy,
            gradients_to_update=[None, None],
            backward_operation=SoftmaxCrossEntropy_backward
        )

        self.loss_val = crossentropy
        return self.loss_val

    def backward(self):
        # You can call autograd's backward here OR in the mlp.
        self.autograd_engine.backward(self.loss_val)
