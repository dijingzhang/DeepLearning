import numpy as np
from mytorch.utils import MemoryBuffer

class Operation:
    def __init__(self, inputs, output, gradients_to_update, backward_operation):
        '''
            Args:
                - inputs: operation inputs (numpy.ndarray)
                - outputs: operation output (numpy.ndarray)
                - gradients_to_update: parameter gradients if for parameter of ,
                            network or None (numpy.ndarray, None)
                - backward_operation: backward function for nn/functional.py. 
                            When passing a function you don't need inputs or parentheses.

            Note: You do not need to modify anything here
        '''
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation


class Autograd():
    def __init__(self):
        """
        WARNING: DO NOT MODIFY THIS METHOD!
        A check to make sure you don't create more than 1 Autograd at a time. You can remove
        this if you want to do multiple in parallel. We do not recommend this
        """
        if getattr(self.__class__, '_has_instance', False):
            raise RuntimeError('Cannot create more than 1 Autograd instance')
        self.__class__._has_instance = True

        self.memory_buffer = MemoryBuffer()
        self.operation_list = []

    def __del__(self):
        """
            WARNING: DO NOT MODIFY THIS METHOD!
            Class destructor. We use this for testing purposes.
        """
        del self.memory_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(self, inputs, output, gradients_to_update, backward_operation):
        '''
            Adds operation to operation list and puts gradients in memory buffer for tracking

            Args:
                - inputs: operation inputs (numpy.ndarray)
                - outputs: operation output (numpy.ndarray)
                - gradients_to_update: parameter gradients if for parameter of
                            network or None (numpy.ndarray, None)
                    NOTE: Given the linear layer as shown in the writeup section
                        2.4 there are 2 kinds of inputs to an operation:
                        1) one that requires gradients to be internally tracked
                            ex. input (X) to a layer
                        2) one that requires gradient to be externally tracked
                            ex. weight matrix (W) of a layer (so we can track dW) 
                - backward_operation: backward function for nn/functional.py. 
                            When passing a function you don't need inputs or parentheses.

            Returns:
                No return required     
        '''
        if len(inputs) != len(gradients_to_update):
            raise Exception("Number of inputs must match the number of gradients to update!")
        
        # TODO: Add all of the inputs to the self.memory_buffer using the add_spot() function
        # This will allow the gradients to be tracked
        for i in range(len(inputs)):
            self.memory_buffer.add_spot(inputs[i])

        # TODO: Append an Operation object to the self.operation_list
        self.operation_list.append(Operation(inputs, output, gradients_to_update, backward_operation))

    def backward(self, divergence):
        """
            The backpropagation through the self.operation_list with a given divergence.
            This function should automatically update gradients of parameters by checking
            the gradients_to_update. Read the write up for further explanation

            Args:
                - divergence: loss value (float/double/int/long)

            Returns:
                No return required
        """
        # TODO: Iterate through the self.operation_list and propagate the gradients.
        # NOTE: Make sure you iterate in the correct direction. How are gradients propagated?
        for i, operation in reversed(list(enumerate(self.operation_list))):
            # TODO: For the first iteration set the gradient to be propagated equal to the divergence.
            #       For the remaining iterations the gradient to be propagated can be retrieved from the
            #       self.memory_buffer.get_param.
            if i == len(self.operation_list) - 1:
                grad_of_output = divergence
            elif not self.memory_buffer.is_in_memory(operation.output):
                # Note that this is an interesting case, where we have a "terminating" node
                # that does not feed into any other node but one that we do not call backward
                # on. For visualization, consider a single initial node branching out to two
                # nodes, and we only call backward on the first one. This deals with the
                # second one.
                continue
            else:
                grad_of_output = self.memory_buffer.get_param(operation.output)

            # TODO: Execute the backward for the Operation
            # NOTE: Make sure to unroll the inputs list if you aren't parsing a list in your backward.
            grad_of_input = operation.backward_operation(grad_of_output, *operation.inputs)
            
            # TODO: Loop through the inputs and their gradients and add them to the gradient in the
            #       self.memory_buffer. Check with the Operation's gradients_to_update if you need to
            #       directly update a gradient.
            for input, grad_of_input, gradient_to_update in zip(operation.inputs,
                                                                 grad_of_input,
                                                                 operation.gradients_to_update):
                self.memory_buffer.update_param(input, grad_of_input)
                if gradient_to_update is not None:
                    gradient_to_update += grad_of_input

    def zero_grad(self):
        self.memory_buffer.clear()
        self.operation_list = []
