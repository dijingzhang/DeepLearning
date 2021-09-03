import numpy as np

class Dropout(object):
	def __init__(self, p=0.5):
		# Dropout probability
		self.p = p
		self.mask = None
	def __call__(self, x):
		return self.forward(x)

	def forward(self, x, train = True):
		# 1) Get and apply a mask generated from np.random.binomial
		# 2) Scale your output accordingly
		# 3) During test time, you should not apply any mask or scaling.
		mask = np.random.binomial(1, self.p, x.shape)
		self.mask = mask
		if train == True:
			x *= mask
			x /= self.p
		return x
	def backward(self, delta):
		# 1) This method is only called during trianing.
		return delta * self.mask
