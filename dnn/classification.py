#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

作製日：2015/12/07
更新日：2015/12/07
作成者：松本浩幸

"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import matplotlib.pyplot as plt
import sys
import six
import six.moves.cPickle as pickle

def discrimination():
	# parameters
	input_size	= 900
	h_units1	= 1000
	h_units2	= 300
	output_size	= 40
	
	in_data = 'input.csv'
	in_data = np.loadtxt(in_data, delimiter=',', dtype=np.float32)
	in_data2d = in_data[np.newaxis, :]

	with open('model.pkl', 'rb') as i:
		model = pickle.load(i)

	in_data = in_data2d[0:1]
	h1 = F.dropout(F.relu(model.l1(Variable(in_data))), train=False)
	h2 = F.dropout(F.relu(model.l2(h1)), train=False)
	y = model.l3(h2)
	
	out_class = y.data
	out_class_max = max(out_class)

	#print 'Output: ' + str(out_class)
	#print 'Max index: ' + str(np.argmax(out_class))
	
	f = open('output.txt', 'w')
	for i in range(output_size):
		if i == np.argmax(out_class):
			f.writelines('1')
		else:
			f.writelines('0')
	f.close()

if __name__ == "__main__":
	discrimination()
