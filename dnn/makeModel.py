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

# Data input and type changes
def data_read(train, teach, N):
	# data trasform
	# type: train -> float, teach -> int
	f_train = np.loadtxt(train, delimiter = ',', dtype = np.float32)
	f_teach = np.loadtxt(teach, delimiter = ',', dtype = np.int32)
	# data separates	
	x_train, x_test = np.split(f_train, [N])
	y_train, y_test = np.split(f_teach, [N])

	return x_train, x_test, y_train, y_test

# Model definition
def model_design(inData, hidData1, hidData2, outData):
	model = chainer.FunctionSet(l1=F.Linear(inData, hidData1), l2=F.Linear(hidData1, hidData2), l3=F.Linear(hidData2, outData))
	return model

# Main function
def test_DeepLearning():
	# parameters
	batchsize	= 450
	n_epoch		= 100
	
	input_size	= 900
	h_units1	= 1000
	h_units2	= 300
	output_size	= 40
	
	FIL_train = 'data/main_data/train.csv'
	FIL_teach = 'data/main_data/teach.csv'
	N = 900

	# input file data
	x_train, x_test, y_train, y_test = data_read(FIL_train, FIL_teach, N)
	N_test = y_test.size

	# model setting 
	# 4 layer network
	model = model_design(input_size, h_units1, h_units2, output_size)
	
	# 3 layer network
	# model = model_design(input_size, h_units1, h_units2, h_units3, output_size)

	# opt = optimizers.SGD()
	opt = optimizers.Adam()
	opt.setup(model.collect_parameters())

	# Forward computation
	def forward(x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.relu(model.l1(x)), train=train)
		h2 = F.dropout(F.relu(model.l2(h1)), train=train)
		y = model.l3(h2)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

	# Training 
	# init 
	train_loss	= []
	train_acc	= []
	test_loss	= []
	test_acc	= []

	l1_W	= []
	l2_W	= []
	l3_W	= []

	# Training loop
	for epoch in xrange(1, n_epoch+1):
		print 'epoch', epoch
				
		perm = np.random.permutation(N)
		sum_accuracy = 0
		sum_loss = 0
		for i in xrange(0, N, batchsize):
			x_batch = x_train[perm[i:i+batchsize]]
			y_batch = y_train[perm[i:i+batchsize]]
			opt.zero_grads()
			loss, acc = forward(x_batch, y_batch)
			loss.backward()
			opt.update()

			train_loss.append(loss.data)
			train_acc.append(acc.data)
			sum_loss	 += float(cuda.to_cpu(loss.data)) * batchsize
			sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
			
			print 'train mean loss = {}, accuracy = {}'.format(sum_loss/N, sum_accuracy/N)
																															
			sum_accuracy = 0
			sum_loss	 = 0
			
			for i in xrange(0, N_test, batchsize):
				x_batch = x_test[i:i+batchsize]
				y_batch = y_test[i:i+batchsize]
				loss, acc = forward(x_batch, y_batch, train=False)
				test_loss.append(loss.data)
				test_acc.append(acc.data)
				sum_loss	 += float(cuda.to_cpu(loss.data)) * batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
				
			print 'test mean loss = {}, accuracy = {}'.format(sum_loss/N_test, sum_accuracy/N_test)
			
			l1_W.append(model.l1.W)
			l2_W.append(model.l2.W)
			l3_W.append(model.l3.W)

	for i in xrange(0, N_test, 1):
		x_batch = x_test[i:i+1]
		h1 = F.dropout(F.relu(model.l1(Variable(x_batch))), train = False)
		h2 = F.dropout(F.relu(model.l2(h1)), train = False)
		y = model.l3(h2)
		print 'Output: ' + str(y.data)
	
	plt.figure(figsize = (8, 6))
	plt.plot(range(len(train_acc)), train_acc)
	plt.plot(range(len(test_acc)), test_acc, "r")
	plt.legend(["train_acc", "test_acc"], loc = 4)
	plt.title("Accuracy of digit recognition.")
	plt.show()
	
	plt.figure(figsize = (8, 6))
	plt.plot(range(len(train_loss)), train_loss)
	plt.plot(range(len(test_loss)), test_loss, "r")
	plt.legend(["train_loss", "test_loss"], loc = 1)
	plt.title("Loss of digit recognition.")
	plt.show()
	
	model.to_cpu()
	with open('model.pkl', 'wb') as o:
		pickle.dump(model, o)

if __name__ == "__main__":
	test_DeepLearning()
