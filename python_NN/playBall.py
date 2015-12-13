#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import sys
import csv

def discrimination():
	# Parameters
	batchsize	= 25
	n_epoch		= 10

	input_size	= 900
	hidden_size	= 500
	output_size	= 5
	
	# File name
	F_input = 'data/dis/data1.csv'
	F_l1w_data = 'data/dis/l1w.csv'
	F_l2w_data = 'data/dis/l2w.csv'
	F_l3w_data = 'data/dis/l3w.csv'

	f_in = np.loadtxt(F_input, delimiter=',', dtype=np.float32)
	
	# print("f_in: "), str(f_in)

	f_l1w = np.loadtxt(F_l1w_data, delimiter=',', dtype=np.float32)
	f_l2w = np.loadtxt(F_l2w_data, delimiter=',', dtype=np.float32)
	f_l3w = np.loadtxt(F_l3w_data, delimiter=',', dtype=np.float32)
	
	model = FunctionSet(l1=F.Linear(input_size, hidden_size),
						l2=F.Linear(hidden_size, hidden_size),
						l3=F.Linear(hidden_size, output_size))
	
	# Set parameters
	"""
	l1_W = []
	l2_W = []
	l3_W = []

	l1_W.append(f_l1w)
	l2_W.append(f_l2w)
	l3_W.append(f_l3w)
	"""

	model.l1.W = np.array(f_l1w)
	model.l2.W = np.array(f_l2w)
	model.l3.W = np.array(f_l3w)

	x_batch = f_in
	h1 = F.dropout(F.relu(model.l1(Variable(x_batch))), train=False)
	h2 = F.dropout(F.relu(model.l2(h1)), train=False)
	y = model.l3(h2)
	print 'Output: ' + str(y.data)


if __name__ == "__main__":
	discrimination()

"""
# 学習に用いるデータ数
N = 25
# データをそれぞれ学習に使うやつとテストに使うやつにわけるよ
# 上で定義したN個は，全データが100データあった場合その中のNデータを
# 学習に使い，残りのデータを評価用に使う．
# 用意できたデータ数と相談して決めると良い
x_train, x_test = np.split(f_train, [N])
y_train, y_test = np.split(f_teach, [N])
N_test = y_test.size

# モデル設定
# 層を増やしたいやんちゃボーイはここをいじるといいよぉ
# *.Linear は全結合NNのことです（活性化関数はここじゃない）
model = FunctionSet(l1=F.Linear(input_size, hidden_size),
					l2=F.Linear(hidden_size, hidden_size),
					l3=F.Linear(hidden_size, output_size))


# ニューラルネットワークの構造
# ドロップアウト使う
# 活性化関数にReLU関数(F.sigmoidもある)
def forward(x_data, y_data, train=True):
	x, t = Variable(x_data), Variable(y_data)
	h1 = F.dropout(F.relu(model.l1(x)),  train=train)
	h2 = F.dropout(F.relu(model.l2(h1)), train=train)
	y = model.l3(h2)

	return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# 最適化する　アダムとかいうやつ
# アダムは適切なパラメタが決まっているから簡単
# そのかわり乱数でズラしまくって最適解を見つける欲張りアダム
optimizer = optimizers.Adam()
optimizer.setup(model)

# 初期化
train_loss 	= []
train_acc	= []
test_loss	= []
test_acc	= []

l1_W = []
l2_W = []
l3_W = []

# イテレーション
for epoch in xrange(1, n_epoch+1):
	print 'epoch', epoch

	# トレーニングデータをいじる
	# データ順をバラバラにして局所解に陥る可能性を少なくする
	perm = np.random.permutation(N)
	sum_accuracy = 0
	sum_loss = 0
	for i in xrange(0, N, batchsize):
		x_batch = x_train[perm[i:i+batchsize]]	#  バラバラにしたデータのiからバッチサイズまで
		y_batch = y_train[perm[i:i+batchsize]]


		optimizer.zero_grads()
		loss, acc = forward(x_batch, y_batch)

		loss.backward()
		optimizer.update()

		train_loss.append(loss.data)
		train_acc.append(acc.data)
		sum_loss	 += float(cuda.to_cpu(loss.data)) * batchsize
		sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

	# まとめれ表示
	print 'train mean loss = {}, accuracy = {}'.format(sum_loss / N, sum_accuracy / N)


	# 評価するのん
	# 現段階での学習状況で評価値がどんなもんかって感じ
	sum_accuracy = 0
	sum_loss	 = 0
	for i in xrange(0, N_test, batchsize):
		x_batch = x_test[i:i+batchsize]
		y_batch = y_test[i:i+batchsize]

		loss, acc = forward(x_batch, y_batch, train=False)	# 学習はfalseに

		test_loss.append(loss.data)
		test_acc.append(acc.data)
		sum_loss	 += float(cuda.to_cpu(loss.data)) * batchsize
		sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

	print 'test mean loss = {}, accuracy = {}'.format(sum_loss/N_test, sum_accuracy/N_test)

	# ここまでのパラメタを保存する
	l1_W.append(model.l1.W)
	l2_W.append(model.l2.W)
	l3_W.append(model.l3.W)

# 重みのデータ
print("out1: "), str(model.l1.W)
print("out2: "), str(model.l2.W)
print("out3: "), str(model.l3.W)

# Save data
with open('write.csv', 'w') as f:
	writer = csv.writer(f, lineterminator=',')
	writer.writerows(model.l3.W)
#	writer.writerows(array2d)
#
#・最後に全ての出力層の出力をデータごとに表示する
#・グラフにプロットして分かりやすくする(acc, loss)
#

for i in xrange(0, N_test, 1):
	x_batch = x_test[i:i+1]
	h1 = F.dropout(F.relu(model.l1(Variable(x_batch))), train=False)
	h2 = F.dropout(F.relu(model.l2(h1)), train=False)
	y = model.l3(h2)
	print 'Output: ' + str(y.data)

# グラフに表示(acc)
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc, "r")
plt.legend(["train_acc", "test_acc"], loc=4)
plt.title("Accuracy of digit recognition.")
plt.show()

# グラフに表示(loss)
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_loss)), train_loss)
plt.plot(range(len(test_loss)), test_loss, "r")
plt.legend(["train_loss", "test_loss"], loc=1)
plt.title("Loss of digit recognition.")
plt.show()
"""
