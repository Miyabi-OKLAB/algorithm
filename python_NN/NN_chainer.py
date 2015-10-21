#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
chainerのホームページ: 
http://docs.chainer.org/en/stable/tutorial/basic.html

import周りは公式でいれろって書いてあったやつ
あと必要なやつ

以下のサイトで詳しく説明されている(そこからコードをコピペ): 
http://qiita.com/kenmatsu4/items/7b8d24d4c5144a686412

結果表示と読み込みの部分をいじった

fetch_mldata はwebからデータを取得する

追記

どうやら，input_sizeは256が限界っぽい・・・

ヘルプみー
上記の参考サイトでは784次元を入力しているが以下のコードでの900次元はエラーが出る

まぁ，一応力ずくで256でやっても認識率は問題ない
"""
import matplotlib.pyplot as plt 	# グラフ出したりする時に
from sklearn.datasets import fetch_mldata	# データ入力の時使う
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import sys


# ミニバッチのサイズを定義
# データサイズに合わせて調整してあげるといいよ！
batchsize	= 2

# 学習の繰り返し回数
# ミニバッチとこの回数のため，
# 最終的なイテレーションはミニバッチxエポック
n_epoch		= 20

# モデルの各素子数
input_size	= 900
hidden_size	= 1000
output_size	= 5

# インプットするデータを設定
# インプットするデータを用意

# データファイルの名前を設定
print 'input file datasets'
FILENAME_train = 'data/testdata03/train/train.csv'
FILENAME_teach = 'data/testdata03/teach/teach.csv'

# テスト↓
# 2と7の２値データ.txtファイルを各5データずつ
# 学習データ ( -> float)
f_train = np.loadtxt(FILENAME_train, delimiter = ',', dtype = np.float32)
# f_train /= 255

# 教師データ ( -> int)
f_teach = np.loadtxt(FILENAME_teach, delimiter = ',', dtype = np.int32)

# 学習に用いるデータ数
N = 50
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
model = FunctionSet(l1 = F.Linear(input_size, hidden_size),
					l2 = F.Linear(hidden_size, hidden_size),
					l3 = F.Linear(hidden_size, output_size))


# ニューラルネットワークの構造
# ドロップアウト使う
# 活性化関数にReLU関数(F.sigmoidもある)
# 256の件ここら辺怪しい
def forward(x_data, y_data, train=True):
	x, t = Variable(x_data), Variable(y_data)
	h1 = F.dropout(F.relu(model.l1(x)),  train = train)
	h2 = F.dropout(F.relu(model.l2(h1)), train = train)
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

	print 'test mean loss = {}, accuracy = {}'.format(sum_loss / N_test, sum_accuracy / N_test)

	# ここまでのパラメタを保存する
	l1_W.append(model.l1.W)
	l2_W.append(model.l2.W)
	l3_W.append(model.l3.W)

#
#・最後に全ての出力層の出力をデータごとに表示する
#・グラフにプロットして分かりやすくする(acc, loss)
#

for i in xrange(0, N_test, 1):
	x_batch = x_test[i:i+1]
	h1 = F.dropout(F.relu(model.l1(Variable(x_batch))), train = False)
	h2 = F.dropout(F.relu(model.l2(h1)), train = False)
	y = model.l3(h2)
	print 'Output: ' + str(y.data)

# グラフに表示(acc)
plt.figure(figsize = (8, 6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc, "r")
plt.legend(["train_acc", "test_acc"], loc = 4)
plt.title("Accuracy of digit recognition.")
plt.show()

# グラフに表示(loss)
plt.figure(figsize = (8, 6))
plt.plot(range(len(train_loss)), train_loss)
plt.plot(range(len(test_loss)), test_loss, "r")
plt.legend(["train_loss", "test_loss"], loc = 1)
plt.title("Loss of digit recognition.")
plt.show()
