# algorithm
- 雅@OKLAB でのアルゴリズム班の開発リポジトリ
- Neural Network の作成

## Documentation
### Neuron.java
ニューロン一個の振る舞いをするクラス

利用したい場合は、以下のように、インスタンス化する。
```java
ArrayList<Integer> id = new ArrayList<Intger>();
id.add(0);  id.add(0);
int inputSize = 3;

Neuron neuron = new Neuron(id, inputSize);
```
すると、idは[0, 0]で要素数3のベクトルを入力として受け付けられるNeuronインスタンスが作成できる。

たとえば、このneuronにベクトル[1, 0, 1]を入力したい場合には、
```java
ArrayList<Integer> vector = new ArrayList<Integer>(inputSize);
vector.add(1); vector.add(0); vector.add(1);

neuron.input(vector);
```
というように、input()メソッドを使うことで簡単におこなうことができる。

入力したからには出力を得たい。その場合には、output()メソッドを使う。
```java
int out = neuron.output();
```
0または1のint型の値が返ってくる。

もちろん、inputしていない状態でoutputしても返ってくるのは0の値だけ。
### Network.java
- Neuronインスタンスを配置してNNとして構築

### DeepNeuralNetwork.java
- main クラス
- ここに基本的な例を今のところ3つ載せている。(Aug 19th, 2015)

## Memo
#### Aug 11th
- Neuronオブジェクトのidをint型じゃなくてマトリックスにしないと、ネットワーク構築が面倒！
***

#### Aug 16th
- Aug 11th の問題は解決
- NetworkクラスのinitLayerメソッド難航中
***

#### Aug 19th
- Aug 16th の問題は解決
- Networkクラスの雛形完成
