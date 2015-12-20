Neural Network 
===
- python/chainerを用いたNeural Network Model
- このreadmeはアキバくんが書いたreadmeがかっこよかったので書きたかっただけ（ゆるして）

## Documentation
### makeModel.py
- 識別学習のためのモデル構築を行うコード
- 学習後にモデルデータを吐き出す
- 使用する際には主に以下のパラメタを変更する
```python
batchsize   = num
n_epoch     = num
input_size  = num
hidden_size = num
output_size = num
```
### batchisize
バッチサイズは学習データを小分けにして学習する際のサイズのこと  
- epochは小分けにしたデータを学習すること  
- itelationは全体の学習データを学習すること  
つまり，epoch * batch が itelation になる  

### classification.py
- 識別を行うコード
- input.csvを読み込みoutput.txtを吐き出す
- パラメタはmakeModel.pyに合わせてあげて

### p4j.java
- javaからclassification.pyを実行させるコード
- そのままp4jを呼び出してあげればpythonが動く
- 秋葉くんありがとう:wq
