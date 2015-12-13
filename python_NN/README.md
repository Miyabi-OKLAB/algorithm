Neural Network 
===
- python/chainerを用いたNeural Network Model
- このreadmeはアキバくんが書いたreadmeがかっこよかったので書きたかっただけ（ゆるして）

## Documentation
### python_NN
一度コード内でも説明しているが書きたいのでここでも説明する  

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
