// -------------------------------------------------------------------------------
// Name:        	Neuron.java
// Author:      	Utahka.A
// Created:     	Jul 10th, 2015
// Last Date:   	Sep  7th, 2015
// Note:
// -------------------------------------------------------------------------------
import java.util.ArrayList;
import java.lang.Math;

public class Neuron
{
    /*
        ** Field **
        - id:           各ニューロンのid ---> matrix で表現
        - inputsize:    前の層のニューロンの数 重みつけベクトルや入力ベクトルのサイズ
        - weight:       重みつけベクトル
        - in:           入力ベクトル
        - threshold:    閾値
        - sum:          和
    */
    private ArrayList<Integer> id;
    private int inputSize;
    private ArrayList<Double> weight;
    private ArrayList<Integer> in;
    private double threshold;
    private double sum;

    /*
        ** コンストラクタ for Input Layer **
        - 入力される値は必ず 1 値
        - weight, threshold を指定する Ver.
    */
    public Neuron(ArrayList<Integer> id)
    {
        this.id = id;

        this.weight = new ArrayList<Double>(1);
        this.weight.add(1.0);
        this.threshold = 0.0;

        this.inputSize = this.weight.size();
        this.sum = 0;
    }

    /*
        ** コンストラクタ for Hidden and Output Layer **
        - ランダムで weight と threshold を決定
    */
    public Neuron(ArrayList<Integer> id, int inputSize)
    {
        this.id = id;
        this.weight = new ArrayList<Double>(inputSize);
        this.inputSize = inputSize;

        this.setThreshold();
        this.setWeight();
    }

    public static Neuron mkNeuron(ArrayList<Integer> id, int inputSize)
    {
        return new Neuron(id, inputSize);
    }

    public static Neuron mkNeuronForInputLayer(ArrayList<Integer> id)
    {
        return new Neuron(id);
    }

    @Override
    public String toString()
    {
        return "Neuron(id:" + this.id.toString() + " input size:" + String.valueOf(this.inputSize) + ")";
    }

    // threshold のランダムセット
    private void setThreshold()
    {
        this.threshold = Math.random();
    }

    // weight のランダムセット
    private void setWeight()
    {
        for (int i = 0; i <= this.inputSize; i++)
        {
            this.weight.add(new Double(Math.random()));
        }
    }

    // しきい値を Get
    public double getThreshold()
    {
        return this.threshold;
    }

    /*
        ** 入力用メソッド **
        - 入力ベクトルの要素の重みつき和をとる
    */
    public void input(ArrayList<Integer> inputVector)
    {
        if (inputVector.size() == this.inputSize)
        {
            this.in = inputVector;
            this.summation();
        }
        else
        {
            // エラー処理
        }
    }

    public void input(int inputValue)
    {
        this.sum += inputValue;
    }

    /*
        ** 入力の重みつけ和計算 **
        - inputメソッド中でのみ呼びだし
        - やっていることは入力ベクトルと重みベクトルの内積を取っているだけ
    */
    private void summation()
    {
        for (int i = 0; i < this.inputSize; i++)
        {
            this.sum += this.weight.get(i) * this.in.get(i);
        }
    }

    /*
        ** 出力用メソッド **
        - 重みつき和と閾値を比べて出力を判定
    */
    public int output()
    {
        return  this.sum > this.threshold ? 1 : 0;
    }
}
