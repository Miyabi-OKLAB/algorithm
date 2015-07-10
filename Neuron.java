import java.util.ArrayList;
import java.util.Random;

class Neuron
{
    // member variable
    private Random rand;
    private int id, inputSize;              // ID
    private ArrayList<Float> weight;        // 重みつけ
    private double threshold;               // しきい値
    private double sum;                     // 和
    private ArrayList<Integer> in;          // input
    private int out;                        // output

    // コンストラクタ1
    Neuron(int id, ArrayList<Float> weight, double threshold)
    {
        this.id = id;
        this.weight = weight;
        this.threshold = threshold;
        this.inputSize = this.weight.size();
        this.sum = 0;
    }

    // コンストラクタ2
    // 未完成
    Neuron(int id)
    {
        this.rand = new Random();
        this.threshold = 0;
        this.id = id;
    }

    // ランダムセット
    public void setThreshold()
    {
        this.threshold = rand.nextInt(10) + 1;
    }

    // しきい値を Get
    public double getThreshold()
    {
        return this.threshold;
    }

    public void input(ArrayList<Integer> data)
    {
        if (data.size() == this.inputSize)
        {
            this.in = data;
            this.summation();
        }
        else
        {
            // エラー処理
        }
    }

    /*
        ** 入力の重みつけ和計算 **
        inputメソッド中でのみ呼びだし
    */
    private void summation()
    {
        for (int i = 0; i < this.inputSize; i++)
        {
            this.sum += this.weight.get(i) * this.in.get(i);
        }
    }

    public int output()
    {
        if (this.sum > this.threshold)
        {
            this.out = 1;
            return 1;
        }
        else
        {
            this.out = 0;
            return 0;
        }
    }
}
