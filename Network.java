// -------------------------------------------------------------------------------
// Name:        	Network.java
// Author:      	Utahka.A
// Created:     	Jul 10th, 2015
// Last Date:   	Sep  7th, 2015
// Note:            - コメントなど、よくわからない場合には連絡ください (Aug 05th, 2015)
// -------------------------------------------------------------------------------
import java.util.ArrayList;

public class Network
{
    /*
        ** field **
        - networkSizeList:  NNのニューロン構造を示す配列
                            [EX.] {3, 2, 2, 3} なら
                            要素は、入力層、隠れ層、出力層のニューロンの数を表している

        - networklayer:     Neuron を層的に格納する２次元配列
                               {3, 2, 2, 3}
                                。 。 。 。
                                。 。 。 。
                                。 　 　 。  ...みたいになります
                            - 入力層: networklayer の 0番目の要素
                            - 出力層: networklayer の最後の要素
                            - 隠れ層: networklayer のそれ以外の要素
    */
    private ArrayList<Integer> networkSizeList;
    private ArrayList<ArrayList<Neuron>> networklayer;
    private int howManyLayer;
    private ArrayList<Integer> outputVector;

    public Network(ArrayList<Integer> networkSizeList)
    {
        this.networkSizeList = networkSizeList;
        this.howManyLayer = this.networkSizeList.size();
        this.outputVector = new ArrayList<Integer>();
    }

    public static Network make(ArrayList<Integer> networkSizeList)
    {
        Network network = new Network(networkSizeList);

        ArrayList<Integer> id = new ArrayList<Integer>(2);
        network.networklayer = new ArrayList<ArrayList<Neuron>>(network.howManyLayer);
        for (int i = 0; i < network.howManyLayer; i++)
        {
            int size = network.networkSizeList.get(i);
            ArrayList<Neuron> sub = new ArrayList<Neuron>(size);
            id.add(i);
            for (int j = 0; j < size; j++)
            {
                id.add(j);
                if (i == 0)
                {
                    sub.add(new Neuron(new ArrayList<Integer>(id)));
                }
                else
                {
                    sub.add(new Neuron(new ArrayList<Integer>(id), network.networkSizeList.get(i-1)));
                }
                id.remove(1);
            }
            network.networklayer.add(new ArrayList<Neuron>(sub));
            sub.clear();
            id.clear();
        }
        return network;
    }

    public static ArrayList<Integer> mkVector(int[] arr)
    {
        ArrayList<Integer> arrList = new ArrayList<Integer>(arr.length);
        for (int element: arr)
        {
            arrList.add(element);
        }
        return new ArrayList<Integer>(arrList);
    }

    public void printNetwork()
    {
        for (ArrayList<Neuron> layer: this.networklayer)
        {
            for (Neuron neuron: layer)
            {
                System.out.println(neuron);
            }
        }
    }

    public void input(ArrayList<Integer> inputVector)
    {
        ArrayList<Integer> buffer = new ArrayList<Integer>();
        int layerNumber = 0;
        for (int size: this.networkSizeList)
        {
            for (int i = 0; i < size; i++)
            {
                ArrayList<Integer> id = this.mkID(layerNumber, i);
                Neuron targetNeuron = this.catchNeuron(id);
                if (layerNumber == 0)
                {
                    targetNeuron.input(inputVector.get(i));
                }
                else
                {
                    targetNeuron.input(this.outputVector);
                }
                buffer.add(targetNeuron.output());
            }
            this.outputVector = new ArrayList<Integer>(buffer);
            buffer.clear();
            layerNumber++;
        }
    }

    public ArrayList<Integer> output()
    {
        return this.outputVector;
    }

    /*
        Neuron をネットワークから参照取り出しする。
    */
    public Neuron catchNeuron(ArrayList<Integer> id)
    {
        ArrayList<Neuron> layer = this.networklayer.get(id.get(0));
        return layer.get(id.get(1));
    }

    /*
        Neuronオブジェクトで利用可能な id の作成
        静的なので、インスタンス化なしで利用可能
    */
    public static ArrayList<Integer> mkID(int x, int y)
    {
        ArrayList<Integer> id = new ArrayList<Integer>(2);
        id.add(x); id.add(y);
        return id;
    }
}
