// -------------------------------------------------------------------------------
// Name:        	DeepNeuralNetwork.java
// Author:      	Utahka.A
// Created:     	Jul 10th, 2015
// Last Date:   	Sep  7th, 2015
// Note:
// -------------------------------------------------------------------------------
import java.util.ArrayList;

class DeepNeuralNetwork
{
    /* ** Example(1) **
    public static void main(String[] args)
    {
        // 重み係数の設定
        ArrayList<Float> weight = new ArrayList<Float>(3);
        weight.add(new Float(0.5)); weight.add(new Float(0.5)); weight.add(new Float(0.5));

        // 入力ベクトルの設定
        ArrayList<Integer> in = new ArrayList<Integer>(3);
        in.add(1); in.add(0); in.add(1);

        // 閾値の設定
        double threshold = 0.5;  // てきとー
        int out = 0;

        Neuron neuron = new Neuron(1, weight, threshold);
        neuron.input(in);
        out = neuron.output();

        System.out.println(out);
    }
    */

    /* ** Example(2) **
    public static void main(String[] args)
    {
        ArrayList<Integer> in = new ArrayList<Integer>(3);
        in.add(1); in.add(0); in.add(1);

        ArrayList<Integer> id = Network.mkID(0, 0);

        Neuron neuron = new Neuron(id, 3);
        neuron.input(in);
        System.out.println(neuron.output());
    }
    */

    /* ** Example(3) **
    public static void main(String[] args)
    {
        ArrayList<Integer> networkDesign = new ArrayList<Integer>(3);
        networkDesign.add(3); networkDesign.add(10); networkDesign.add(2);

        ArrayList<Integer> in = new ArrayList<Integer>(3);
        in.add(1); in.add(0); in.add(1);

        Network network = new Network(networkDesign);
        network.input(in);
        System.out.println(network.output());
    }
    */

    // ** Example(4) **
    public static void main(String[] args)
    {
        ArrayList<Integer> networkDesign = Network.mkVector(new int[]{3, 10, 2});
        ArrayList<Integer> in = Network.mkVector(new int[]{1, 0, 1});

        Network network = Network.make(networkDesign);
        network.input(in);
        System.out.println(network.output());
    }
    //
}
