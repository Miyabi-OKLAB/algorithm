// -------------------------------------------------------------------------------
// Name:        	DeepNeuralNetwork.java
// Author:      	Utahka.A
// Created:     	Jul 10th, 2015
// Last Date:   	Jul 11th, 2015
// Note:
// -------------------------------------------------------------------------------
import java.util.ArrayList;

class DeepNeuralNetwork
{
    public static void main(String[] args)
    {
        ArrayList<Float> weight = new ArrayList<Float>(3);
        weight.add(new Float(0.5)); weight.add(new Float(0.5)); weight.add(new Float(0.5));

        ArrayList<Integer> in = new ArrayList<Integer>(3);
        in.add(1); in.add(0); in.add(1);

        double threshold = 0.5;  // てきとー
        int out = 0;

        Neuron neuron = new Neuron(1, weight, threshold);
        neuron.input(in);
        out = neuron.output();

        System.out.println(out);
    }
}
