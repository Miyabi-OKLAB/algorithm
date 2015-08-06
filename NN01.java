import java.util.Random;

public class NN01{
	public Random rand;
	public int input_size;
	public int hidden_layer_size;
	public int output_size;
	public int[] n_ins;
	public int[] n_hiddens;
	public int[] n_outs;
	public double[][] W_ih;
	public double[][] W_ho;
	public double[] T_h;
	public double[] T_o;


	public NN01(int input_size, int hidden_layer_size, int output_size, Random rand){

		this.rand = rand;
		this.input_size = input_size;
		this.hidden_layer_size = hidden_layer_size;
		this.output_size = output_size;

		this.n_ins = new int[input_size];
		this.n_hiddens = new int[hidden_layer_size];
		this.n_outs = new int[output_size];

		// weight
		this.W_ih = new double[input_size][hidden_layer_size];
		this.W_ho = new double[hidden_layer_size][output_size];

		// theta
		this.T_h = new double[hidden_layer_size];
		this.T_o = new double[output_size];

		// random
		if(rand == null)	this.rand = new Random(123);
		else				this.rand = rand;

		//********************************************		
		// Weight random
		// in - hid
		for(int i = 0; i < input_size; i++){
			for(int j = 0; j < hidden_layer_size; j++){
				W_ih[i][j] = rand.nextDouble();
			}
		}
		// hid - out
		for(int i = 0; i < hidden_layer_size; i++){
			for(int j = 0; j < output_size; j++){
				W_ho[i][j] = rand.nextDouble();
			}
		}

		// Theta random
		// hid
		for(int i = 0; i < hidden_layer_size; i++){
			T_h[i] = rand.nextDouble();
		}
		// out		
		for(int i = 0; i < output_size; i++){
			T_o[i] = rand.nextDouble();
		}
		//********************************************
	}

	public void finetune(int[][] Train_X, int[][] Train_Y, int n_ins, int train_N){
		
		// イテレーション
		for(int N = 0; N < train_N; N++){
			// データ数
			for(int data_num = 0; data_num < n_ins; data_num++){
	
				for(int i = 0; i < this.input_size; i++){
					this.n_ins[i] = Train_X[data_num][i];
				}

			}
		}
	}

	// Sigmoid function
	public static double sigmoid(double x){
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}

	public static void test_nn(){
		Random rand = new Random(123);

		// モデルサイズ
		int input_size = 10;
		int hidden_layer_size = 10;
		int output_size = 2;

		// イテレーション
		int data_num = 10;
		int train_N = 10;
		
		// モデル
		NN01 nn01 = new NN01(input_size, hidden_layer_size, output_size, rand);

		// 学習データ
		int[][] Train_X = {
			{1,1,1,1,1,0,0,0,0,0},
			{1,1,0,1,1,0,0,0,0,0},
			{1,1,1,1,1,0,0,0,0,0},
			{1,0,1,1,1,0,0,0,0,0},
			{1,1,1,1,1,0,0,0,0,0},
			{0,0,0,0,0,1,1,1,1,1},
			{0,0,0,0,0,1,0,1,1,1},
			{0,0,0,0,0,1,1,1,1,1},
			{0,0,0,0,0,1,1,1,0,1},
			{0,0,0,0,0,1,1,1,1,1}
		}

		// 教師データ
		int[][] Train_Y = {
			{1,0},
			{1,0},
			{1,0},
			{1,0},
			{1,0},
			{0,1},			
			{0,1},			
			{0,1},			
			{0,1},			
			{0,1}
		}

		// 学習
		nn01.finetune(Train_X, Train_Y, data_num, train_N);


	}

	public static void main(String[] args){
		test_nn();

	}

}