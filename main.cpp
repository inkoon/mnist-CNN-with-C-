/*
*	Convolution Neural Network example
*/
#include"cnn.h"
#include<mnist.h>

int num_of_epoch = 50;

int main() {
	srand(1);		//������� ���� ������ ������ ���.
	int num_of_valid;
	MNIST* valid = ReadMNIST("./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte", &num_of_valid, 0);
	int num_of_train;
	MNIST* train = ReadMNIST("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte", &num_of_train, 0);

	num_of_train /= 10; num_of_valid /= 10;

	//INPUT = 28x28x1
	//5x5 kernel�� ���.
	//kernel_depth�� ũ�⸸ŭ depth�� ������.
	//C1�� depth�� 6, C2�� depth�� 12

	int kernel_depths[] = { 6, 72 };
	CNN cnn = CreateCNN(28, 1, 2, kernel_depths, 5);

	//ReadParameters(&cnn,"parameter.txt");
	float learning_rate = 0.04F;
	//tip : train, valid�� 96%�� �Ѿ����� learing_rate�� ����� ����� ����ȭ ����.
	for (int e = 1; e <= num_of_epoch; e++) {
		int answer = 0;
		for (int j = 0; j < num_of_train; j++) {
			int a = ForwardCNN(&cnn, train[j].image);
			if (a == train[j].label) {
				answer++;
			}
			BackwardCNN(&cnn, train[j].label, learning_rate);
		}
		printf("epoch = %d\ttrain : %f\t", e, (float)answer / num_of_train);
		answer = 0;
		for (int j = 0; j < num_of_valid; j++) {
			int a = ForwardCNN(&cnn, valid[j].image);
			if (a == valid[j].label) {
				answer++;
			}
		}
		printf("validation : %f", (float)answer / num_of_valid);
		puts("");
		//WriteParameters(&cnn,"parameter.txt");
	}
	return 0;
}