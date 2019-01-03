#ifndef MLP_H
#define MLP_H
#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<math.h>
#include<openblas/cblas.h>		//openblas는 선택입니다. cblas_sgemm을 사용할수 없다면 아래의 my_sgemm을 사용할 수 있다.
#ifndef CBLAS_H
#define CblasRowMajor 0
#define CblasNoTrans 111
#define CblasTrans 112	//112는 openblas의 CblasTrans의 상수
#endif
#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif
#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif
#ifndef __cplusplus
typedef struct FCLayer FCLayer;
typedef struct Network Network;
#endif
struct FCLayer {		// Fully connected layer
	int n;				// 뉴런의 개수
	float* w;			// 가중치 , [이전레이어의 크기] x [현재레이어의 크기] 의 2차원 행렬
	float* neuron;	    // 뉴런
	float* error;		// 오차
};
struct Network {
	FCLayer* layers;	//레이어들의 배열
	int depth;			//레이어의 개수
};
inline Network CreateNetwork(int* size_of_layers, int num_of_layers) {
	Network network;
	network.layers = (FCLayer*)calloc(num_of_layers, sizeof(FCLayer));
	network.depth = num_of_layers;
	for (int i = 0; i < num_of_layers; i++) {
		network.layers[i].n = size_of_layers[i];
		network.layers[i].error = (float*)calloc(size_of_layers[i], sizeof(float));
		if (i != 0) {	//첫번째 레이어는 가중치와 뉴런값이 없음.
			network.layers[i].w = (float*)calloc(size_of_layers[i - 1] * size_of_layers[i], sizeof(float));
			network.layers[i].neuron = (float*)calloc(size_of_layers[i], sizeof(float));
			for (int j = 0; j < size_of_layers[i - 1] * size_of_layers[i]; j++) {
				network.layers[i].w[j] = (float)rand() / RAND_MAX * 2 - 1.0F;	//[-1,1] 로 초기화
			}
		}
	}
	return network;
}
inline void my_sgemm(int major, int transA, int transB, int M, int N, int K, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc) {
	//aAB+bC=C
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			float c = C[m*ldc + n];
			C[m*ldc + n] = 0.0F;
			for (int k = 0; k < K; k++) {
				float a, b;
				a = transA == CblasTrans ? A[k*lda + m] : A[m*lda + k];
				b = transB == CblasTrans ? B[n*ldb + k] : B[k*ldb + n];
				C[m*ldc + n] += a*b;
			}
			C[m*ldc + n] = alpha*C[m*ldc + n] + beta*c;
		}
	}
}
 
//딥러닝에서는 ReLU를 많이 사용하나, 
//간단한 실습이니 Sigmoid를 사용
inline float Sigmoid(float x) {
	return 1.0F / (1.0F + expf(-x));
}
inline float Sigmoid_Derivative(float x) {
	return x*(1.0F - x);
}
inline int Forward(Network* network, float* input) {
	network->layers[0].neuron = input;
	for (int i = 1; i < network->depth; i++) {
		//현재 레이어의 뉴런의 값은 이전 레이어의 뉴런과 가중치의 계산으로 나온다. aAB+bC
		my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans
					, network->layers[i].n      // M
					, 1                         // N
					, network->layers[i - 1].n  // K
					, 1.0F    // alpha
					, network->layers[i].w, network->layers[i - 1].n    // A, lda
					, network->layers[i - 1].neuron, 1      // B, ldb
					, 0.0F    // beta
					, network->layers[i].neuron, 1);        // C, ldc
		for (int j = 0; j < network->layers[i].n; j++) {
			network->layers[i].neuron[j] = Sigmoid(network->layers[i].neuron[j]);
		}
	}
	int a = 0;
	float max_value = network->layers [ network->depth - 1 ].neuron [ 0 ];
	for (int i = 0; i < network->layers[network->depth - 1].n; i++) {
		if (max_value < network->layers[network->depth - 1].neuron[i]) {
			max_value = network->layers[network->depth - 1].neuron[i];
			a = i;
		}
	}
	return a;
}
inline void Backward(Network* network, int label, float learning_rate) {
	//Calculate last layer's error
	for ( int i = 0; i < network->layers [ network->depth - 1 ].n; i++ ) {
		network->layers [ network->depth - 1 ].error [ i ] = 0.0F - network->layers [ network->depth - 1 ].neuron [ i ];
	}
	network->layers [ network->depth - 1 ].error [ label ] = 1.0F - network->layers [ network->depth - 1 ].neuron [ label ];
	//Calculate other layer's error
	for ( int i = network->depth - 1; i > 0; i-- ) {
		my_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans  //ORDER
					, network->layers [ i - 1 ].n, 1, network->layers [ i ].n // M N K
					, 1.0F // alpha
					, network->layers [ i ].w, network->layers [ i - 1 ].n // A, lda
					, network->layers [ i ].error, 1                    // B, ldb
					, 0.0F											    // beta
					, network->layers [ i - 1 ].error, 1);				// C, ldc
	}
	//Update weights
	for ( int i = network->depth - 1; i >= 1; i-- ) {
		float* Gradient = ( float* ) calloc(network->layers [ i ].n, sizeof(float));
		for ( int j = 0; j < network->layers [ i ].n; j++ ) {
			Gradient [ j ] = network->layers [ i ].error [ j ] * Sigmoid_Derivative(network->layers [ i ].neuron [ j ]);
		}
		my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans
					, network->layers [ i ].n, network->layers [ i - 1 ].n, 1
					, learning_rate
					, Gradient, 1          //A
					, network->layers [ i - 1 ].neuron, network->layers [ i - 1 ].n  //B
					, 1.0F
					, network->layers [ i ].w, network->layers [ i - 1 ].n);        //C
		free(Gradient);
	}
}

#endif