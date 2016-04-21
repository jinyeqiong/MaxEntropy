/*
 * SparseClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier_H_
#define SRC_SparseClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseClassifier {
public:
	SparseClassifier() {
		_dropOut = 0.5;
		hiddenCol=100; //第二层100列
	}
	~SparseClassifier() {

	}

public:
	int _labelSize;
	int _linearfeatSize;

	dtype _dropOut;
	Metric _eval;
	int hiddenCol;

	SparseUniLayer<xpu> _layer_linear; //稀疏层
	UniLayer<xpu> _layer_hidden;  //稠密层

public:

	inline void init(int labelSize, int linearfeatSize) {
		_labelSize = labelSize;
		_linearfeatSize = linearfeatSize;

		_layer_linear.initial(hiddenCol, _linearfeatSize, false, 40, 2); //n*m
		_layer_hidden.initial(_labelSize,hiddenCol, false, 40, 2); //m*2

		_eval.reset();

	}

	inline void release() {
		_layer_linear.release();
		_layer_hidden.release();
	}

	inline dtype process(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;
		int offset = 0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			Tensor<xpu, 2, dtype> output1, outputLoss1;
			Tensor<xpu, 2, dtype> output2, outputLoss2;

			//initialize

			output1 = NewTensor<xpu>(Shape2(1, hiddenCol), d_zero);
			outputLoss1 = NewTensor<xpu>(Shape2(1, hiddenCol), d_zero);

			output2 = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
			outputLoss2 = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

			//forward propagation
			vector<int> linear_features;

				//random to dropout some feature
			const vector<int>& feature = example.m_features;
			srand(iter * example_num); //只是设置随机数种子
			linear_features.clear();
			for (int idy = 0; idy < feature.size(); idy++) {
				if (1.0 * rand() / RAND_MAX >= _dropOut) {
					linear_features.push_back(feature[idy]);
				}
			}

			/*
            yes_YOUTODO：调用ComputeForwardScore
            */
			_layer_linear.ComputeForwardScore(linear_features, output1);
			_layer_hidden.ComputeForwardScore(output1, output2);

			// get delta for each output
			cost += softmax_loss(output2, example.m_labels, outputLoss2, _eval,
					example_num);

			// loss backward propagation
            /*
                yes_YOUTODO：调用ComputeBackwardLoss
            */
			_layer_hidden.ComputeBackwardLoss(output1, output2, outputLoss2,outputLoss1);
			_layer_linear.ComputeBackwardLoss(linear_features, output1, outputLoss1);

			//release

			FreeSpace(&output1);
			FreeSpace(&outputLoss1);
			FreeSpace(&output2);
			FreeSpace(&outputLoss2);

		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	int predict(const vector<int>& features, vector<dtype>& results) {

		Tensor<xpu, 2, dtype> output1;
		Tensor<xpu, 2, dtype> output2;

		//initialize

		output1 = NewTensor<xpu>(Shape2(1, hiddenCol), d_zero);
		output2 = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation

		/*
         yes_YOUTODO：调用ComputeForwardScore
        */
		_layer_linear.ComputeForwardScore(features, output1);
		_layer_hidden.ComputeForwardScore(output1, output2);

		// decode algorithm
		int result = softmax_predict(output2, results);

		//release
		FreeSpace(&output1);
		FreeSpace(&output2);

		return result;

	}

	dtype computeScore(const Example& example) {

		Tensor<xpu, 2, dtype> output1;
		Tensor<xpu, 2, dtype> output2;

		//initialize

		output1 = NewTensor<xpu>(Shape2(1, hiddenCol), d_zero);
		output2 = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation
		/*
         yes_YOUTODO：调用ComputeForwardScore
        */
		_layer_linear.ComputeForwardScore(example.m_features, output1);
		_layer_hidden.ComputeForwardScore(output1, output2);

		// get delta for each output
		dtype cost = softmax_cost(output2, example.m_labels);

		//release
		FreeSpace(&output1);
		FreeSpace(&output2);

		return cost;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_layer_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	void writeModel();

	void loadModel();

	void checkgrads(const vector<Example>& examples, int iter) {
		checkgrad(this, examples, _layer_linear._W, _layer_linear._gradW,
				"_layer_linear._W", iter, _layer_linear._indexers, false);
		checkgrad(this, examples, _layer_linear._b, _layer_linear._gradb,
				"_layer_linear._b", iter);
		checkgrad(this, examples, _layer_hidden._W, _layer_hidden._gradW,
				"_layer_hidden._W", iter);
		checkgrad(this, examples, _layer_hidden._b, _layer_hidden._gradb,
				"_layer_hidden._b", iter);
	}

public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

};

#endif /* SRC_SparseClassifier_H_ */
