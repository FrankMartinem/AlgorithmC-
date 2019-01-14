#include "pch.h"
#include "svm.h"
#include <iostream>
#include <fstream>
using namespace std;

const int feature_size = 2;
const int train_size = 200;
svm_problem prob;

void init_svm_problem() {
	prob.l = train_size;
	prob.y = new double[train_size];
	prob.x = new svm_node*[train_size];
	svm_node *x_space = new svm_node[train_size*(1 + feature_size)];
	ifstream  in;
	in.open("train_data.txt");
	double value, lb;
	for (int i = 0; i < train_size; i++) {
		in >> lb;
		//prob.y[i] = lb ;
		if (i < train_size / 2) prob.y[i] = 1;
		else       prob.y[i] = -1;
		for (int j = 0; j < feature_size; j++) {
			in >> value;
			if (value != 0.0) {
				x_space[i*(feature_size + 1) + j].index = j + 1;
				x_space[i*(feature_size + 1) + j].value = value;
			}
		}
		//结束标记-1
		x_space[i*(feature_size + 1) + feature_size].index = -1;
		prob.x[i] = &x_space[i*(feature_size + 1)];
	}
	in.close();
}

svm_parameter param;
void  init_svm_parameter() {
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.0001;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 11;
	param.eps = 1e-5;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}

const int test_size = 100;
double predict_lable[test_size];
double test_lable[test_size];

int  main() {
	init_svm_problem();
	init_svm_parameter();
	if (param.gamma == 0) param.gamma = 0.5;
	svm_model* model = svm_train(&prob, &param);
	ifstream in;
	in.open("test_data.txt");
	svm_node *test = new svm_node[3];
	for (int i = 0; i < test_size; i++) {
		double value;
		in >> test_lable[i];
		for (int j = 0; j < feature_size; j++) {
			in >> value;
			if (value != 0.0) {
				test[j].index = j + 1;
				test[j].value = value;
			}
		}
		test[feature_size].index = -1;
		predict_lable[i] = svm_predict(model, test);
	}

	int yes = 0;
	for (int i = 0; i < test_size; i++) {
		// cout<<test_lable[i] <<" , "<<predict_lable[i]<<endl ;
		if (test_lable[i] == predict_lable[i])  yes++;
	}
	cout << yes << endl;
	printf("%.2lf%%\n", (0.0 + yes) / test_size);
	in.close();
	return 0;
}
