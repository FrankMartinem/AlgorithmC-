#include "pch.h"
#include "svm.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

const int feature_size = 40;
const int train_size = 600;
svm_problem prob;

void init_svm_problem()
{
	fstream f_train;
	f_train.open("train.txt",ios::in);
	prob.l = train_size;
	prob.y = new double[train_size];
	prob.x = new svm_node*[train_size];
	svm_node *x_space = new svm_node[train_size*(1 + feature_size)];
	double value, lb;
	int idx;
	for (int i = 0; i < train_size; i++) {
		f_train >> lb;
		prob.y[i] = lb ;
		//if (i < train_size / 2) prob.y[i] = 1;
		//else       prob.y[i] = -1;
		idx = 0;
		for (int j = 0; j < feature_size; j++) {
			f_train >> value;
			
			if (value != 0.0) {
				x_space[i*(feature_size + 1) + j].index = j + 1;
				x_space[i*(feature_size + 1) + j].value = value;
			}
		}
		//½áÊø±ê¼Ç-1
		x_space[i*(feature_size + 1) + feature_size].index = -1;
		prob.x[i] = &x_space[i*(feature_size + 1)];
	}
	f_train.close();

}

svm_parameter param;
void  init_svm_parameter() {
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.001;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 11;
	param.eps = 1e-4;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}

const int test_size = 300;
double predict_lable[test_size];
double test_lable[test_size];

int  main() {
	init_svm_problem();
	init_svm_parameter();
	if (param.gamma == 0) param.gamma = 0.5;
	svm_model* model = svm_train(&prob, &param);
	ifstream in;
	in.open("test.txt");
	svm_node *test = new svm_node[feature_size];
	for (int i = 0; i < test_size; i++) {
		double value;
		int idx = 0;
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
	printf("%.2lf%%\n", (0.0 + yes) / test_size * 100);
	in.close();
	return 0;
}