//处理一句话 : 标签 词vector
#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "N3L.h"

using namespace std;

class Instance {
public:
	Instance() {
	}
	~Instance() {
	}

	int size() const {
		return words.size();
	}

	void clear() {
		labels.clear();
		words.clear();
	}

	void allocate(int length) {
		clear();
		words.resize(length);
	}

	void copyValuesFrom(const Instance& anInstance) {
		allocate(anInstance.size());
		labels = anInstance.labels;
		for (int i = 0; i < anInstance.size(); i++) {
			//labels[i] = anInstance.labels[i];
			words[i] = anInstance.words[i];
		}

	}

	void assignLabel(const string& resulted_labels) {
		//assert(resulted_labels.size() == words.size());
		labels.clear();
		labels = resulted_labels;
		/*for (int idx = 0; idx < resulted_labels.size(); idx++) {
			labels.push_back(resulted_labels[idx]);
		}*/
	}

	void Evaluate(const string resulted_labels, Metric& eval) const {
		//std::cout<<"the raw label: "<<labels<<std::endl;
		if (resulted_labels == labels)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}



public:
	string labels;
	vector<string> words;

};

#endif

