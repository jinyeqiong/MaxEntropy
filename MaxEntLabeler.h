/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_


#include "N3L.h"

//#include "basic/SparseClassifier.h"
#include "basic/SparseClassifier2.h"
//#include "basic/SparseClassifier3.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  Alphabet m_featAlphabet;
  Alphabet m_labelAlphabet;

public:
  Options m_options;
  Pipe m_pipe;

#if USE_CUDA==1
  SparseClassifier<gpu> m_classifier;
#else
  SparseClassifier<cpu> m_classifier;
#endif

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:

  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmb1File, const string& wordEmb2File, const string& charEmbFile, const string& sentiFile);
  int predict(const vector<int>& features, string& outputs);

};

#endif /* SRC_NNCRF_H_ */
