/*
 * Labeler.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "MaxEntLabeler.h"
#include "string"
#include "Argument_helper.h"
#include <iomanip>

Labeler::Labeler() {
	// TODO Auto-generated constructor stub
}

Labeler::~Labeler() {
	// TODO Auto-generated destructor stub
	m_classifier.release();
}

//创建训练模型：处理多句话
int Labeler::createAlphabet(const vector<Instance>& vecInsts) {
	cout << "Creating Alphabet..." << endl;

	int numInstance; //句子的数量
	hash_map<string, int> feature_stat; //存放特征及其个数
	m_labelAlphabet.clear();

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->words;
		const string &labels = pInstance->labels;

		vector<string> features;

		m_labelAlphabet.from_string(labels); //存储标签

		extractLinearFeatures(features, pInstance);//抽取pInstance的标签放入features中

		//特征出现的个数
		for (int j = 0; j < features.size(); j++)
			feature_stat[features[j]]++;

		//每100个输出，每4000个换行，句子计数
		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
	cout << "Label num: " << m_labelAlphabet.size() << endl;
	cout << "Total feature num: " << feature_stat.size() << endl;

	m_featAlphabet.clear();

	hash_map<string, int>::iterator feat_iter;
	for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end();
			feat_iter++) {
		if (feat_iter->second > m_options.featCutOff) { //给定一个阈值
			m_featAlphabet.from_string(feat_iter->first); //大于阈值的特征才被记录，存储特征
		}
	}

	cout << "Remain feature num: " << m_featAlphabet.size() << endl;

	m_labelAlphabet.set_fixed_flag(true);
	m_featAlphabet.set_fixed_flag(true);

	return 0;
}

//0 l ou3 d i2 前字 后字 前字 后字 相同个数
void Labeler::extractLinearFeatures(vector<string>& features,
		const Instance* pInstance) {
	features.clear();
	const vector<string> &words = pInstance->words; //句子的各词


	//以声母 韵母分割的情况
	for (int idx = 0; idx < 4; idx++) {
		/*
        Yes_YOUTODO：抽取线性特征： uni-gram bi-gram
        */

		features.push_back("uni_"+words[idx]); //添加一个音素的 uni_zh
	}
	features.push_back("biss_"+words[0]+" "+words[2]); //添加两个因素的 bi_zh_ch
	features.push_back("bisy_"+words[0]+" "+words[3]);
	features.push_back("bisy_"+words[2]+" "+words[1]);
	features.push_back("biyy_"+words[1]+" "+words[3]);


	//添加前后字
	for(int idy=4;idy<9;idy++) //前字+声韵母
	{
		features.push_back("bifs1_"+words[idy]+" "+words[0]);
		features.push_back("bify1_"+words[idy]+" "+words[1]);
	}
	for(int idy=9;idy<14;idy++) //声韵母+后字
	{
		features.push_back("bisb1_"+words[0]+" "+words[idy]);
		features.push_back("biyb1_"+words[1]+" "+words[idy]);
	}
	for(int idy=14;idy<19;idy++) //前字+声韵母
	{
		features.push_back("bifs2_"+words[idy]+" "+words[2]);
		features.push_back("bify2_"+words[idy]+" "+words[3]);
	}
	for(int idy=19;idy<24;idy++) //声韵母+后字
	{
		features.push_back("bisb2_"+words[2]+" "+words[idy]);
		features.push_back("biyb2_"+words[3]+" "+words[idy]);
	}
	features.push_back("F_"+words[24]); //前字相同的个数
	features.push_back("B_"+words[25]); //后字相同的个数

}

void Labeler::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();
	const string &labels = pInstance->labels;
	//int curInstSize = labels.size();

	//将标签放入vector中，有是1，无是0；1 0 / 0 1等（标签类的个数）
	string orcale = labels;
	int numLabels = m_labelAlphabet.size();
	for (int j = 0; j < numLabels; ++j) {
		string str = m_labelAlphabet.from_id(j);
		if (str.compare(orcale) == 0) {
            /*
            Yes_YOUTODO：为 exam.m_labels初始化
            */
			exam.m_labels.push_back(1);
		} else
			exam.m_labels.push_back(0);
	}

	//特征用词袋子表示，有则1，无则0； 1 0 1 0 0 1 1 0（所有特征 维度）
	vector<string> features;
	extractLinearFeatures(features, pInstance);
	for (int j = 0; j < features.size(); j++) {
		int featId = m_featAlphabet.from_string(features[j]);
		if (featId >= 0)
           /*
            Yes_YOUTODO：为 exam.m_features
            */
			exam.m_features.push_back(featId);
	}

}

void Labeler::initialExamples(const vector<Instance>& vecInsts,
		vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam); //将Instance->Example(标签向量和 特征向量)
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
}

void Labeler::train(const string& trainFile, const string& devFile,
		const string& testFile, const string& modelFile,
		const string& optionFile, const string& wordEmb1File,
		const string& wordEmb2File, const string& charEmbFile,
		const string& sentiFile) {

	//将标签初始化 （导入文件：example->option.sparse）
	if (optionFile != "")
		m_options.load(optionFile);
	else{
		std::cout<<"There is no option.sparse file!"<<std::endl;
		exit(0);
	}
	m_options.showOptions();

	//Instance：string vector<string>
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	//将训练语料读入trainInsts中
	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance); //Instance
	//将开发集语料读入devInsts中
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	//将测试语料读入testInsts中
	if (testFile != "")
		m_pipe.readInstances(testFile,testInsts,m_options.maxInstance);

	//创建训练模型
	//得到标签库（所有句子的标签）和特征库（所有特征 【id 特征】）（特征： uni_ bi_）
	createAlphabet(trainInsts);

	m_classifier.init(m_labelAlphabet.size(), m_featAlphabet.size()); //声明数组
	m_classifier.setDropValue(m_options.dropProb); //设置阈值

	//转换成数字向量 labels_vector<int>  features_vector<int>
	vector<Example> trainExamples, devExamples, testExamples;
	initialExamples(trainInsts, trainExamples); //将Instance -> Example
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	double bestDIS = 0; //迭代maxIter次之后，开发集的准确率最大值
	double bestTEST=0; //迭代maxIter次之后，测试集的准确率最大值

	int inputSize = trainExamples.size();//训练句子的个数

	int batchBlock = inputSize / m_options.batchSize; //(初始时 batchSize=1)
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) { //迭代次数
		std::cout << "##### Iteration " << iter << std::endl;

		//The num of batchSize samples of random selecion
		random_shuffle(indexes.begin(), indexes.end()); //将indexes打乱顺序
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			//将训练句子打乱顺序放入subExamples中
			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			//the current times of runnig "process" , the current times of processing batchBlock
			int curUpdateIter = iter * batchBlock + updateIter;
			double cost = m_classifier.process(subExamples, curUpdateIter);//处理训练语料

			eval.overall_label_count += m_classifier._eval.overall_label_count;
			eval.correct_label_count += m_classifier._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_classifier.checkgrads(subExamples, curUpdateIter+1);
				std::cout << "current: " << updateIter + 1 << ", total block: "
						<< batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = "
						<< eval.getAccuracy() << std::endl;
			}
			m_classifier.updateParams(m_options.regParameter,
					m_options.adaAlpha, m_options.adaEps);//更新w

		}

		//batchBlock=1时，更新一次w，进行一次开发集和测试集，输出开发及测试结果；
		//当迭代完成时，输出开发集最好的情况下的开发结果和测试结果
		if (devNum > 0) { //迭代多次，选取最好的一次迭代结果

			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				string result_labels;
				predict(devExamples[idx].m_features, result_labels);

				/*if (m_options.seg)
				 devInsts[idx].SegEvaluate(result_labels, metric_dev);
				 else
				 devInsts[idx].Evaluate(result_labels, metric_dev);*/
				devInsts[idx].Evaluate(result_labels, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_labels);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			std::cout << "dev:" << std::endl;
			//std::cout<<"compare accuracy: "<<metric_dev.getAccuracy()<<std::endl;
			metric_dev.print();


			if (!m_options.outBest.empty()
					&& metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest,
						decodeInstResults); //输出的开发集最好情况下的 开发 结果标签
				bCurIterBetter = true;
			}

			//开发集进行一次，测试就进行一次
			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					string result_labels;
                    /*
                    Yes_YOUTODO：在测试集上进行测试
                    */
					predict(testExamples[idx].m_features,result_labels);

					//当前得到的标签是result_labels！！！
					//std::cout <<idx+1<<": the raw label: "<<testInsts[idx].labels<<"---";
					//std::cout <<"now the label: "<<result_labels <<std::endl;
					testInsts[idx].Evaluate(result_labels, metric_test);


					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_labels);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest,
							decodeInstResults); //输出的开发集最好情况下的 测试 结果标签
				}
			}

			if (m_options.saveIntermediate
					&& metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS
						<< ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				bestTEST=metric_test.getAccuracy();

			}

		}
		// Clear gradients
	}
	std::cout << "Now exceeds best Dev performance of " << bestDIS
											<< ". Saving model file.." << std::endl;
	std::cout << "Now exceeds best Test performance of " << bestTEST
												<< ". Saving model file.." << std::endl;
}

int Labeler::predict(const vector<int>& features, string& output) {

	vector<dtype> labelprobs;
	int labelId = m_classifier.predict(features, labelprobs);

	output = m_labelAlphabet.from_id(labelId);

	return 0;
}

int main(int argc, char* argv[]) {
#if USE_CUDA==1
	InitTensorEngine();
#else
	InitTensorEngine<cpu>();
#endif

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
	std::string wordEmb1File = "", wordEmb2File = "", charEmbFile = "",
			optionFile = "";
	std::string outputFile = "";
	std::string sentiFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string",
			"training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string",
			"development corpus to train a model, optional when training",
			devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
			"testing corpus to train a model or input file to test a model, optional when training and must when testing",
			testFile);
	ah.new_named_string("model", "modelFile", "named_string",
			"model file, must when training and testing", modelFile);
	ah.new_named_string("word1", "wordEmb1File", "named_string",
			"pretrained word embedding file 1 to train a model, optional when training",
			wordEmb1File);
	ah.new_named_string("word2", "wordEmb2File", "named_string",
			"pretrained word embedding file 2 to train a model, optional when training",
			wordEmb2File);
	ah.new_named_string("char", "charEmbFile", "named_string",
			"pretrained char embedding file to train a model, optional when training",
			charEmbFile);
	ah.new_named_string("sen", "sentiFile", "named_string",
			"sentiment word file to train a model, optional when training",
			sentiFile);
	ah.new_named_string("option", "optionFile", "named_string",
			"option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string",
			"output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	Labeler tagger;
	tagger.train(trainFile, devFile, testFile, modelFile, optionFile,
			wordEmb1File, wordEmb2File, charEmbFile, sentiFile);

	//训练集目的：更新w
	//开发集目的：寻找开发集结果最好的情况下，对应的测试集的结果
	//测试集：输出结果

	//ah.write_values(std::cout);
#if USE_CUDA==1
	ShutdownTensorEngine();
#else
	ShutdownTensorEngine<cpu>();
#endif
}
