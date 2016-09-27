#include "feature.hpp"
#include "mcdetect.hpp"
#include "f_Haralick.hpp"
#include "f_Geometric.hpp"
#include "lib.hpp"
#include <fstream>

cLib L;

#define READ_DB 1
#define READ_PCA 0
#define READ_SVM 0
#define READ_LDA 0

cFeature::cFeature(void)
{
	db_status.i_NORMALIZED = db_status.c_NORMALIZED; db_status.SELECTED;
	qda_data.lda_trained=false;

	// Next: set SVM structure. Use param to simply this
	svm_data.i_svm = ml::SVM::create();
	svm_data.i_svm->setType(ml::SVM::C_SVC);	svm_data.i_svm->setKernel(ml::SVM::POLY);
	svm_data.i_svm->setGamma(2.7803);	svm_data.i_svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	svm_data.i_svm->setC(1000); svm_data.i_svm->setDegree(3);

	svm_data.c_svm = ml::SVM::create();
	svm_data.c_svm->setType(ml::SVM::C_SVC);	svm_data.c_svm->setKernel(ml::SVM::POLY);
	svm_data.c_svm->setGamma(2.7803);	svm_data.c_svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	svm_data.c_svm->setC(1000); svm_data.c_svm->setDegree(3);

	this->getSelData();
	//cout<<this->i_select<<"\n-------\n"<<this->p_select<<"\n-------\n";
}
void cFeature::exec()
{
	generateDB();
}
void cFeature::addFV(size_t fileidx, string impath, float label)
{
	// Add individual+cluster features for image 'fileidx' to main DB. Selection is done implicitly
	// raw_feats is the original individual+cluster (dim = 129+144) vector
	cLib L; cMcDetect M; cGeometric G; Mat raw_mat, GI, BI;
	if(L.readImage(impath,GI)) // Get grey level image
		return; // invalid image path

	BI = M.exec(GI);// Get binary image
	//namedWindow("Display Image", WINDOW_NORMAL); imshow("Display Image", GI); waitKey(0);
	vector<vector<vector<double>>> raw_feats = G.exec(BI,GI);
	if (raw_feats.size() && raw_feats[0].size() && raw_feats[0][0].size()) {
		int i_feats = raw_feats[0].size();
		for (size_t i = 0; i < raw_feats[0].size(); i++) { // Insert the file index for each individual object feature
			db_data.i_file_idx.push_back(fileidx);
			db_data.i_labels.push_back(label);
			db_data.i_file_name.push_back(impath);
		}
		// Start with individual features.
		L.Mat2Vec(raw_feats[0], raw_mat, false);
		this->selectFV(raw_mat,'i'); // remove irrelevant features
		if (this->i_DB.cols) {
			vconcat(raw_mat, this->i_DB, this->i_DB); // add to main feature database
		} else {
			this->i_DB = raw_mat;
		}

		if(raw_feats[1].size() && raw_feats[1][0].size()){ // add cluster features
			for (size_t i = 0; i < raw_feats[1].size(); i++) { // Insert the file index for each cluster object feature
				db_data.c_file_idx.push_back(fileidx);
				db_data.c_labels.push_back(label);
				db_data.c_file_name.push_back(impath);
			}
			L.Mat2Vec(raw_feats[1], raw_mat, false);
			this->selectFV(raw_mat,'c'); // remove irrelevant features
			if (this->c_DB.cols) {
				vconcat(raw_mat, this->c_DB, this->c_DB); // add to main feature database
			} else {
				this->c_DB = raw_mat;
			}
		}
	}
	// End of function
}
void cFeature::selectFV(Mat &fv, char c_opt)
{
	// Select features from fv that passed relevance criteria. getSelData() should be called first
	// to set up the columns to be selected, before this method is called
	// If fv is not provided then selection is done on stored database. c_opt determines class of
	// features (i=individual, c=cluster, p=pca)
	if (fv.cols) {
		Mat t_DB;
		if('i'==c_opt && this->i_select.cols) { // select 35 individual features from original 129
			t_DB = fv.col(this->i_select.at<uchar>(0));
			for (size_t i = 1; i < this->i_select.cols; i++) {
				hconcat(t_DB, fv.col(this->i_select.at<uchar>(i)), t_DB);
			}
		} else if('c'==c_opt && this->c_select.cols) { // select 99 cluster features from original 144
			t_DB = fv.col(this->c_select.at<uchar>(0));
			for (size_t i = 1; i < this->c_select.cols; i++) {
				hconcat(t_DB, fv.col(this->c_select.at<uchar>(i)), t_DB);
			}
		} else if('p'==c_opt) {
			t_DB = fv.col(this->c_select.at<uchar>(0));
			for (size_t i = 1; i < this->p_select.cols; i++) {
				hconcat(t_DB, fv.col(this->p_select.at<uchar>(i)), t_DB);
			}
		}
		fv = t_DB;
	}
	// End of function
}
void cFeature::normalizeDB(Mat &fv, char c_opt, bool normalizeVec)
{
	// Prepare the normalization data as well as normalize the database (normalizeVec), OR,
	// normalize feature vector fv column-wise (normalizeVec=true).
	cLib L;
	if (normalizeVec) {
		if ('i'==c_opt && db_status.i_NORMALIZED) {
			for (size_t i = 0; i < fv.cols; i++) {
				fv.col(i) = (fv.col(i)-this->norm_data.i_mean[i])/this->norm_data.i_stdev[i];
			}
		} else if ('c'==c_opt && db_status.c_NORMALIZED) {
			for (size_t i = 0; i < fv.cols; i++) {
				fv.col(i) = (fv.col(i)-this->norm_data.c_mean[i])/this->norm_data.c_stdev[i];
			}
		} else fprintf(stderr, "Normalization variables (mu and Sigma) not initialized\n");
	} else {
		if (this->i_DB.data && 'i'==c_opt){
			L.getMeanStdev(this->i_DB, this->norm_data.i_mean, this->norm_data.i_stdev, false);
			db_status.i_NORMALIZED=true;
			this->normalizeDB(this->i_DB,'i',1);
		}	else if (this->c_DB.data && 'c'==c_opt){
			L.getMeanStdev(this->c_DB, this->norm_data.c_mean, this->norm_data.c_stdev, false);
			db_status.c_NORMALIZED=true;
			this->normalizeDB(this->c_DB,'c',1);
		}
	}
}
void cFeature::getSelData()
{
	// Get preselected features from Matlab data file
	ifstream idxfile("/media/femkha/HDB_FemkhaAcer/images/full//fromM2C");
	std::string str; std::vector<int> inputs[3]; size_t idx=0;
	while (std::getline(idxfile, str))
	{
		if (str.size()) {
			std::istringstream in; in.str(str);
			std::copy( std::istream_iterator<int>( in ), std::istream_iterator<int>(),  std::back_inserter( inputs[idx++] ) );
		}
	}

	this->i_select = Mat::zeros(1,inputs[0].size(), CV_8UC1); // first, initialize all to zero, de-selecting all features
	this->c_select = Mat::zeros(1,inputs[1].size(), CV_8UC1);
	this->p_select = Mat::zeros(1,inputs[2].size(), CV_8UC1);
	//i_select.colRange(i_select.cols-4,i_select.cols) *= 0; // strip off coordinates
	// Next: use features preselected using MATLAB version (read from text file)
	for (size_t i = 0; i < inputs[0].size(); i++) { // individual features
		this->i_select.at<uchar>(i) = inputs[0][i];
	}
	for (size_t i = 0; i < inputs[1].size(); i++) { // cluster features
		this->c_select.at<uchar>(i) = inputs[1][i];
	}
	idx=0;
	for (size_t i = 0; i < inputs[2].size(); i++) { // cluster features
		if(inputs[2][i]) // this vector is in zeros and ones
			this->p_select.at<uchar>(idx++) = i;
	}
	this->p_select = this->p_select.colRange(0,idx);
	// End of function
}
Mat cFeature::doPCA(Mat &vec, char opt)
{
	// Transform cluster features vec to PCA features. During generation, the generating vector is first cleaned
	// of all rows containing NaN/Inf values.	Selection not done here, do it after calling this function
	// issue: find a way of handling where there's no data for pca analysis after cleaning
	cLib L;
	string file_name = DATA_DIR + "pca_data.xml";
	if ('c'==opt) { // if no mat provided. set/train/prepare pca structure using pcadata
		pca_data.PCA_INIT = false;
		if (READ_PCA) { // Read PCA structure from file

			FileStorage fs(file_name,FileStorage::READ);
						if (fs.isOpened()) { // Read
							try {
								pca_data.metadata.read(fs.root());
								pca_data.PCA_INIT = true;
								std::cout << "PCA retrieved from File" << std::endl;
							} catch(...) { std::cout << "Error processing PCA file" << std::endl;;}
						} else std::cerr << "Cannot open file. Failed to initialize PCA structure" << std::endl;
		} else { // Generate PCA structure and save
			const int MAX_COMPONENTS = 0; // set to default (retain all components)
			try {
				Mat cleanVec = L.removeNanInf<DBType>(this->c_DB,1,1,1);
				if (cleanVec.data) {
					FileStorage fs(file_name,FileStorage::WRITE);
					//PCA pca(this->c_DB,Mat(),PCA::DATA_AS_ROW,MAX_COMPONENTS);
					PCA pca(cleanVec,Mat(),PCA::DATA_AS_ROW,MAX_COMPONENTS);
					pca.write(fs);
					fs.release();
					pca_data.metadata=pca;
					pca_data.PCA_INIT = true;
					std::cout << "PCA generated" << std::endl;
				} else std::cout << "No data for PCA analysis after cleaning" << std::endl;
			} catch(...){
				std::cout << "Failed to initialize PCA structure" << std::endl;; // all failed
			}
		}
	} else {
		if(true==pca_data.PCA_INIT){
			if('p' == opt){ // project sample given by 'vec'
				//CV_Assert( vec.cols == pcaset.cols );
				Mat coeff(vec.size(),vec.type());
				pca_data.metadata.project(vec, coeff);
				return coeff;
			} else if('r'==opt){// reconstruct
				Mat reconstructed(vec.size(),vec.type());
				pca_data.metadata.backProject(vec, reconstructed);
				return reconstructed;
			}
		} else std::cout << "PCA structure not initialized" << std::endl;
	}
	return Mat();
	// End of function
}
void cFeature::doSVM(Mat &data, Mat &labels, Mat &res, char opt)
{
	String svmpath = DATA_DIR + "svm_data.svm.xml";
	// Train the svm_data.svm structure using data
	Mat_<float> t_data; Mat_<int> t_labels;
	if (!READ_SVM && 't'==opt && i_DB.data) { // train
		// train individual feature machine
		t_data = i_DB.clone(); t_labels = Mat(db_data.i_labels);
		t_data = L.removeNanInf<double>(t_data,1,1,0); // trims down features. hope this remains steady
		svm_data.i_svm->train(t_data,ml::ROW_SAMPLE,t_labels);
		//svm_data.i_svm->save(svmpath);
		// cluster feature machine next
		t_data = c_DB.clone(); t_labels = Mat(db_data.c_labels);
		t_data = L.removeNanInf<double>(t_data,1,1,0); // trims down features. hope this remains steady
		svm_data.c_svm->train(t_data,ml::ROW_SAMPLE,t_labels);
		//svm_data.c_svm->save(svmpath);

		std::cout << svm_data.i_svm->getSupportVectors().size() << " -- " << svm_data.c_svm->getSupportVectors().size() << std::endl;
		std::cout << "SVM models trained and saved" << std::endl;
	} else if(READ_SVM && 't'==opt){ // data not provided, so load from file
		//cv::ml::SVM::load(svmpath); //svm_data.svm = ml::SVM::load<ml::SVM>load(svmpath);
		//svm_data.svm = Algorithm::load<ml::SVM>(svmpath);
		std::cout << "Loaded SVM model from " << svmpath<< std::endl;
	}	else if ('i'==opt && data.data) { // classify/predict
		svm_data.i_svm->predict(data, res);
	} else if ('c'==opt && data.data) { // classify/predict
		svm_data.c_svm->predict(data, res);
	} else std::cerr << "SVM error: illegal option or invalid parameter value(s)" << std::endl;
}
Mat cFeature::doLDA(Mat &vec, char opt)
{
	// Train (opt=='t') using saved database (i/c_DB) or project 'vec' based on
	// trained individual (opt=='i') or cluster (opt=='c') data.
	Mat out;
	if ('t'==opt) { // train both individual and cluster lda. assumption on existence of both databases
		if (READ_LDA) {
			qda_data.iLda.load(DATA_DIR+"lda.xml");
			std::cout << "LDA structure retrieved from file: "<< DATA_DIR+"lda.xml" << std::endl;
		} else {
			//reduce(i_DB, qda_data.i_mean, 0, CV_REDUCE_AVG);	qda_data.i_mean.convertTo(qda_data.i_mean, CV_64F);
			vector<double> i_mn, c_mn, tmp_mat;
			L.getMeanStdev(i_DB, i_mn, tmp_mat, 0, 1); L.getMeanStdev(c_DB, c_mn, tmp_mat, 0, 1);
			qda_data.i_mean = Mat(i_mn).reshape(1,1); qda_data.c_mean = Mat(c_mn).reshape(1,1);
			//std::cout << qda_data.c_mean << std::endl<< std::endl;
			qda_data.iLda.compute(i_DB, Mat(db_data.i_labels));	qda_data.cLda.compute(c_DB, Mat(db_data.c_labels));
			qda_data.iLda.save(DATA_DIR+"i_lda.xml"); qda_data.cLda.save(DATA_DIR+"c_lda.xml");
			std::cout << "LDA structure generated/trained and saved in files: " << DATA_DIR+"?_lda.xml" << std::endl;
		}
		qda_data.lda_trained = true;
	} else { // project
		if ('i'==opt && qda_data.lda_trained)
			out = qda_data.iLda.subspaceProject(qda_data.iLda.eigenvectors(), qda_data.i_mean, vec);
		else if ('c'==opt && qda_data.lda_trained)
			out = qda_data.cLda.subspaceProject(qda_data.cLda.eigenvectors(), qda_data.c_mean, vec);
		else std::cerr << "Cannot project vector. Wrong option or LDA structure not trained" << std::endl;
	}
	return out;
	// some of ideas borrowed from: http://answers.opencv.org/question/64165/how-to-perform-linear-discriminant-analysis-with-opencv/
	// and http://stackoverflow.com/questions/32001390/what-is-correct-implementation-of-lda-linear-discriminant-analysis
}
void cFeature::generateDB()
{
	// Set up training data
	getDB();
	// Do post-processing on raw vectors
	if(this->i_DB.data && this->c_DB.data){
		//this->saveClassData(false);
		if(this->i_DB.data)	{
			normalizeDB((Mat &)noArray(),'i',0); // Normalize the database
		}
		if(this->c_DB.data)	{
			normalizeDB((Mat &)noArray(),'c',0);
			std::cout << "Doing PCA analysis on data" << std::endl;
			doPCA((Mat &)noArray(),'c'); selectFV(this->c_DB,'p'); // Convert cluster DB to PCA domain space and select relevant components
		}
	}
	doLDA((Mat &)noArray(),'t'); doLDA(i_DB,'i');	doLDA(c_DB,'c');

	// End of function
}
void cFeature::getDB()
{
	// Acquire (either by reading in saved file or extracting from scratch) the database into class variables i/c_DB
	// set MAX_FILE_COUNT = -1 to loop over all images, or to any value to restrict to it
	const int MAX_FILE_COUNT = -1;
	std::vector<bool> t_opts(4,false);  // +ve train, -ve train, +ve test, -ve test;
	t_opts[0] = 1; t_opts[1] = 1;
	int LABEL; string dir;//, s_iDb1 = DATA_DIR+"iDb1.xml", s_cDb1 = DATA_DIR+"cDb1.xml";// = "/home/femkha/db_par/1.pgm";

	// Get raw feature vectors
	if (READ_DB) { // Read in saved databases
		this->saveClassData(true);
		if(i_DB.data)	std::cout << "Individual calcification features read from file "<< std::endl;
		else std::cerr << "Cannot read Individual calcification features read from file "<< std::endl;
		if(c_DB.data)	std::cout << "Cluster calcification features read from file "<< std::endl;
		else std::cerr << "Cannot read Cluster calcification features read from file "<< std::endl;
	} else { // Otherwise, generate and save databases
		for (size_t i = 0; i < t_opts.size(); i++) {
			if (t_opts[i] && i==0) { // Get positive training cases
				LABEL = 1; dir = trainPosDir;
			} else if (t_opts[i] && i==1) { // Get negative training cases
				LABEL = 0; dir = trainNegDir;
			} else if (t_opts[i] && i==2) { // Get positive test samples
				LABEL = 1; dir = testPosDir;
			} else if (t_opts[i] && i==3) {  // Get negative test samples
				LABEL = 0; dir = testNegDir;
			} else continue;
			//dir="/media/femkha/HDB_FemkhaAcer/images/mias-database/";
			//F.addFV(1,string("/media/femkha/HDB_FemkhaAcer/images/full/calc/mdb212.pgm"),0);

			DIR *dp; struct dirent *entry;
		    if( ( dp=opendir(dir.c_str()) ) != NULL ){
					short file_cnt = 0; bool maxx = (MAX_FILE_COUNT>=0)?file_cnt<MAX_FILE_COUNT:true;
					while((entry=readdir(dp)) && maxx ){ //
						// Note: d_type might be incompatible with windows..! use (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) instead
						if(entry->d_type==DT_REG && regex_match (entry->d_name,regex(string(".*\\.(")+EXTS+")",regex_constants::icase))){
							string impath = dir+entry->d_name;
							cout<<impath<<endl;
							addFV(file_cnt,impath,LABEL);
							++file_cnt;
							maxx = (MAX_FILE_COUNT>=0)?file_cnt<MAX_FILE_COUNT:true;
						}
					}
					(void) closedir(dp);
				} else {
					perror("Couldn't open the directory.");
		    }
		} // end for
		// Next: store raw, unnormalized, selected individual + cluster features
		//L.serializeMat(s_iDb1,this->i_DB,false); L.serializeMat(s_cDb1,this->c_DB,false);
	}

	// End of function
}
void cFeature::saveClassData(bool o_READ)
{
	// Save (o_READ==false) or Restore (o_READ == true) class attributes
	// issue: fix the serialization of filenames (this->db_data.c_file_name)
	Mat out; string impath = DATA_DIR+"db_data.xml";
	if (o_READ) {
		FileStorage fs(impath.c_str(), FileStorage::READ);
		if (fs.isOpened()){
			// retrieve individual database
			fs["i_DB"]>>this->i_DB;
			fs["db_data__i_file_idx"]>>this->db_data.i_file_idx;
			fs["db_data__i_labels"]>>this->db_data.i_labels;
			//fs["db_data__i_file_name"]>>this->db_data.i_file_name;
			// retrieve cluster database
			fs["c_DB"]>>this->c_DB;
			fs["db_data__c_file_idx"]>>this->db_data.c_file_idx;
			fs["db_data__c_labels"]>>this->db_data.c_labels;
			//fs["db_data__c_file_name"]>>this->db_data.c_file_name;
			// retrieve other attributes
			fs["db_status__i_NORMALIZED"]>>this->db_status.i_NORMALIZED;
			fs["db_status__c_NORMALIZED"]>>this->db_status.c_NORMALIZED;
			fs["db_status__SELECTED"]>>this->db_status.SELECTED;
			fs["i_select"]>>this->i_select;
			fs["c_select"]>>this->c_select;
			fs["p_select"]>>this->p_select;
			fs["pca_data__PCA_INIT"]>>this->pca_data.PCA_INIT;
		} else cerr << "Serialization error: failed to open " << impath << endl;
		fs.release();
	} else {
		cv::FileStorage fs(impath.c_str(), cv::FileStorage::WRITE);
		fs << "i_DB" << this->i_DB;
		fs << "db_data__i_file_idx" << this->db_data.i_file_idx;
		fs << "db_data__i_labels" << this->db_data.i_labels;
		fs << "db_data__i_file_name" << this->db_data.i_file_name;
		// store cluster features
		fs << "c_DB" << this->c_DB;
		fs << "db_data__c_file_idx" << this->db_data.c_file_idx;
		fs << "db_data__c_labels" << this->db_data.c_labels;
		fs << "db_data__c_file_name" << this->db_data.c_file_name;
		// store other attributes
		fs << "db_status__i_NORMALIZED" << this->db_status.i_NORMALIZED;
		fs << "db_status__c_NORMALIZED" << this->db_status.c_NORMALIZED;
		fs << "db_status__SELECTED" << this->db_status.SELECTED;
		fs << "i_select" << this->i_select;
		fs << "c_select" << this->c_select;
		fs << "p_select" << this->p_select;
		fs << "pca_data__PCA_INIT" << this->pca_data.PCA_INIT;
		fs.release();
	}
	// End of function
}
