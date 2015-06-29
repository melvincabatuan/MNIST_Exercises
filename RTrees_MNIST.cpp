#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <fstream>
#include <iostream>

using cv::Ptr;
using cv::Mat;
using cv::imshow;
using cv::waitKey;
using cv::namedWindow;
using cv::Scalar;
using cv::noArray;
using cv::getTickCount;
using cv::getTickFrequency;
using cv::TermCriteria;

using namespace cv::ml;
 
using std::ifstream;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ios;
using std::abs;

#define DEBUG 1


/// MNIST Dataset
#define NUM_TEST_IMAGES 10000
#define NUM_TRAIN_IMAGES 60000  
#define IMAGE_SIZE (28*28)

std::string imagesFilename = "train-images-idx3-ubyte";
std::string labelsFilename = "train-labels-idx1-ubyte"; 
std::string testImagesFilename = "t10k-images-idx3-ubyte";
std::string testLabelsFilename = "t10k-labels-idx1-ubyte"; 
std::string outputFilename = "RTrees_MNIST.xml";


/* Global variables to keep track of time*/
double before, after, duration_in_ms;  


int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}



void read_Mnist(string filename, vector<cv::Mat> &vec){
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_train_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_train_images,sizeof(number_of_train_images));
        number_of_train_images = ReverseInt(number_of_train_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_train_images; ++i)
        {
            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}



void read_Mnist_Label(string filename, vector<int> &vec)
{
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_train_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_train_images,sizeof(number_of_train_images));
        number_of_train_images = ReverseInt(number_of_train_images);

        for(int i = 0; i < number_of_train_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}



/// Limits the training to the ntrain_samples (80%) in the source

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}





static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save)
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;

    // compute prediction error on 
    // train data[0 , ..., ntrain_samples-1]; and 
    // test data[0 , ..., nsamples_all-1]
 

    before = static_cast<double>(getTickCount());

    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

   
        // The method is used to predict the response for a new sample. 
        // In case of a classification, the method returns the class label.        

        float r = model->predict( sample );  /// sample is the row feature vector
        

        // Tally correct classifications
        // +1 if prediction is correct
        // +0 if prediction is wrong 
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    after = static_cast<double>(getTickCount());
    duration_in_ms = 1000.0*(after - before)/getTickFrequency();
    cout << "Prediction for all data completed after "<< duration_in_ms <<" ms...\n";
    cout << "Average prediction time per sample is "<< duration_in_ms/nsamples_all <<" ms.\n";

    test_hr /= nsamples_all - ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
    /// Note: 0 training samples here will give 100 % training error


    printf( "Recognition rate: train = %.2f%%, test = %.2f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
}





inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}





int main(void)
{

  

/********************** BEGIN LOAD MNIST ************************/

    // 1. Read MNIST training images into OpenCV Mat vector
    
    std::vector<cv::Mat> trainData;
    read_Mnist(imagesFilename.c_str(), trainData);

# ifdef DEBUG

    cout<< "trainData.size() = " << trainData.size() <<endl;
    namedWindow("1st", CV_WINDOW_NORMAL);
    imshow("1st", trainData[0]);
    waitKey();
 
#endif

    // 2. Read MNIST labels into vector
       
    std::vector<int> trainVecLabels(NUM_TRAIN_IMAGES);
    read_Mnist_Label(labelsFilename.c_str(), trainVecLabels);

# ifdef DEBUG
    cout<< "trainVecLabels.size() = " << trainVecLabels.size() <<endl;
    cout<< "trainVecLabels[0] = " << trainVecLabels[0] <<endl;
#endif


    // 3. Read MNIST test images into OpenCV Mat vector
    
    std::vector<cv::Mat> testData;
    read_Mnist(testImagesFilename.c_str(), testData);

# ifdef DEBUG

    cout<< "testData.size() = " << testData.size() <<endl;
    namedWindow("1stTest", CV_WINDOW_NORMAL);
    imshow("1stTest", testData[0]);
    waitKey();
 
#endif

    // 4. Read MNIST test labels into vector
       
    std::vector<int> testVecLabels(NUM_TEST_IMAGES);
    read_Mnist_Label(testLabelsFilename.c_str(), testVecLabels);

# ifdef DEBUG
    cout<< "testVecLabels.size() = " << testVecLabels.size() <<endl;
    cout<< "testVecLabels[0] = " << testVecLabels[0] <<endl;
#endif




    // 5. Preprocess (if none just convert images to rows)
    Mat data;          // data for training


    std::vector<cv::Mat>::const_iterator img = trainData.begin();    
    std::vector<cv::Mat>::const_iterator end = trainData.end();

    for( ; img != end ; ++img )
    {
       /// PREPROCESS TRAIN IMAGES HERE
       Mat temp;
       img->convertTo(temp, CV_32FC1);
       Mat row = temp.reshape(1,1);
       data.push_back(row); 
    }

   
    img = testData.begin();    
    end = testData.end();

    for( ; img != end ; ++img )
    {
       /// PREPROCESS TEST IMAGES HERE
       Mat temp;
       img->convertTo(temp, CV_32FC1);
       Mat row = temp.reshape(1,1);
       data.push_back(row); 
    }
  
    cout<< "data.size() = " << data.size() <<endl;


     // Augment labels
     std::vector<int> vecLabels;
     vecLabels.reserve( trainVecLabels.size() + testVecLabels.size() ); // preallocate memory
     vecLabels.insert( vecLabels.end(), trainVecLabels.begin(), trainVecLabels.end() );
     vecLabels.insert( vecLabels.end(), testVecLabels.begin(), testVecLabels.end() );



    // Convert vecLabels to Mat
    Mat responses;
    Mat(vecLabels).copyTo(responses);
    cout<< "responses.size() = " << responses.size() <<endl;


/********************** END LOAD MNIST ************************/

/// Mat data - contains all features; 1 feature vector per row
/// Mat responses - contains all labels; 1 label per row



   
    Ptr<TrainData> tdata = prepare_train_data(data, responses, NUM_TRAIN_IMAGES);




/************** BEGIN CLASSIFIER TRAIN AND TEST ******************/

    // 4. Create classifier and Train 

    cout << "Training the classifier ...\n";

Ptr<RTrees> model = RTrees::create();
         model->setMaxDepth(10);
         model->setMinSampleCount(10);
         model->setRegressionAccuracy(0);
         model->setUseSurrogates(false);
         model->setMaxCategories(15);
         model->setPriors(Mat());
         model->setCalculateVarImportance(true);
         model->setActiveVarCount(4);
         model->setTermCriteria(TC(100,0.01f));

     before = static_cast<double>(getTickCount());    
   
             model->train(tdata);

     after =  static_cast<double>(getTickCount()); 
     duration_in_ms = 1000.0*(after - before)/getTickFrequency();
 
    cout << "Training completed after "<< duration_in_ms <<" ms...\n";
   
    



    cout << "Testing and saving ...\n";   
    test_and_save_classifier(model, data, responses, NUM_TRAIN_IMAGES, 0, outputFilename);



    cout << "Number of trees: " << model->getRoots().size() << endl;

  
    // Print variable importance
    Mat var_importance = model->getVarImportance();
    if( !var_importance.empty() )
    {
        double rt_imp_sum = sum( var_importance )[0];
        printf("var#\timportance (in %%):\n");
        int i, n = (int)var_importance.total();
        for( i = 0; i < n; i++ )
            printf( "%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i)/rt_imp_sum);
    }


/************** END CLASSIFIER TRAIN AND TEST ******************/

    return 0;
}
