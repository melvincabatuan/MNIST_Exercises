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
std::string filename_to_save = "BOOST_MNIST.xml";


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



   
 




/************** BEGIN CLASSIFIER TRAIN AND TEST ******************/

    // 4. Create classifier and Train 

 
    const int class_count = 10; // {0,1,...,9}

    Mat weak_responses;

    int i, j, k;

    Ptr<Boost> model;

    int nsamples_all = data.rows;
    int ntrain_samples = NUM_TRAIN_IMAGES;
    int var_count = data.cols;


        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // As currently boosted tree classifier in MLL can only be trained
        // for 2-class problems, we transform the training database by
        // "unrolling" each training sample as many times as the number of
        // classes (26) that we have.
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat new_data( ntrain_samples*class_count, var_count + 1, CV_32F );
        Mat new_responses( ntrain_samples*class_count, 1, CV_32S );

        // 1. Unroll the database type mask
        printf( "Unrolling the database...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            const float* data_row = data.ptr<float>(i);
            for( j = 0; j < class_count; j++ )
            {
                float* new_data_row = (float*)new_data.ptr<float>(i*class_count+j);
                memcpy(new_data_row, data_row, var_count*sizeof(data_row[0]));
                new_data_row[var_count] = (float)j;
                new_responses.at<int>(i*class_count + j) = responses.at<int>(i) == j+'0';
            }
        }

        Mat var_type( 1, var_count + 2, CV_8U );
        var_type.setTo(Scalar::all(VAR_ORDERED));
        var_type.at<uchar>(var_count) = var_type.at<uchar>(var_count+1) = VAR_CATEGORICAL;

        Ptr<TrainData> tdata = TrainData::create(new_data, ROW_SAMPLE, new_responses,
                                                 noArray(), noArray(), noArray(), var_type);
        vector<double> priors(2);
        priors[0] = 1;
        priors[1] = 26;

        cout << "Training the classifier (may take a few minutes)...\n";
        model = Boost::create();
        model->setBoostType(Boost::GENTLE);
        model->setWeakCount(100);
        model->setWeightTrimRate(0.95);
        model->setMaxDepth(5);
        model->setUseSurrogates(false);
        //model->setPriors(Mat(priors));
        model->train(tdata);
        cout << endl;




     Mat temp_sample( 1, var_count + 1, CV_32F );
    float* tptr = temp_sample.ptr<float>();

    // compute prediction error on train and test data
    double train_hr = 0, test_hr = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class = 0;
        double max_sum = -DBL_MAX;
        const float* ptr = data.ptr<float>(i);
        for( k = 0; k < var_count; k++ )
            tptr[k] = ptr[k];

        for( j = 0; j < class_count; j++ )
        {
            tptr[var_count] = (float)j;
            float s = model->predict( temp_sample, noArray(), StatModel::RAW_OUTPUT );
            if( max_sum < s )
            {
                max_sum = s;
                best_class = j + '0';
            }
        }

        double r = std::abs(best_class - responses.at<int>(i)) < FLT_EPSILON ? 1 : 0;
        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all-ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
    printf( "Recognition rate: train = %.2f%%, test = %.2f%%\n",
            train_hr*100., test_hr*100. );

    cout << "Number of trees: " << model->getRoots().size() << endl;

    // Save classifier to file if needed
    if( !filename_to_save.empty() )
        model->save( filename_to_save );
 

/************** END CLASSIFIER TRAIN AND TEST ******************/

    return 0;
}
