/*****************************************************************************/
/*

The MIT License (MIT)

Copyright (c) Melvin Cabatuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 

/*****************************************************************************/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <fstream>
#include <iostream>
#include <cmath> 

#define DEBUG 1

/// OpenCV
using cv::approxPolyDP;
using cv::findContours;
using cv::getTickCount;
using cv::getTickFrequency;
using cv::imshow;
using cv::Mat;
using cv::namedWindow;
using cv::noArray;
using cv::Ptr;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using cv::Size;
using cv::TermCriteria;
using cv::Vec4i;
using cv::waitKey;

using namespace cv::ml;


/// std
using std::ifstream;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ios;
using std::abs;


/// MNIST Dataset
#define NUM_TEST_IMAGES 10000
#define NUM_TRAIN_IMAGES 60000  
#define IMAGE_SIZE (28*28)

const string trainImagesFilename = "train-images-idx3-ubyte";
const string trainLabelsFilename = "train-labels-idx1-ubyte"; 
const string testImagesFilename  = "t10k-images-idx3-ubyte";
const string testLabelsFilename  = "t10k-labels-idx1-ubyte";


/// Classifier 
const string filename_to_load = "rtrees_roi.xml";

 

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_Mnist_Data(string filename, vector<cv::Mat> &vec){
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


/// Sets active data for training set only (taken from OpenCV letter_recog.cpp sample)
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



static void test_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses, int rdelta)
{
    int i; 
    //int nsamples_all = data.rows;
    int nsamples_all = 10;
    double test_hr = 0;

    // compute prediction error on 
    // train data[0 ; ...; ntrain_samples-1]; and 
    // test data[0 ; ...; nsamples_all-1]
 

    double before = static_cast<double>(getTickCount());

    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);

   
        // The method is used to predict the response for a new sample. 
        // In case of a classification, the method returns the class label.        

        float r = model->predict( sample );  /// sample is the row feature vector

        cout <<" result("<< i <<") = "<< r << "\n";
        cout <<" label("<< i <<") = "<< responses.at<int>(i) << "\n";
        cout <<" \n ";

        

        // Tally correct classifications
        // +1 if prediction is correct
        // +0 if prediction is wrong 

        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
        test_hr += r;
    }

    double after = static_cast<double>(getTickCount());
    double duration_in_ms = 1000.0*(after - before)/getTickFrequency();
    cout << "Prediction for all data completed after "<< duration_in_ms <<" ms...\n";
    cout << "Average prediction time per sample is "<< duration_in_ms/nsamples_all <<" ms.\n";

    test_hr /= nsamples_all;
    /// Note: 0 training samples here will give 100 % training error


    printf( "Recognition rate: test = %.2f%%\n", test_hr*100. );
}





bool isCentral(Point center){
   return ( (abs(center.x - 14) < 4) && (abs(center.y - 14) < 4) );
}



void 
preprocess(vector<cv::Mat> data, Mat& processed, std::vector<int> dataVecLabels, std::vector<int> &vecLabels){

    //cv::RNG rng(12345);
    Mat blurred;  // blurred image 
    Mat thresh;   // thresholded image
    vector< vector<cv::Point> > contours; // detected contours
    vector<cv::Vec4i> hierarchy;        // contours hierarchy  

    /// Default rectangle roi
    Rect r(Point(4, 4), Point(24, 24)); // digits are 20x20 inside 28x28 region

    /// Insert dataVecLabels vector labels to vecLabels (contains all labels)
    vecLabels.insert( vecLabels.end(), dataVecLabels.begin(), dataVecLabels.end() );

    /// Convert a vector of Mat data into one Mat containing image features (all pixels) per row 
    vector<Mat>::const_iterator img = data.begin();    
    vector<Mat>::const_iterator end = data.end();

    for( int index = 0; img != end ; ++img, ++index )
    {

       /// PREPROCESS HERE .....        
       blur( *img, blurred, Size(3,3) );  // blur image  Note: img is a pointer    
       threshold( blurred, thresh,0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );        

       /// Find contours   
       //findContours( thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
       findContours( thresh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) ); // get only the EXTERNAL contour

       // cout << "contours.size() = "<< contours.size() << endl;

       /// if contours are found, update roi rectangle
       if(contours.size() > 0){

         /// Approximate contours to polygons + get bounding circles
         vector< vector<cv::Point> > contours_poly( contours.size() );
         vector<cv::Point2f>center( contours.size() );
         vector<float>radius( contours.size() );

         for( int i = 0; i < contours.size(); i++ )
         { 
             approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
             minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
         }

          // int squareCount = 0;           

          for( int i = 0; i < contours.size(); i++ )
          {
    
              if(radius[i] > 7 && isCentral(center[i])){  // isCentral() avoids roi outside 28x28 image
                   
                 // squareCount++;  

                 // cout<< "squareCount = " << squareCount << endl;
     
                 // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                 
                 /// Center can go out of hand; controlled by isCentral(), note: digits are 20x20 inside 28x28
                 r = Rect(Point(center[i].x - 10, center[i].y - 10), Point(center[i].x + 10, center[i].y + 10));  

               } // END if
           }// END for
      }// END if(contours.size() > 0)



          /// Crop image 
          Mat ROI = thresh(r);  

          /// Resize to 20X20; preparation during detection
          Mat tmp1, tmp2;
          resize(ROI,tmp1, Size(20,20), 0, 0, cv::INTER_LINEAR ); //unnecessary for now

/// Debugging
          //imshow("ROI", ROI);
          //cout << "vecLabels[" << index <<"] = " << vecLabels[index] << endl;
          //waitKey(1); // 1 ms

          /// Convert to float
          tmp1.convertTo(tmp2,CV_32FC1); 

          /// Reshape into row
          Mat row = tmp2.reshape(1,1);

          /// Push into processed data
          processed.push_back(row);

    } // END for imgIteration
} // END preprocess method








inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}
/*
At each node the recursive procedure may stop (that is, stop splitting the node further) in one of the following cases:

1. Depth of the constructed tree branch has reached the specified maximum value.
2. Number of training samples in the node is less than the specified threshold 
when it is not statistically representative to split the node further.
3. All the samples in the node belong to the same class or, in case of regression, the variation is too small.
4. The best found split does not give any noticeable improvement compared to a random choice.
*/





template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}
















int main(void)
{


/************************ LOADING MNIST BEGIN ************************************/

    // 1. Read MNIST training images into a std::vector of OpenCV Mat objects    
    std::vector<cv::Mat> trainData;
    read_Mnist_Data(trainImagesFilename.c_str(), trainData);

    // 2. Read MNIST training labels into std::vector       
    std::vector<int> trainVecLabels(NUM_TRAIN_IMAGES);
    read_Mnist_Label(trainLabelsFilename.c_str(), trainVecLabels);

# ifdef DEBUG

    cout<< "trainData.size() = " << trainData.size() <<endl;
    namedWindow("1st trainData", CV_WINDOW_NORMAL);
    cout<< "trainVecLabels.size() = " << trainVecLabels.size() <<endl;
    cout<< "trainVecLabels[0] = " << trainVecLabels[0] <<endl;
    imshow("1st trainData", trainData[0]);
    waitKey(0);
 
# endif

    // 3. Read MNIST test images into a std::vector of OpenCV Mat    
    std::vector<cv::Mat> testData;
    read_Mnist_Data(testImagesFilename.c_str(), testData);

    // 4. Read MNIST test labels into vector       
    std::vector<int> testVecLabels(NUM_TEST_IMAGES);
    read_Mnist_Label(testLabelsFilename.c_str(), testVecLabels);

# ifdef DEBUG

    cout<< "testData.size() = " << testData.size() <<endl;
    namedWindow("1st Test", CV_WINDOW_NORMAL);
    imshow("1st Test", testData[0]);
    waitKey(0);
 
    cout<< "testVecLabels.size() = " << testVecLabels.size() <<endl;
    cout<< "testVecLabels[0] = " << testVecLabels[0] <<endl;

# endif


    // 5. Preprocess (if NONE just convert images to rows)

    Mat allData;
    std::vector<int> vecLabels;
    vecLabels.reserve( NUM_TRAIN_IMAGES + NUM_TEST_IMAGES ); // preallocate memory

    double before_tick_count = static_cast<double>(getTickCount());
   
    preprocess(  trainData, allData, trainVecLabels, vecLabels);
    preprocess(  testData, allData, testVecLabels, vecLabels);

    double after_tick_count = static_cast<double>(getTickCount());
    double duration_in_ms = 1000.0*(after_tick_count - before_tick_count) / getTickFrequency();
 
    Mat responses;
    Mat(vecLabels).copyTo(responses);

# ifdef DEBUG

     cout<< " allData   - Mat containing all data elements; 1 image per row. " <<endl;
     cout<< " allData.size() = " << allData.size() <<endl;
     cout<< "\n responses - Mat containing all labels; 1 label per row. " <<endl;
     cout<< " responses.size() = " << responses.size() <<endl;
     cout<< "\n Preprocessing took "<< duration_in_ms/60000 << " mins." <<endl;     

# endif


/************************ LOADING MNIST END ************************************/

// allData   - Mat containing all data elements; 1 image per row (ROW_SAMPLE)
// responses - Mat containing all labels; column vector

 
// Loading classifier:

        before_tick_count = static_cast<double>(getTickCount());

        cout << "Loading the classifier ...\n";
    
        Ptr<RTrees> model = load_classifier<RTrees>(filename_to_load);

        if( model.empty() )
            return false;
       
        after_tick_count = static_cast<double>(getTickCount());
        duration_in_ms = 1000.0*(after_tick_count - before_tick_count) / getTickFrequency();
         
        cout<< "\n Loading took "<< duration_in_ms/60000 << " mins." <<endl;    

        cout << "Testing ...\n";   
        test_classifier(model, allData, responses, 0);
 

    return 0;
}
