#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
using namespace cv;
using namespace cv::dnn;
using namespace std;

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat& frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    Scalar meanVal = Scalar(104, 117, 123);
    Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);
    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    vector<vector<int>> bboxes;

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = { x1, y1, x2, y2 };
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
        }
    }
    return make_tuple(frameOpenCVDNN, bboxes);
}

int main(int argc, char** argv)
{
    string faceProto = "opencv_face_detector.pbtxt";
    string faceModel = "opencv_face_detector_uint8.pb";

    string ageProto = "age_deploy.prototxt";
    string ageModel = "age_net.caffemodel";

    string genderProto = "gender_deploy.prototxt";
    string genderModel = "gender_net.caffemodel";

    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

    vector<string> ageList = { "(0-4)", "(5-9)", "(10-13)", "(14-16)", "(17-20)", "(21-25)", "(26-30)",
    "(34-40)", "(48-54)", "(60-100)" };

    vector<string> genderList = { "Male", "Female" };

    string device = "cpu";

    string videoFile = "0";

    if (argc == 2)
    {
        if ((string)argv[1] == "gpu")
            device = "gpu";
        else if ((string)argv[1] == "cpu")
            device = "cpu";
        else
            videoFile = argv[1];
    }
    else if (argc == 3)
    {
        videoFile = argv[1];
        if ((string)argv[2] == "gpu")
            device = "gpu";
    }

    Net ageNet = readNet(ageModel, ageProto);
    Net genderNet = readNet(genderModel, genderProto);
    Net faceNet = readNet(faceModel, faceProto);

    if (device == "cpu")
    {
        cout << "Using CPU device" << endl;
        ageNet.setPreferableBackend(DNN_TARGET_CPU);

        genderNet.setPreferableBackend(DNN_TARGET_CPU);

        faceNet.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
        cout << "Using GPU device" << endl;
        ageNet.setPreferableBackend(DNN_BACKEND_CUDA);
        ageNet.setPreferableTarget(DNN_TARGET_CUDA);

        genderNet.setPreferableBackend(DNN_BACKEND_CUDA);
        genderNet.setPreferableTarget(DNN_TARGET_CUDA);

        faceNet.setPreferableBackend(DNN_BACKEND_CUDA);
        faceNet.setPreferableTarget(DNN_TARGET_CUDA);
    }
    VideoCapture cap(0);
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    cap.open(deviceID, apiID);
    if (!cap.isOpened())
    {
        return -1;
    }
    
    int padding = 20;
    while (true) try {
        Mat frame;
        cap >> frame;
        if (!cap.read(frame))break;
        if (frame.empty())
        {
            cout << "ERROR! blank frame grabbed\n";
            waitKey(1);
            break;
        }
        vector<vector<int>> bboxes;
        Mat frameFace;
        tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);

        if (bboxes.size() == 0) {
            cout << "No face detected, checking next frame." << endl;
            continue;
        }
        for (auto it = begin(bboxes); it != end(bboxes); ++it) {
            Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2 * padding, it->at(3) - it->at(1) + 2 * padding);
            Mat face = frame(rec);

            Mat blob;
            blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
            genderNet.setInput(blob);
            vector<float> genderPreds = genderNet.forward();
            int max_index_gender = distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
            string gender = genderList[max_index_gender];
            cout << "Gender: " << gender << endl;
            ageNet.setInput(blob);
            vector<float> agePreds = ageNet.forward();
            int max_indice_age = distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
            string age = ageList[max_indice_age];
            cout << "Age: " << age << endl;
            string label = gender + ", " + age;
            putText(frameFace, label, Point(it->at(0), it->at(1) - 15), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
            imshow("Gender Classfication", frameFace);
            //imwrite("output.jpg", frameFace);
            if (waitKey(15) == 27)
                    break;
        }
        destroyAllWindows();
    }
    catch (exception& ex) {
        cout << "Errorrrrrrr\n" << ex.what() << endl;
    }
}