#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char *keys =
    "{help h usage ? |      | Program by: Maksymilian Jackowski }"
    "{image i |<none>| input image}"
    "{video v |<none>| input video}"
    "{camera c |<none>| use camera}";

// GLOBAL VAR
std::vector<std::string> classes;

int inpWidth = 608;
int inpHeight = 608;
int outWidth = 608;
int outHeight = 608;
float confThreshold = 0.5;
float nmsThreshold = 0.4;

// FUNCTION
std::vector<std::string> getOutputsNames(const cv::dnn::Net &net);

void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs);

int main(int argc, char **argv)
{
    // config vars
    std::string classesFile = "yolo/coco.names";
    std::ifstream file(classesFile.c_str());
    std::string line;

    while (std::getline(file, line))
        classes.push_back(line);

    cv::CommandLineParser parser(argc, argv, keys);

    std::string modelConfguration = "yolo/yolov3.cfg";
    std::string modelWeights = "yolo/weights/yolov3_training_last_best.weights";
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // output vars
    std::string str, outputFile;
    cv::VideoCapture capture;
    cv::VideoWriter video;
    cv::Mat frame, blob;

    try
    {
        if (parser.has("camera"))
        {
            outputFile = "cameraView_OUT.avi";
            if (!capture.open(0))
                throw std::runtime_error("error");
            std::cout << "camera opened" << std::endl;
        }
        else if (parser.has("video"))
        {
            str = parser.get<std::string>("video");
            std::ifstream ifile(str);
            if (!ifile.is_open())
                throw std::runtime_error("error");
            capture.open(str);
            str.replace(str.end() - 4, str.end(), "_OUT.avi");
            outputFile = str;
            std::cout << "video uploaded. " << outputFile << std::endl;
        }
        else if (parser.has("image"))
        {
            str = parser.get<std::string>("image");
            std::ifstream ifile(str);
            if (!ifile.is_open())
                throw std::runtime_error("error");
            capture.open(str);
            str.replace(str.end() - 4, str.end(), "_OUT.jpg");
            outputFile = str;
            std::cout << "image uploaded. " << outputFile << std::endl;
        }
        else if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
    }
    catch (...)
    {
        std::cout << "Error while trying open image/video. Maybe wrong path ;)" << std::endl;
        return 0;
    }

    if (!parser.has("image") || !parser.has("help"))
        video.open(outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 28,
                   cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    static const std::string winName = "Car counter";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    while (cv::waitKey(1) < 0)
    {
        capture >> frame;
        if (frame.empty() && !parser.has("image"))
        {
            std::cout << "Koniec materialu!" << std::endl;
            break;
        }

        cv::dnn::blobFromImage(frame, blob, 1 / 255, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames(net));

        postprocess(frame, outs);

        cv::imshow(winName, frame);
    }

    cv::VideoCapture cap;
    return 0;
}

std::vector<std::string> getOutputsNames(const cv::dnn::Net &net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (std::size_t i = 0; i < outLayers.size(); i++)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (std::size_t i = 0; i < outs.size(); i++)
    {
        float *data = (float *)outs[i].data;
        for (std::size_t j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int cx = (int)(data[0] * frame.cols);
                int cy = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = cx - width / 2;
                int top = cy - height / 2;
                boxes.push_back(cv::Rect(left, top, width, height));
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (std::size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(255, 178, 50), 3);
    }
}
