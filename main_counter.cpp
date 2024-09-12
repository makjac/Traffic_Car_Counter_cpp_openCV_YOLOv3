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

float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 320;
int inpHeight = 320;
int outWidth = 320;
int outHeight = 320;
std::vector<std::string> classes;
std::vector<int> centX;
std::vector<int> centXH;

int vehicle[4] = {0, 0, 0, 0};

void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &out);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, bool color);

std::vector<std::string> getOutputsNames(const cv::dnn::Net &net);

int main(int argc, char **argv)
{

    cv::CommandLineParser parser(argc, argv, keys);

    std::string classesFile = "yolo/coco.names";
    std::ifstream file(classesFile.c_str());
    std::string line;

    while (std::getline(file, line))
        classes.push_back(line);

    std::string modelConfiguration = "yolo/yoloc.cfg";
    // Note: The 'yolov3_.weights' file is included in the repository but is compressed in a RAR archive. Please extract the weights from the RAR file before running the program.
    std::string modelWeights = "yolo/weights/yolov3_.weights";

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::string str, outputFile;
    cv::VideoCapture capture;
    cv::VideoWriter video;
    cv::Mat frame, blob;

    try
    {
        if (parser.has("camera"))
        {
            str = "cameraView_OUT.avi";
            capture.open(0);
            outputFile = str;
            std::cout << "camera" << std::endl;
        }
        else if (parser.has("image"))
        {
            str = parser.get<std::string>("image");
            std::ifstream ifile(str);
            if (!ifile)
                throw("error");
            capture.open(str);
            str.replace(str.end() - 4, str.end(), "_OUT.jpg");
            outputFile = str;
            std::cout << "image uploaded. " << outputFile << std::endl;
        }
        else if (parser.has("video"))
        {
            str = parser.get<std::string>("video");
            std::ifstream ifile(str);
            if (!ifile)
                throw("error");
            capture.open(str);
            str.replace(str.end() - 4, str.end(), "_OUT.avi");
            outputFile = str;
            std::cout << "video uploaded. " << outputFile << std::endl;
        }
        else if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
        else
            capture.open(parser.get<int>("device"));
    }
    catch (...)
    {
        std::cout << "Error while trying open image/video. Maybe wrong path ;)" << std::endl;
        return -1;
    }

    if (!parser.has("image"))
        video.open(outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    static const std::string winName = "Car counter";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    while (cv::waitKey(1) < 0)
    {

        capture >> frame;
        if (frame.empty() && !parser.has("image"))
        {
            std::cout << "KONIEC!";
            break;
        }

        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames(net));

        cv::line(frame, cv::Point(0, frame.rows - (frame.rows / 6)), cv::Point(frame.cols, frame.rows - (frame.rows / 6)), cv::Scalar(0, 0, 0), 2, cv::LINE_8);
        postprocess(frame, outs);

        cv::putText(frame, "cars: " + std::to_string(vehicle[0]), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, "motors: " + std::to_string(vehicle[1]), cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, "buses: " + std::to_string(vehicle[2]), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, "trucks: " + std::to_string(vehicle[3]), cv::Point(20, 115), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image"))
            cv::imwrite(outputFile, detectedFrame);
        else
            video.write(detectedFrame);
        cv::imshow(winName, frame);
    }
    capture.release();
    if (!parser.has("image"))
        video.release();

    return (0);
}

void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> cent;
    for (std::size_t i = 0; i < outs.size(); i++)
    {
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
                cent.push_back(centerY);
                centX.push_back(centerX);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (std::size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        bool ifcount = false;
        int countLineHeightTop = frame.rows - (frame.rows / 8);
        int countLineHeightBottom = frame.rows - (frame.rows / 8);
        bool rep = true;
        if (cent[idx] > countLineHeightTop - 20 && cent[idx] < countLineHeightTop + 20)
        {
            if (!centXH.empty())
            {
                for (std::size_t j = 0; j < indices.size(); j++)
                    if (centX[idx] < centXH[indices[j]] - 50 && centX[idx] > centXH[indices[j]] + 50)
                    {
                        rep = false;
                        break;
                    }
            }
            if (rep)
            {
                int classId = classIds[idx];
                CV_Assert(classId < (int)classes.size());
                switch (classId)
                {
                // car
                case 2:
                    vehicle[0]++;
                    break;
                // motor
                case 3:
                    vehicle[1]++;
                    break;
                // bus
                case 5:
                    vehicle[2]++;
                    break;
                // truck
                case 7:
                    vehicle[3]++;
                    break;
                default:
                    break;
                }
                ifcount = true;
            }
            centXH.clear();
            centX = centXH;
        }
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, ifcount);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, bool color)
{
    if (color)
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    else
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(13, 255, 0), 3);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1. * labelSize.width) + 40, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
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
        {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
