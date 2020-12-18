#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
	const std::string model_file = "E:/caffe_model/bvlc_googlenet.caffemodel";
	const std::string config_file = "E:/caffe_model/bvlc_googlenet.prototxt";
	const std::string classes_file = "E:/caffe_model/classification_classes_ILSVRC2012.txt";
	const std::string input_file = "E:/caffe_model/space_shuttle.jpg";

	/* 与训练模型有关 */
	float scale = 1.0;
	bool swap_rb = true;
	int train_width = 224;
	int train_height = 224;
	Scalar mean = Scalar(104.0, 117.0, 123.0);

	/* 读入类别 */
	std::ifstream ifs(classes_file.c_str());
	if (!ifs.is_open()) {
		std::cerr << "File " + classes_file + " not found";
		return -1;
	}
	std::string line;
	while (std::getline(ifs, line)){
		classes.push_back(line);
	}
	
	/* 初始化网络 */
	Net net = readNet(model_file, config_file);
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);

	Mat frame, blob;
	// 读取原始数据
	frame = imread(input_file);
	if (frame.empty()) {
		std::cerr << "Invalid input file" << std::endl;
		return -1;
	}
	// 转换数据
	blobFromImage(frame, blob, scale, Size(train_width, train_height), mean, swap_rb, false);
	// 设置网络
	net.setInput(blob);
	// 向前预测
	Mat prob = net.forward();
	Point classIdPoint;
	double confidence;
	// 选取可信度最高的类别
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	std::string label = format("%s : %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
		classes[classId].c_str()),confidence);
	std::cout << label << std::endl;

	std::getchar();
	return 0;
}