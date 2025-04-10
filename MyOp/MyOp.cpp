// MyOp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#define _CRT_SECURE_NO_WORRIES
#include <iostream>
#include<Windows.h>
#include<string>//
#include<vector>//

#include"OpenCV/cv.h"
#include"openCV\cvaux.h"
#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2/objdetect.hpp>//
#include<opencv2/opencv.hpp>//
#ifdef _DEBUG
#pragma comment(lib,"OpenCV_world331d.lib")
#else
#pragma comment(lib,"OpenCV_world331d.lib")
#endif // DEBUG


using namespace std;
using namespace cv;


//卡通化
void cartoonify(Mat& src, Mat& dst, const char* typeName)
{
    Mat gray;
    cvtColor(src, gray, CV_RGB2GRAY);//变为灰度图
    medianBlur(gray, gray, 5);//去噪

    if (typeName == "Scharr")
    {
        Mat edge1 = Mat(gray.size(), CV_8U);//初始化
        Mat edge2;
        Scharr(gray, edge1, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);//从X方向差分运算
        Scharr(gray, edge2, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);//从Y方向差分运算
        edge1 += edge2;//整体Scharr
        edge1.copyTo(gray);//
    }
    else if (typeName == "Canny")
    {
        Canny(gray, gray, 80, 160, 3);
    }
    else
    {
        Laplacian(gray, gray, CV_8U, 5);
    }
    Mat mask(src.size(), CV_8U);
    threshold(gray, mask, 120, 255, THRESH_BINARY_INV);//图像二值化，将图像上的像素点的灰度值设置为0或255

    //对灰度图中值滤波
    //medianBlur(mask,mask,3);
    //对原始图双边滤波
    Size smallSzie(src.cols / 2, src.rows / 2);//缩小图像，提高效率
    Mat s_src = Mat(smallSzie, src.type());
    resize(src, s_src, smallSzie, 0, 0, INTER_LINEAR);
    Mat tmp = Mat(smallSzie, CV_8UC3);
    int iterator = 7;
    for (int i = 0; i < iterator; i++) {
        int ksize = 9;
        double sigmaColor = 9;
        double sigmaSpace = 7;
        bilateralFilter(s_src, tmp, ksize, sigmaColor, sigmaSpace);
        bilateralFilter(tmp, s_src, ksize, sigmaColor, sigmaSpace);
    }
    Mat b_src;//处理完成，恢复原始尺寸
    resize(s_src, b_src, src.size(), 0, 0, INTER_LINEAR);

    //掩膜叠加
    dst = Mat(src.size(), src.type(), Scalar::all(0)); //初始化
    //dst.setTo(0);
    b_src.copyTo(dst, mask);
    imshow("mask", mask);
}

//浮雕化
void relief(Mat& src, Mat& dst)
{
    Mat relief1(src.size(), CV_8UC3);
    Mat relief2(src.size(), CV_8UC3);
    for (size_t i = 1; i < src.rows - 1; i++)
    {
        for (size_t j = 1; j < src.cols - 1; j++)
        {
            for (size_t k = 0; k < 3; k++)

            {
                int res1 = src.at<Vec3b>(i, j)[k] - src.at<Vec3b>(i - 1, j - 1)[k] + 128;//雕刻
                int res2 = src.at<Vec3b>(i + 1, j + 1)[k] - src.at<Vec3b>(i - 1, j - 1)[k] + 128;//浮雕
                relief1.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(res1);
                relief2.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(res1);
            }
        }
    }
    relief2.copyTo(dst);

    imshow("relief0", relief1);
}

//马赛克
bool Generate_Mosaic(Mat& src,vector<Rect>&faces)
{
    if (faces.empty())return false;

    int step = 10;//步长

    for (int t = 0; t < faces.size(); t++)
    {
        int x = faces[t].tl().x; //人脸矩形框起点x坐标
        int y = faces[t].tl().y;//人脸矩形框起点y坐标
        int width = faces[t].width;//人脸矩形框宽
        int height = faces[t].height;//人脸矩形框高

        //仅对人脸区域进行像素修改。遍历人脸矩形框区域像素，并对其进行修改
        for (int i = y; i < (y + height); i += step)
        {
            for (int j = x; j < (x + width); j += step)
            {
                //将人脸矩形框再细分为若干个小方块，依次对每个方块修改像素（相同方块赋予相同灰度值）
                for (int k = i; k < (step + i); k++)
                {
                    for (int m = j; m < (step + j); m++)
                    {
                        //对矩形区域像素值进行修改，RGB三通道
                        for (int c = 0; c < 3; c++)
                        {
                            src.at<Vec3b>(k, m)[c] = src.at<Vec3b>(i, j)[c];
                        }
                    }
                }
            }
        }
    }
    return true;
}
bool Video_Demo()
{
    string harr_file = "haarcascade_frontalface_default.xml";//人脸检测配置文件
 
    CascadeClassifier detector;
    detector.load(harr_file);//创建人脸检测器
    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        cout << "can not open the camera!" << endl;
    }

    Mat frame;
    while (cap.read(frame))
    {
        flip(frame, frame, 1);

        //人脸检测
        vector<Rect>faces;
        detector.detectMultiScale(frame, faces, 1.1, 3);

        if (Generate_Mosaic(frame, faces))
        {
            imshow("Demo", frame);
        }
        char key = waitKey(10);
        if (key == VK_SPACE)break;
    }

    cap.release();

    return true;
}

int main()
{
    std::cout << "Hello World!\n"; 
    
   
    VideoCapture cap;
    if (!cap.open(0))
        return 0;
    for (;;)
    {
        Mat src;
        cap.read(src);
        int nKey= waitKey(30);
        if (nKey == VK_ESCAPE || src.empty())//VK_SPACE
            break;
        resize(src, src, Size(600, 400), 0, 0, INTER_LINEAR);
        //卡通化
        Mat dst1;
        cartoonify(src, dst1, "Canny");
        //浮雕化
        Mat dst2;
        relief(src, dst2);
        
        imshow("src", src);
        imshow("cartoonImg_Canny", dst1);
        imshow("reliefImg", dst2);
    }
    //马赛克
    Video_Demo();//一个人好的，多个人断点
    
    //VideoCapture cap;//拍照
    //if (!cap.open(0))//打开0号摄像头
    //    return 0;
    //int nImgId = 0;
    //for (;;)
    //{
    //    Mat frame;
    //    cap.read(frame);//或者cap>>frame;
    //    int nKey = waitKey(30);//等待30秒
    //    if (nKey == VK_ESCAPE || frame.empty())//VK_SPACE
    //        break;
    //    else if (nKey == VK_SPACE)
    //    {
    //        char cName[128];
    //    
    //        sprintf_s(cName, "c:\\opencv\\%d.png", nImgId);
    //        imwrite(cName, frame);
    //        nImgId++;
    //    }
    //    Mat gray;
    //    cvtColor(frame, gray, CV_BGR2GRAY);//变为灰度图
    //    Mat edge;
    //    Canny(gray, edge, 120, 60, 3);//边缘提取
    //    std::vector<cv::Point2f>features;//定义特征点
    //    goodFeaturesToTrack(gray, features, 5000, 0.01, 10);//寻找特征点，将灰度图gray里的特征提取出来后放到定义的features里 KLT
    //    for (auto pt : features)//在原图上画出特征点
    //    {
    //        circle(frame, pt, 2, CV_RGB(255, 0, 0));
    //    }
    //    //for (auto pt : features)//在原图上画出特征点
    //    //{
    //    //    circle(edge, pt, 2, Scalar(0, 0, 0), 2);
    //    //}
    //    //Sobel();
    //    imshow("gray", gray);
    //    imshow("pretty", frame);
    //    imshow("edge", edge);
    //}
    ////Sleep(100);
    ////imwrite("C:\\opencv\\a.png", frame);
    system("pause");
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
