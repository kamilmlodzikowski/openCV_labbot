#include "ros/ros.h"
#include <iostream>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define MAX_MANA 5
using namespace std;
using namespace cv;
cv::Mat detectedEdges;
int marker_size = 0;
bool GO = 0;
bool GO_left = 0;
bool GO_right = 0;
ros::Publisher vel_pub;
int mana = 0;
int prev_marker_size = 0;
vector<Vec3f> prev_circles;
bool first = 1;
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	//-- Show what you got
	imshow("twarze", frame);
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	int lowThreshold = 10;
	int ratio = 3;
	int kernel_size = 3;
	cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
	detectAndDisplay(image);
	//cv::Mat hsv_image;
	//cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
	//cv::Mat result;
	//cv::inRange(hsv_image, cv::Scalar(100, 100, 100), cv::Scalar(120, 255, 255), result); // Finding blue color on image
	//GaussianBlur( result, result, Size(9, 9), 2, 2 );  // Reduce the noise so we avoid false circle detection
	vector<Vec3f> circles;
	cv::Mat src_gray;
	cvtColor(image, src_gray, CV_BGR2GRAY);
	vector<Rect> faces;
	face_cascade.detectMultiScale(src_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//marker_size=0;
	//HoughCircles( result, circles, CV_HOUGH_GRADIENT, 1, 1, 200, 50, 0, 0 );  // Apply the Hough Transform to find the circles

  /*  for(size_t i = 0; i < circles.size(); i++)
	{
	  Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	  int radius = cvRound(circles[i][2]);
	  marker_size = radius;
	  //circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );  // circle outline
	}

	// Prawdopodobienstwo wykrycia kola
	if(circles.size()>0 && mana != MAX_MANA)
	{
	  mana++;
	  prev_marker_size=marker_size;
	  prev_circles=circles;
	}
	else if(circles.size()>0)
	{
	  prev_marker_size=marker_size;
	  prev_circles=circles;
	}
	else if(circles.size()==0 && mana>0)
	{
	  marker_size=prev_marker_size;
	  circles=prev_circles;
	  mana--;
	}
  */
  /*  cv::Mat fin;
	fin= cv::Mat::zeros(result.size(), result.type());
	for( size_t i = 0; i < circles.size(); i++)
	{
	  Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	  int radius = cvRound(circles[i][2]);
	  circle( fin, center, radius, 255, -1, 8, 0 );
	}
  */
	if (marker_size < 50 && marker_size>0)
	{
		/*
	  if(faces[1].x<image.cols/3)
		{
		  GO=false;
		  GO_right=false;
		  GO_left=true;

		}
		else if (faces[1].x>image.cols/3*2)
		{
		  GO_right = true;
		  GO=false;
		  GO_left=false;
		}
	  */
		if (1)
		{
			GO_left = false;
			GO_right = false;
			GO = true;
		}
	}
	else
	{
		GO = false;
		GO_left = false;
		GO_right = false;
	}

	if (GO)
	{
		geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
		vel->linear.x = -0.5;
		vel->angular.z = 0.0;
		vel_pub.publish(vel);                      //uncomment for ride
		if (first == 1)
		{
			cout << circles[0][1] << endl;
			cout << " Bzi";
			first = 0;
		}
		else
		{
			cout << "uu";
		}
	}
	else if (GO_left)
	{
		geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
		vel->linear.x = -0.0;
		vel->angular.z = -0.5;
		vel_pub.publish(vel);                      //uncomment for ride
		cout << "LEWO   ";
	}
	else if (GO_right)
	{
		geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
		vel->linear.x = -0.0;
		vel->angular.z = 0.5;
		vel_pub.publish(vel);                      //uncomment for ride
		if (first == 1)
			cout << "PRAWO   ";
	}
	else
	{
		geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
		vel->linear.x = -0.0;
		vel->angular.z = 0.0;
		vel_pub.publish(vel);                       //uncomment for ride
		if (first == 0)
		{
			first = 1;
			cout << endl;
		}
	}

	imshow("Image", image);

	//imshow("Image edited", result);
	//imshow("Image circle found",fin);
	waitKey(1);
}

int main(int argc, char **argv)
{
	cout << "SIEMANECZKOoooo" << endl;

	//initialize node
	ros::init(argc, argv, "cv_example");

	//node handler
	ros::NodeHandle n;
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)2 ERROR\n"); return -1; };
	//subsribe topic
	ros::Subscriber sub = n.subscribe("/camera/rgb/image_raw", 1, imageCallback);
	vel_pub = n.advertise<geometry_msgs::Twist>("cmd_joy", 1);                      //uncomment for ride

	// publish
	image_transport::ImageTransport it(n);
	image_transport::Publisher pub = it.advertise("camera/imageEdges", 1);

	sensor_msgs::ImagePtr msg;

	ros::Rate loop_rate(5);
	while (n.ok())
	{
		if (!detectedEdges.empty())    // Check if grabbed frame is actually full with some content
		{
			msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", detectedEdges).toImageMsg();
			pub.publish(msg);
			cv::waitKey(1);
		}
		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}