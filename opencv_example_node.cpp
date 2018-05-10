#include "ros/ros.h"
#include <iostream>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>

#define MAX_MANA 5
using namespace std;
using namespace cv;
cv::Mat detectedEdges;
int marker_size=0;
bool GO=0;
ros::Publisher vel_pub;
int mana=0;
int prev_marker_size=0;
vector<Vec3f> prev_circles;
bool first=1;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
  int lowThreshold=10;
  int ratio = 3;
  int kernel_size = 3;
  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
  cv::Mat result;
  cv::inRange(hsv_image, cv::Scalar(90, 70, 70), cv::Scalar(130, 255, 255), result); // Finding blue color on image
  GaussianBlur( result, result, Size(9, 9), 2, 2 );  // Reduce the noise so we avoid false circle detection
  vector<Vec3f> circles;
//  cv::Mat src_gray;
//  cvtColor( image, src_gray, CV_BGR2GRAY );

  marker_size=0;
  HoughCircles( result, circles, CV_HOUGH_GRADIENT, 1, 1, 200, 50, 0, 0 );  // Apply the Hough Transform to find the circles
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
  	  marker_size = radius;
//      circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );  // circle outline
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


  cv::Mat fin;
  fin= cv::Mat::zeros(result.size(), result.type());
  for( size_t i = 0; i < circles.size(); i++)
  {

      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle( fin, center, radius, 255, -1, 8, 0 );

  }

  if(marker_size < 35 && marker_size>0)
  {
	   GO = true;
   }
   else
   {
	    GO = false;
  }

if(GO)
{
	  geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
  	vel->linear.x=-0.5;
  	vel->angular.z=0.0;
//  	vel_pub.publish(vel);                                                        //uncomment for ride
    if(first==1)
    {
      cout << " Bzi";
      first=0;
    }
    else
    {
      cout<<"uu";
    }
}
else
{
  	geometry_msgs::TwistPtr vel(new geometry_msgs::Twist());
  	vel->linear.x=-0.0;
  	vel->angular.z=0.0;
//  	vel_pub.publish(vel);                                                        //uncomment for ride
    if(first==0)
    {
      first=1;
      cout<<endl;
    }
}

//  imshow("Image",image);
  imshow("Image edited", result);
  imshow("Image circle found",fin);
  waitKey(1);

}

int main(int argc, char **argv)
{
  cout<<"SIEMANECZKOoooo"<<endl;

  //initialize node
  ros::init(argc, argv, "cv_example");

  // node handler
  ros::NodeHandle n;

  // subsribe topic
  ros::Subscriber sub = n.subscribe("/cv_camera/image_raw", 1, imageCallback);
//  vel_pub = n.advertise<geometry_msgs::Twist>("cmd_joy", 1);                      //uncomment for ride

  // publish
  image_transport::ImageTransport it(n);
  image_transport::Publisher pub = it.advertise("camera/imageEdges", 1);

  sensor_msgs::ImagePtr msg;

  ros::Rate loop_rate(5);
  while (n.ok())
  {
    if(!detectedEdges.empty())    // Check if grabbed frame is actually full with some content
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
