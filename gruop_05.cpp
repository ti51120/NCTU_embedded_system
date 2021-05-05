#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>



#define device_path "/dev/fb0"
#define classifier_path "/run/media/mmcblk1p1/haarcascade_frontalface_alt.xml"
#define model_path "/run/media/mmcblk1p1/individual_face.xml"
#define scale 3.5



using namespace std;
using namespace cv;
using namespace cv::face;

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path ); 
void Detect(VideoCapture camera, Mat &frame, CascadeClassifier faceCascade);
void Train(VideoCapture camera, Mat &frame,  CascadeClassifier faceCascade, int persons, map<int, pair<string, double> > &label_and_names);
void Recognize(VideoCapture camera, Mat &frame, CascadeClassifier faceCascade, map<int, pair<string, double> > &label_and_names);
void Output_to_framebuffer(Mat &converted_frame, Size frame_size); 					  // output frame to screen
vector<Rect> detect_faces(Mat &frame, Mat &gray_frame, CascadeClassifier classifier); //get detected faces and store into vector
char console_input();  // get user input


Mat frame;
char mode, key;

// open the framebuffer device
framebuffer_info fb_info = get_framebuffer_info(device_path);
ofstream ofs(device_path);


int main(int argc, const char *argv[])
{
	// load face detection classifier
	CascadeClassifier faceCascade;
    faceCascade.load(classifier_path);

	VideoCapture camera(2);

	// set propety of the frame
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 750);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 450);	
	camera.set(CV_CAP_PROP_FPS, 30);

	map<int, pair<string, double> > label_and_names;

	while(mode != 'q' && mode != 'Q')
	{
		cout << "\'D\' for detection only\n\'T\' for training faces\n\'R\' for recognize\n\'Q\' for quit\nMode: ";
		cin >> mode;
		switch(mode)
		{
			case 'D':
			case 'd':
				Detect(camera, frame, faceCascade);
				break;
			case 'T':
			case 't':
			{
				int persons;
				cout << "Input persons you wanna train: ";
				cin >> persons;
				Train(camera, frame, faceCascade, persons, label_and_names);
			}
				break;
			case 'R':
			case 'r':
				Recognize(camera, frame, faceCascade, label_and_names);
				break;
			default:
				break;
		}
	}
	camera.release();
    faceCascade.~CascadeClassifier();
	
	return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.
    
	// open deive with linux system call "open()"    
    int fd=-1;
    fd=open(framebuffer_device_path, O_RDWR);

	// put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    if(fd >= 0){

		// get attributes of the framebuffer device thorugh linux system call "ioctl()"
		if(!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info)){
			fb_info.xres_virtual = screen_info.xres_virtual;
			fb_info.bits_per_pixel = screen_info.bits_per_pixel;
 		}
	}
    return fb_info;
}

void Output_to_framebuffer(Mat &converted_frame, Size frame_size)
{
    int framebuffer_width = fb_info.xres_virtual;
    int framebuffer_depth = fb_info.bits_per_pixel;

	for ( int y = 0; y < frame_size.height; y++ )
	{
		ofs.seekp(y*framebuffer_width*2);
		ofs.write(reinterpret_cast<char*>(converted_frame.ptr(y)), frame_size.width*2);
	}

}

char console_input()
{
	char buf[1];
	int flag=fcntl(0,F_GETFL);
	fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
	int numRead = read(0, buf, 1);
	fcntl(0, F_SETFL, flag);	
	return (numRead > 0) ? buf[0] : 0;
}

vector<Rect> detect_faces(Mat &frame, CascadeClassifier classifier)
{
    vector<Rect> faces;
	Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
	resize(gray_frame, gray_frame, Size(gray_frame.size().width/scale, gray_frame.size().height/scale));
	classifier.detectMultiScale(gray_frame, faces, 1.1, 3, 0, Size(30, 30));
    return faces;
}

void Detect(VideoCapture camera, Mat &frame, CascadeClassifier faceCascade)
{	
	cout << "Face detection" << endl;

	Mat converted_frame;
	vector<Rect> faces;

	while(true)
	{
		// get video frame from stream
		camera >> frame;
		if(!camera.read(frame)) cout << "Unable to retrieve frame from camera!" << endl;

		// get size of the video frame
		Size frame_size = frame.size();

		// transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
		cvtColor(frame, converted_frame, COLOR_BGR2BGR565);

		//convert to grayscale image to minimize computation cost and record faces detected
		faces = detect_faces(frame, faceCascade);

		//draw rectangle and put text
		for(int i= 0; i < faces.size(); i++)
		{
			rectangle(converted_frame, cvPoint(cvRound(faces[i].x * scale), cvRound(faces[i].y * scale)), 
				            cvPoint(cvRound((faces[i].x+faces[i].width-1) * scale), cvRound((faces[i].y+faces[i].height-1) * scale)), Scalar(0, 255, 0), 1);
		}

		// output the video frame to framebufer row by row
		Output_to_framebuffer(converted_frame, frame_size);
		faces.clear();
			
		key = console_input();
		if(key == 'q' || key == 'Q') break;
	}
	cout << "Done detecting" << endl;
}

void Train(VideoCapture camera, Mat &frame,  CascadeClassifier faceCascade, int persons, map<int, pair<string, double> > &label_and_names)
{
	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
	Mat face_region_of_interest;
	string name;
	int epoch;
	double precision;

	label_and_names.clear();

	while(persons--)
	{
		cout << "Training face" << endl;
		cout << "Enter name: ";
		cin >> name;
		cout << "Epoch: ";
		cin >> epoch;

		vector<Mat> images;
		vector<int> labels;
		vector<Rect> faces;

		for(size_t i = 0; i < epoch + 1; ++i)
		{

			camera >> frame;
			if(!camera.read(frame)) cout << "Unable to retrieve frame from camera!" << endl;	

			Size frame_size=frame.size();

			faces = detect_faces(frame, faceCascade);
 			
    		cvtColor(frame, frame, COLOR_BGR2GRAY);			

			//get face characteristics
			if(i != epoch && faces.size() != 0)
			{
				cout << "Face pos: (" << faces[0].x << ", " << faces[0].y << ')' << endl;
				face_region_of_interest = frame(faces[0]); 					
				images.push_back(face_region_of_interest);
				labels.push_back(persons);  				
			}

			if(i == epoch)
			{
				//train the model
				model->update(images, labels);
				model->save(model_path);
				
				//get proper threshold to distinguish face
				face_region_of_interest = frame(faces[0]); 							
				model->predict(face_region_of_interest,persons, precision);
				label_and_names[persons] = make_pair(name, precision);

				cout << precision << endl;
			}	
			
			faces.clear();	
		}
		cout << (persons > 0) ? "NEXT PERSON\n" : "Done training\n";
	}

}

void Recognize(VideoCapture camera, Mat &frame, CascadeClassifier faceCascade, map<int, pair<string, double> > &label_and_names)
{
	Ptr<LBPHFaceRecognizer> model = Algorithm::load<LBPHFaceRecognizer>(model_path); 

	double precision;
	int label, range;    
	string text;	
	Mat converted_frame;
	Mat face_region_of_interest;
	vector<Rect> faces;

	clock_t start, end;
  	double cpu_time_used;

	cout << "Identifying face" << endl;
	cout << "range: ";
	cin >> range;

	while(true)
	{	
		camera >> frame;
		if(!camera.read(frame)) cout << "Unable to retrieve frame from camera!" << endl;	

		Size frame_size=frame.size();

		cvtColor(frame, converted_frame, COLOR_BGR2BGR565);		

		faces = detect_faces(frame, faceCascade);
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		
		start = clock();
	
		for(size_t i=0; i<faces.size(); i++)
		{
			rectangle(converted_frame, cvPoint(cvRound(faces[i].x * scale), cvRound(faces[i].y * scale)), 
				            cvPoint(cvRound((faces[i].x+faces[i].width-1) * scale), cvRound((faces[i].y+faces[i].height-1) * scale)), Scalar(0, 255, 0), 1);

			// give recognization results 
			face_region_of_interest = frame(faces[i]);				            
			label = model->predict(face_region_of_interest);
			model->predict(face_region_of_interest, label, precision);

			text = (precision > label_and_names[label].second + range) ? "unknown" : label_and_names[label].first;			
			putText(converted_frame, text, cvPoint(cvRound(faces[i].x * scale), cvRound(faces[i].y * scale)), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 0), 2);
		
			cout << "Threshold: " << precision << endl;
			cout << "===============================" << endl;
			cout << "(" << label << ", " << text << ')' << endl;
		}
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

		printf("Time = %f\n", cpu_time_used);


		Output_to_framebuffer(converted_frame, frame_size);
		faces.clear();

		key = console_input();
		if(key == 'q' || key == 'Q') break;
	}
	
	cout << "Done recognizing" << endl;
}



