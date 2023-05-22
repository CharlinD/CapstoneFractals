// Generating Julia Set images using CUDA
// Charlin Duff
// last modified: 5/22/23

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// ~~~~~~ Defining host & device copies of color struct ~~~~~~
// There is a way to make them share memory but idk it so they
// each get their very own struct

// Host version of pixel Color Struct
struct Color {

	float r; //red
	float g; //green
	float b; //blue

	Color() { //default constructor
		r = 0;
		g = 0;
		b = 0;
	}

	Color(float red, float green, float blue) { //known value constructor
		r = red;
		g = green;
		b = blue;
	}
};


// Device version of pixel Color struct
__device__ struct dColor {

	float r; //red
	float g; //green
	float b; //blue

	__device__ dColor() { //default constructor
		r = 0;
		g = 0;
		b = 0;
	}

	__device__ dColor(float red, float green, float blue) { //known value constructor
		r = red;
		g = green;
		b = blue;
	}
};

//~~~~~~~ fractalImage & map GPU function definition ~~~~~~~~~

	// map maps value, which is a number between in_min and in_max, to the corresponding number between
	// out_min and out_max

__device__ double map(double value, double in_min, double in_max, double out_min, double out_max) {

	return (((value - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min;
}

// fractalImage takes our array index and uses it to determine what pixel we are at,
// what complex number that pixel corresponds to, whether that pixel is in the
// specified fractal set, and populates that index value of our pixel array with the 
// appropriate color values for that particular pixel/complex number

__device__ dColor fractalImage(int ind) {

	// A Julia set is an IFS fractal given by z = z^2+c. Where z is initalized at the current point 
	// being tested (az+bzi), and c is a set point (ac+bci) on the complex plane best chosen within 
	// circle radius 2 from the origin. A Julia set is entirely contained within this circle so a 
	// choice of c outside these bounds would be frivilous to test. A good window choice will be
	// -2<=x,y<=2
	
	// A Mandelbrot set is an IFS fractal given by z = z^2+c, where z is initialized at 0+0i and
	// c is set to the current point being tested. Like a Julia set, the Mandelbrot Set is entirely 
	// contained within circle radius 2 from the origin so a good window choice will be -2<=x,y<=2.

	//*** INITIALIZING VARIABLES ***//

	//Need to define image parameters again inside function so the device knows the image width and height:
	int Width = 1000;
	int Height = 1000;

	// setting the number of maximum iterations we will run on a single point while testing for divergence:
	int MAX_ITERATIONS = 200;

	// ~~~~ Window ~~~~~~
	
	// x,y min & x,y max set the complex plane boundaries we want to use when chosing complex
	// numbers to map to pixels coordinates on our image

	// for a "zoomed" effect, you can set min & max values closer together around a point of interest
	// normally been stting it to min = -1.5, max = 1.5 for starting windows
	double xminimum = -1.5;
	double xmaximum = 1.5;

	double yminimum = -1.5;
	double ymaximum = 1.5;
	
	// ~~~~~~~~~~~~~~~~~~~

	// ind is the current index we are at in our 1D array of pixels, so we must break it
	// up into its 2D (x,y) parts so we can appropriately perform our "2D" "are you in the 
	// Mandelbrot set" calculation. x denotes a row and y denotes a column.
	
	// (x,y) tells us what pixel we are currently at in our image.
	int x = ind % Width;
	int y = ind / Height;

	// initalizing c:

	// The value of c is static thoughout the iterative calculation, just z is
	// being updated after each iteration.

	// a represents the real part of c, b represents the imaginary part, the map
	// function takes the (x,y) coordinate of the pixel in our image and maps it
	// to the corresponding (x+yi) coordinate within our min/max bounds on the complex
	// plane
	double a = map(x, 0, Width, xminimum, xmaximum);
	double b = map(y, 0, Height, yminimum, ymaximum);

	//ac, bc = c real value, c imaginary value
	// c of interest: -0.79+0.15i, -0.54+0.54i, (for z^4): 0.6+0.55i
	double ac = -0.32193;
	double bc = 0.62762;

	// n is a counter to keep track of how many times z=z^2+c iterates
	// before diverging (if it diverges)
	int n = 0;

	//*** ITERATING Z = Z^2 +C ***//

	for (int i = 0; i < MAX_ITERATIONS; i++) {

		// We do the calculation keeping the real part and imaginary parts seperate

		// az represents the real part of z = z^2, bz represents the imaginary part
		// of z = z^2. c is added after since it is a static number

		//for z^4: az = (a*a*a*a) - (6*a*a*b*b) + (b*b*b*b)
		//         bz = (4*a*a*a*b) - (4*a*b*b*b)

		double az = a*a - b*b;
		double bz = 2*a*b;

		// setting next values of a & b for the next iteration, here we are also adding 
		// the real and imaginary parts of c to the real and imaginary parts of the current z
		a = az + ac;
		b = bz + bc;

		// To save time, we break out of the loop if it is clear z is diverging:

		// if a + b > 2, we know z is beginning to diverge outside the bounds of the 
		// Mandelbrot set, so we will break out of the iterative loop since we already 
		// know this z is not in the set.
		if ((a + b) > 2) {
			break;
		}

		// increase n per iteration:
		n++;
	}

	//*** COLORING THE PIXEL ***//

	// Coloring our (x,y) pixel value base on # of iterations it took for z to diverge
	// (if it diverges)

	// Setting the color brightness of the pixel based on # iterations z completed 
	// before diverging (if it diverges):

	// bright is the number of iterations z completed mapped to a value between 0
	// and 255 for RGB coloring purposed
	int bright = map(n, 0, MAX_ITERATIONS, 0, 255);

	// Coloring pixels that are in set white
	if (n == MAX_ITERATIONS) {

		bright = 255;
	}

	//coloring rogue pixels black to make a prettier picture
	if (bright < 20) {

		bright = 0;
	}

	// Creating an empty pixel struct and coloring it appropriately:

	dColor p;

	// setting our RGB values:
	// Red is just set to bright
	// Green is the sq root of bright mapped to a corresponding value between 0,255
	// Blue is bright squared being mapped to a corresponding value between 0,255

	// dColor(R,G,B):

	p = dColor(bright , map(sqrt((float)bright), 0, sqrt((float)255), 0, 255), map(bright * bright, 0, 255 * 255, 0, 255));

	// The inital, uncolored pixel in our 1D array will be replaced with p
	return p;

}

// Popcorn kernel 
__global__ void Popcorn(dColor* d_image) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	d_image[index] = fractalImage(index);

}

//************STRUCT DEFINITIONS FOR BMP IMAGE HEADERS****************//

// BMP headers and SaveImage was all ripped from the internet dont ask me what any of this means

#pragma pack(2)
struct BMPHeader {
	unsigned short bfType;       //specifies the file type
	unsigned long bfSize;       //specifies the size in bytes of the bitmap file
	unsigned short bfReserved1;  //reserved: must be 0
	unsigned short bfReserved2;  //reserved: must be 0
	unsigned long bOffbits;     //specifies the offset in bytes from the bitmapfileheader to the bitmap bits
};

#pragma pack(2)
struct BMPInfoHeader {
	unsigned long bisize;           //specifies the # of bytes required by the struct
	long biWidth;          //specifies width in pixels
	long biHeight;         //specifies height in pixels
	unsigned short biPlanes;		 //specifies the # of color planes, must be 1
	unsigned short biBitCount;       //specifies the # of bits per pixel
	unsigned long biCompression;    //specifies the type of compression
	unsigned long biSizeImage;      //size of image in bytes
	long biXPelsPerMeter;  //number of pixels per meter in X axis
	long biYPelsPerMeter;  //number of pixels per meter in Y axis
	unsigned long biCirUsed;        //number of colors used by the bitmap
	unsigned long biCirImportant;   //number of colors that are important?
};


//************ END OF STRUCT DEFINITIONS FOR BMP IMAGE HEADERS ****************//

// function header for SaveImage
// see under main for function defn
void saveImage(string filePath, int width, int height, Color* image);

// Define N,M our kernel size, M threads on N blocks
// max threads per block = 1024 (max M)

// to decide how many bolcks/threads:
// N = (Width * Height) / 1000 (blocks)
// M = 1000 (threads) (i use 1000 for simplicity, these #s must be ints and
//                     dividing by 1024 doesnt always result in a whole num.)

// for 800x800 img, N = 640, M = 1000
// for 1000x1000 img, N,M = 1000

#define N 1000
#define M 1000

int main(void) {

	//Image Parameters:
	int Width = 1000;
	int Height = 1000;
	char type = 'j';    // lets program know which fractal you are generating
	                    // set to 'j' for Julia or 'm' for Mandelbrot

	// host copy of pixel array
	Color* image;

	// device copy of pixel array
	dColor* d_image;
	
	// device copy of height & width
	
	// device copy of type

	// calculating necessary size for arrays based on image size and pixel size
	int size = Width * Height * 3 * sizeof(float);

	// Allocate space for device copy of array
	cudaMalloc((void**)&d_image, size);

	// Allocate space for host copy of array
	image = (Color*)malloc(size);

	// Copy array to device
	cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);

	// Pops our kernel (Launches kernel on GPU)
	Popcorn <<< N, M >>> (d_image);            // <<<numBlocks,numThreadsperblock>>> defines kernel size, 
						     // in this case, uses M threads from N different blocks
						     // d_image is our variable we want to pass to the device

        // Copy GPU array back to host after kernel runs
	cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

	// Save pixel array to image file on computer
	saveImage("ExampleFractal.bmp", Width, Height, image);

	// VIOLENTLY DESTROY THE ARRAYS
	free(image);
	cudaFree(d_image);

	return 0;

}

//******************************** SAVE IMAGE FUNCTION DEFN ********************************************//
// saves our pixel array to an image file

void saveImage(string filePath, int width, int height, Color* image) {
	ofstream fout;
	fout.open(filePath, ios::binary); //opening our file

	//Setting the BMP stuff for our file

	BMPHeader header;
	BMPInfoHeader infoHeader;

	header.bfType = 0x4D42; //BM, translates to 'BM' in ASCII code
	header.bfSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + 3 * width * height;
	//^^ size of BMPheader in bytes + size of BMP info header in bytes + 3 bytes*width*height for our r,g,b channels in each pixel
	header.bOffbits = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
	//^^ offset from start of file, how far along in the file do you have to go before you can start loading images
	header.bfReserved1 = 0; //idk why
	header.bfReserved2 = 0; //idk why x2

	infoHeader.biBitCount = 24; //number of bits per pixel (3 bytes, for 3 channels)
	infoHeader.biCirImportant = 0; //for pallates, which we are not using
	infoHeader.biCirUsed = 0; // same as above
	infoHeader.biCompression = 0; //BI_RGB, says we are storing an rgb image
	infoHeader.biHeight = height; //height of image
	infoHeader.biPlanes = 1; //just cuz
	infoHeader.bisize = sizeof(BMPInfoHeader); //determines which info header we are using
	infoHeader.biSizeImage = header.bfSize; //rendundant
	infoHeader.biWidth = width; //width of the image
	infoHeader.biXPelsPerMeter = 0; //not relevant for us
	infoHeader.biYPelsPerMeter = 0; //same as above

	fout.write((char*)&header, sizeof(BMPHeader));         //writing our BMP headers to the file
	fout.write((char*)&infoHeader, sizeof(BMPInfoHeader));

	for (int i = 0; i < width * height; i++) {
		unsigned char r, g, b;    //need to convert r,g,b to char vals for image writing
		Color c = image[i];
		r = (int)floor(c.r);
		g = (int)floor(c.g);
		b = (int)floor(c.b);

		fout.write((char*)&b, sizeof(unsigned char));
		fout.write((char*)&g, sizeof(unsigned char));
		fout.write((char*)&r, sizeof(unsigned char));
	}
	fout.close();
}
//******************************** END OF SAVE IMAGE FUNCTION ********************************************//
