// Generating an image of the MandelBrot Set using CUDA
// Charlin Duff
// last modified: 3/30/23
// WORKS!

#include <cuda.h>
#include <cuda_runtime.h>

//#include <device_launch_parameters.h>
//#include <math.h>

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "mpiMB.h"

using namespace std;


// Device version of Color struct
//__device__ 

struct dColor {

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

//MandelBrot & map GPU function definition

	// map is the function that will actually map values within our min and max to pixel
	// values within height and width for our image

__device__ double map(double value, double in_min, double in_max, double out_min, double out_max) {

	return (((value - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min;
}

//Device function for calculating each pixel of the image based on the index of the GPU thread

__device__ dColor MBSet(int ind,int prank){

	//Need to redefine image parameters here so the device knows the image width and height
	//Image Parameters:
	int Width = 500;
	int Height = 500;

	//setting the number of maximum iterations over each c value to test if it will diverge
	int MAX_ITERATIONS = 200;

	//min & max set the complex plane boundaries we want to use when chosing complex
	// plane values to map to pixels on our image

	// for a "zoomed" effect, can set min & max values closer together
	// images should create full MB set when combined between -2 2

	double xminimum;
	double xmaximum;
	double yminimum;
	double ymaximum;


	//Q1,Q2,Q3,Q4
	if(prank == 0){
         xminimum = 0;
         xmaximum = 0.1;
         yminimum = 0;
         ymaximum = 0.1;
         }
        if(prank == 1){
         xminimum = 0;
         xmaximum = 0.1;
         yminimum = -0.1;
         ymaximum = 0;
         }
        if(prank == 2){
         xminimum = -0.1;
         xmaximum = 0;
         yminimum = -0.1;
         ymaximum = 0;
         }
        if(prank == 3){
         xminimum = -0.1;
         xmaximum = 0;
         yminimum = 0;
         ymaximum = 0.1;
         }

	int x = ind % Width;
	int y = ind / Height;

	//initializing a and b
	double a = map(x, 0, Width, xminimum, xmaximum);
	double b = map(y, 0, Height, yminimum, ymaximum);

	//ai, bi = a intital val, b initial val
	double ac = -0.79;
	double bc = 0.15;

	//counter to keep track of how many times a c iterates before diverging
	//(if it diverges)
	int n = 0;

	for (int i = 0; i < MAX_ITERATIONS; i++) {

		double az = a * a - b * b;
		double bz = 2 * a * b;

		//setting next values of a & b for iteration
		a = az + ac;
		b = bz + bc;

		// if a + b > 2, we know z is diverging to infinity, so we will
		// break out of the iterative loop since we already know this value
		// is not in the set.
		if ((a + b) > 2) {
			break;
		}

		n++;
	}

	//Coloring (x,y) pixel value base on # of iterations it took to diverge
	//(if it diverges)

	//Setting the brightness of the pixel based on num iterations before
	//diverging occurs (if at all)
	int bright = map(n, 0, MAX_ITERATIONS, 0, 255);

	//Coloring pixels that are in mandelbrot set black
	// & cleaning up smudges:
	
	if(n == MAX_ITERATIONS){
		bright = 0;
	}

	
	if (bright < 20) {
		bright = 0;
	}

	//defining color set:

	dColor c;
	
	c = dColor(map(sqrt((float)bright), 0, sqrt((float)255), 0, 255), map(bright * bright, 0, 255 * 255, 0, 255), bright);

	//c = dColor(255,255,255);

	//cout<<"red is "<<c.r<<" green is "<<c.g<<" blue is "<<c.b<<endl;
	return c;

}


__global__ void MandelBrot( dColor* d_image, int* d_rank) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	d_image[index] = MBSet(index, *d_rank);
	
}

// for 800x800, N = 640, M = 1000
// for 1000x1000, N,M = 1000

#define N 250
#define M 1000

void kernel2(int prank, Color* colArray) {

	// FUll image to be 5000x5000
	// Image Parameters:
	int Width = 500;
	int Height = 500;

	// host copies of r,g,b

	Color* image;

	int* rank;

	//*************MANDELBROT VARS & MAP FUNCTION INITIALIZATION STUFF**********************

       // device copies of r,g,b

	dColor* d_image;

	int* d_rank;

	int size = Width * Height * 3 *sizeof(float);

	int ranksize = sizeof(int);

	// Allocate space for device copies of a,b,c

	cudaMalloc((void**)&d_image, size);

	cudaMalloc((void**)&d_rank, ranksize);

	// Allocate space for host copies of a,b,c and set up input vals
	
	image = (Color*)malloc(size);
	rank = (int*)malloc(ranksize);
	*rank = prank;

	// Copy inputs to device

	cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_rank, rank, ranksize, cudaMemcpyHostToDevice);

	//cout<<"launching kernel...\n";

	// Launch MB() kernel on GPU
	 MandelBrot <<< N, M >>> (d_image,d_rank);            // <<<numBlocks,numThreadsperblock>>> in this case, 
						       // uses M threads from N different blocks to calculate

	// Copy result back to host

	cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

	cudaMemcpy(rank, d_rank, ranksize, cudaMemcpyDeviceToHost);

	//cout<<"copied results from kernel!\n";

	//cout<<"starting image to colArray copy\n";
        for(int i=0; i<(Width*Height);i++){
         
         colArray[i] = image[i];
         }

	//cout<<"copied image to colArray\n";
        
	// Cleanup
	free(image);
	free(rank);	

	cudaFree(d_image);
	cudaFree(d_rank);	

	//cout<<"finished cuda code\n";
}

