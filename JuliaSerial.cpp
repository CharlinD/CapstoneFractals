#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;

#pragma pack(2)
struct BMPHeader {
	unsigned short bfType;       //specifies the file type
	unsigned long bfSize;        //specifies the size in bytes of the bitmap file
	unsigned short bfReserved1;  //reserved, must be 0
	unsigned short bfReserved2;  //reserved, must be 0
	unsigned long bOffBits;      //specifies the offset in bytes
};

#pragma pack(2)
struct BMPInfoHeader {
	unsigned long biSize;           //specifies # bytes required by the struct
	long biWidth;                   //specifies width in pixels
	long biHeight;                  //specifies height in pixels
	unsigned short biPlanes;        //specifies # color planes (must be 1)
	unsigned short biBitCount;      //specifies # of bit per pixel
	unsigned long biCompression;    //specifies type of compression
	unsigned long biSizeImage;      //size of image in bytes
	long biXPelsPerMeter;           //# pixels per meter x axis
	long biYPelsPerMeter;           //# pixels per meter y axis
	unsigned long biCirUsed;        //# of colors used by bitmap
	unsigned long biCirImportant;   //# important colors
};


struct Pixel {

	float r;
	float g;
	float b;

	Pixel() {

		r = 0;
		g = 0;
		b = 0;
	}

	Pixel(float red, float green, float blue) {

		r = red;
		g = green;
		b = blue;
	}
};

double map(double value, double in_min, double in_max, double out_min, double out_max) {

	return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void saveImage(string filePath);  //Defined under main

int width = 5000;
int height = 5000;

Pixel* image = new Pixel[width * height];

int main(int args, char** argv) {

	//Start timer
	auto begin = chrono::high_resolution_clock::now();

	//int set = 0;

	double xmin = -1.2;
	double xmax = 1.2;
	double ymin = -1.2;
	double ymax = 1.2;

	double ac = 0.38015;
	double bc = -0.15458;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

			int PixelID = y * width + x;

			double a = map(x, 0, width, xmin, xmax);
			double b = map(y, 0, height, ymin, ymax);

			int MaxIts = 200;
			int n = 0;

			for (int i = 0; i < MaxIts; i++) {

				double az = (a * a) - (b * b);
				double bz = 2 * a * b;

				a = az + ac;
				b = bz + bc;

				if ((a + b) > 2) {
					break;
				}

				n++;
			}

			int bright = floor(map(n, 0, MaxIts, 0, 255));

			if (n == MaxIts) {
				bright = 0;
			}
			if (bright < 20) {

				bright = 0;
			}

			Pixel p;

			//map(sqrt(bright), 0, sqrt(255), 0, 255)
			//map(bright * bright, 0, 255 * 255, 0, 255)

			p = Pixel(map(sqrt(bright), 0, sqrt(255), 0, 255), map(bright * bright, 0, 255 * 255, 0, 255), bright);

			//brightness[PixelID] = bright;
			image[PixelID] = p;

		}
	}

	//saveBright("HTJ1Bright.txt");
	saveImage("HTJulia3.bmp");

	// Stop timer
	auto end = chrono::high_resolution_clock::now();
	auto elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	cout << "Time: " << elapsed.count() * 1e-9 << " seconds \n";

	return 0;
}

void saveImage(string filePath) {

	ofstream fout;
	fout.open(filePath, ios::binary);

	BMPHeader header;
	BMPInfoHeader infoHeader;

	header.bfType = 0x4D42; //BM in ASCII
	header.bfSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + 3 * width * height;
	header.bOffBits = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
	header.bfReserved1 = 0;
	header.bfReserved2 = 0;

	infoHeader.biBitCount = 24; //3 bytes per pixel
	infoHeader.biCirImportant = 0;
	infoHeader.biCirUsed = 0;
	infoHeader.biCompression = 0;
	infoHeader.biHeight = height;
	infoHeader.biPlanes = 1;
	infoHeader.biSize = sizeof(BMPInfoHeader);
	infoHeader.biSizeImage = header.bfSize;
	infoHeader.biWidth = width;
	infoHeader.biXPelsPerMeter = 0;
	infoHeader.biYPelsPerMeter = 0;

	fout.write((char*)&header, sizeof(BMPHeader));
	fout.write((char*)&infoHeader, sizeof(BMPInfoHeader));

	for (int i = 0; i < (width * height); i++) {

		unsigned char r, g, b;
		Pixel c = image[i];
		r = floor(c.r);
		g = floor(c.g);
		b = floor(c.b);

		fout.write((char*)&b, sizeof(unsigned char));
		fout.write((char*)&g, sizeof(unsigned char));
		fout.write((char*)&r, sizeof(unsigned char));
	}

	fout.close();
}