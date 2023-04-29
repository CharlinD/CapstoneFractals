#include <stdio.h>
#include <mpi.h>
#include "mpiMB.h"

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <chrono>

using namespace std;

// BMP headers and SaveImage was all ripped from the internet dont ask me what any of this means

#pragma pack(2)
struct BMPHeader {
        unsigned short bfType;       //specifies the file type
        unsigned int bfSize;       //specifies the size in bytes of the bitmap file
        unsigned short bfReserved1;  //reserved: must be 0
        unsigned short bfReserved2;  //reserved: must be 0
        unsigned int bOffbits;     //specifies the offset in bytes from the bitmapfileheader to the bitmap bits
};

#pragma pack(2)
struct BMPInfoHeader {
        unsigned int bisize;           //specifies the # of bytes required by the struct
        int biWidth;          //specifies width in pixels
        int biHeight;         //specifies height in pixels
        unsigned short biPlanes;                 //specifies the # of color planes, must be 1
        unsigned short biBitCount;       //specifies the # of bits per pixel
        unsigned int biCompression;    //specifies the type of compression
        unsigned int biSizeImage;      //size of image in bytes
        int biXPelsPerMeter;  //number of pixels per meter in X axis
        int biYPelsPerMeter;  //number of pixels per meter in Y axis
        unsigned int biCirUsed;        //number of colors used by the bitmap
        unsigned int biCirImportant;   //number of colors that are important?
};


void saveImage(string filePath, int width, int height, Color* image);

// ********************  MAIN  *********************************//

int main(int argc, char **argv){

//starting timer
auto begin = chrono::high_resolution_clock::now();

int myrank;
int nprocs;

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

char processor_name[MPI_MAX_PROCESSOR_NAME];
int name_len;
MPI_Get_processor_name(processor_name, &name_len);

string pname = processor_name;


 int Width = 500;
 int Height = 500;

Color* myImage = new Color[Width * Height];

//cout<<"about to launch cuda function\n";

kernel2(myrank,myImage);

//kernel16(myrank,myImage);

//cout<<"back in MPIMB \n";

string fname;
string srank = to_string(myrank);

//cout<<srank<<endl;

fname = "test"+pname+srank+".bmp";
//cout<<fname<<endl;

 saveImage(fname, Width, Height, myImage);

MPI_Finalize();

//stop timer
auto end = chrono::high_resolution_clock::now();
auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin); 

cout<<"Time: "<<elapsed.count() * 1e-9<<" seconds\n";

delete[] myImage;

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
        header.bOffbits = header.bfSize - (3 * width * height);
        //^^ offset from start of file, how far along in the file do you have to go before you can start loading images
        header.bfReserved1 = 0; //idk why
        header.bfReserved2 = 0; //idk why x2

        infoHeader.biBitCount = 24; //number of bits per pixel (3 bytes, for 3 channels)
        infoHeader.biCirImportant = 0; //for pallates, which we are not using
        infoHeader.biCirUsed = 0; // same as above
        infoHeader.biCompression = 0; //BI_RGB, says we are storing an rgb image
        infoHeader.biHeight = height; //height of image
        infoHeader.biPlanes = 1; //just cuz
        infoHeader.bisize = 40; //determines which info header we are using
        infoHeader.biSizeImage = header.bfSize; //rendundant
        infoHeader.biWidth = width; //width of the image
        infoHeader.biXPelsPerMeter = 0; //not relevant for us, set to 2000 for funsies
        infoHeader.biYPelsPerMeter = 0; //same as above

        fout.write((char*)&header, sizeof(BMPHeader));         //writing our BMP headers to the file
        fout.write((char*)&infoHeader, sizeof(BMPInfoHeader));

        for (int i = 0; i < width * height; i++) {
                unsigned char r, g, b;    //need to convert r,g,b to char vals
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

