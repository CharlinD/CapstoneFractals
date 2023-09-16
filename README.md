WORK IN PROGRESS, A BIT UNORGANIZED RIGHT NOW

# CapstoneFractals
CUDA/C++ code for generating Julia Sets and the Mandelbrot Set

JuliaSerial.cpp generates Julia Sets in serial. 
julia.cu generates Julia Sets in GPU parallel.
When used in combination, kernel2.cu, mpiMB.cpp, mpiMB.h, and Makefile can be used to make an executable that will create Julia Sets in CPU & GPU parallel on the supercomputer. machines is the file used to tell the supercomputer which/how many nodes and CPU cores to run the executable on.
