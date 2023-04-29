MPIMB: mpiMB.cpp kernel2.cu
	mpicc -c mpiMB.cpp
	nvcc -c kernel2.cu
	/usr/bin/mpicxx -o MPIMB mpiMB.o kernel2.o -lcudart -L/usr/local/cuda-10.2/lib64 -I.
