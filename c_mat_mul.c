#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

void call_kernel(float *x1,float *x2,int n1,int n2,int n3,float *x3,char * cl_name) {

    FILE* programHandle;
    size_t programSize, KernelSourceSize;
    char *programBuffer, *KernelSource;

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input1;                       // device memory used for the input array
    cl_mem input2;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array

	int err;
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);

	//----------------------------------------------------------------------------
    // get size of kernel source
    programHandle = fopen(cl_name, "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context,1,(const char**) &programBuffer,&programSize, NULL);
    free(programBuffer);

    // read kernel source back in from program to check
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &KernelSourceSize);
    KernelSource = (char*) malloc(KernelSourceSize);
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, KernelSourceSize, KernelSource, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "mmul", &err);
	//----------------------------------------------------------------------------

    input1 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float)*n1*n2, NULL, NULL);
    input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float)*n2*n3, NULL, NULL);
    output = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float)*n1*n3, NULL, NULL);

    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float)*n1*n2, x1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float)*n2*n3, x2, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, 2, sizeof(int), &n1);
    clSetKernelArg(kernel, 3, sizeof(int), &n2);
    clSetKernelArg(kernel, 4, sizeof(int), &n3);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);

	//const size_t globalThreads[2] = {n1, n3};
	//const size_t localThreads[2] = {16,16};
    //err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalThreads, localThreads, 0, NULL, NULL);
	size_t global;
	size_t local;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = n1*n2;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    clFinish(commands);
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float)*n1*n3, x3, 0, NULL, NULL );

    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(output);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    printf("nKernel source:\n\n %s \n", KernelSource);
    free(KernelSource);
}

void mat_mul(int n1,int n2,int n3, float *A, float *B, float *C) {
	int i, j, k;
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n3; j++) {
			for (k = 0; k < n2; k++) {
				// C(i, j) = sum(over k) A(i,k) * B(k,j)
				C[i*n3 + j] += A[i*n2+k]*B[k*n3+j];
			}
		}
	}
}

void mat_add(int n1,int n2,int n3, float *A, float *B, float *C) {
	int i, j, k;
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n3; j++) {
			C[i*n3 + j] = A[i*n2+k]+B[k*n3+j];
		}
	}
}



int main(int argc, const char *argv[])
{
	// 4.2X2.3 = 4.3
	int n1 = 100,n2 = 100,n3 = 100;
	float *A = (float *) malloc(n1*n2*sizeof(float));
	float *B = (float *) malloc(n2*n3*sizeof(float));
	float *C = (float *) malloc(n1*n3*sizeof(float));
	float *C_c = (float *) malloc(n1*n3*sizeof(float));

	int i;
	for (i = 0; i < n1*n2; i++) {
		A[i] = 1.1;
	}
	for (i = 0; i < n2*n3; i++) {
		B[i] = 1.2;
	}

	//mat_mul(n1,n2,n3,A,B,C_c);
	mat_add(n1,n2,n3,A,B,C_c);
	call_kernel(A,B,n1,n2,n3,C,"./matrix_mul1.cl");

	for (i = 0;i<n1*n3;i++) {
		printf("%f-----%f \n",C_c[i], C[i]);
	}
	return 0;
}
