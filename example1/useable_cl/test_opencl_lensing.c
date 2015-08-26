#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include "all_cv_test.h"

void call_kernel(float *xi1,float *xi2,int count,float *lpar,float *alpha1,float *alpha2,char * cl_name) {

    FILE* programHandle;
    size_t programSize, KernelSourceSize;
    char *programBuffer, *KernelSource;

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input1;                       // device memory used for the input array
    cl_mem input2;                       // device memory used for the input array
    cl_mem lpar_d;                       // device memory used for the input array
    cl_mem output1;                      // device memory used for the output array
    cl_mem output2;                      // device memory used for the output array

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
    kernel = clCreateKernel(program, "square", &err);
	//----------------------------------------------------------------------------

    input1 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    lpar_d = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * 6, NULL, NULL);
    output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * count, xi1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * count, xi2, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, lpar_d, CL_TRUE, 0, sizeof(float) * 6, lpar, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &lpar_d);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output1);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &output2);
    clSetKernelArg(kernel, 5, sizeof(int), &count);

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(commands);
    err = clEnqueueReadBuffer( commands, output1, CL_TRUE, 0, sizeof(float) * count, alpha1, 0, NULL, NULL );
    err = clEnqueueReadBuffer( commands, output2, CL_TRUE, 0, sizeof(float) * count, alpha2, 0, NULL, NULL );

    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(lpar_d);
    clReleaseMemObject(output1);
    clReleaseMemObject(output2);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    //printf("nKernel source:\n\n %s \n", KernelSource);
    free(KernelSource);
}

int main(int argc, const char *argv[]) {

    float xlc0 = 0.0;
    float ylc0 = 0.0;
    float ql0 = 0.7;
    float rc0 = 0.1;
    float re0 = 1.0;
    float phi0 = 0.0;
    float lpar[] = {ylc0,xlc0,ql0,rc0,re0,phi0};

	int count = 1024*1024;
    float *xi1 = (float *)malloc(sizeof(float)*count);
    float *xi2 = (float *)malloc(sizeof(float)*count);
    float *alpha1 = (float *)malloc(sizeof(float)*count);
    float *alpha2 = (float *)malloc(sizeof(float)*count);
    int correct;

    int i = 0;
    for(i = 0; i < count; i++) {
		xi1[i] = rand() / (float)RAND_MAX;
		xi2[i] = rand() / (float)RAND_MAX;
	}


	call_kernel(xi1,xi2,count,lpar,alpha1,alpha2,"./play_with.cl");

    float *alpha1_c = (float *)malloc(sizeof(float)*count);
    float *alpha2_c = (float *)malloc(sizeof(float)*count);
    correct = 0;
    for(i = 0; i < count; i++) {
		lq_nie(xi1[i],xi2[i],lpar,&alpha1_c[i],&alpha2_c[i]);
		printf("%f-----%f||%f-----%f\n",alpha1[i],alpha1_c[i],alpha2[i],alpha2_c[i]);
    }

	free(xi1);
	free(xi2);
	free(alpha1);
	free(alpha2);
	free(alpha1_c);
	free(alpha2_c);
    return 0;
}
