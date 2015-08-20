#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>


void call_kernel(float *data,unsigned int count,char * cl_name,float *results) {

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

    cl_mem input;                       // device memory used for the input array
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
    kernel = clCreateKernel(program, "square", &err);
	//----------------------------------------------------------------------------

    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(commands);
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    printf("nKernel source:\n\n %s \n", KernelSource);
    free(KernelSource);
}

int main(int argc, const char *argv[]) {

	int count = 1024*1024;
    float *data = (float *)malloc(sizeof(float)*count);              // original data set given to device
    float *results = (float *)malloc(sizeof(float)*count);              // original data set given to device
    unsigned int correct;               // number of correct results returned

    int i = 0;
    for(i = 0; i < count; i++) data[i] = rand() / (float)RAND_MAX;

	call_kernel(data,count,"./simple_add.cl",results);

    correct = 0;
    for(i = 0; i < count; i++) {
        if(results[i] == sin(data[i]+data[i]))
            correct++;
		printf("%f-----%f\n",results[i],sin(data[i]+data[i]));
    }
    printf("'%d/%d' correct values!\n", correct, count);


	free(data);
	free(results);
    return 0;
}
