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
#include "icic_omp.h"

void call_kernel_icic(float *source_map,float *posy1,float *posy2, float ysc1, float ysc2, float dsi,
		int nsx,int nsy,int nlx,int nly,float *lensed_map,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization
	int counts = nsx*nsy;
	int countl = nlx*nly;
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
//----------------------------------------------------------------------------
// Claim Variables for Device
    cl_mem input1;                       // device memory used for the input array
    cl_mem input2;                       // device memory used for the input array
    cl_mem input3;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
//----------------------------------------------------------------------------
// Setup Context of OpenCL
	int err;
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
//----------------------------------------------------------------------------
// Read Kernel Functions from Source Files
    programHandle = fopen(cl_name, "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    program = clCreateProgramWithSource(context,1,(const char**) &programBuffer,&programSize, NULL);
    free(programBuffer);

    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &KernelSourceSize);
    KernelSource = (char*) malloc(KernelSourceSize);
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, KernelSourceSize, KernelSource, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "icic_cl", &err);
//----------------------------------------------------------------------------
// Allocate Memory for Device
    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * counts, NULL, NULL);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * countl, NULL, NULL);
    input3 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * countl, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * countl, NULL, NULL);
//----------------------------------------------------------------------------
// Copy Data to Device
    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * counts, source_map, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * countl, posy1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input3, CL_TRUE, 0, sizeof(float) * countl, posy2, 0, NULL, NULL);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 4, sizeof(float), &ysc1);
    clSetKernelArg(kernel, 5, sizeof(float), &ysc1);
    clSetKernelArg(kernel, 6, sizeof(float), &dsi);
    clSetKernelArg(kernel, 7, sizeof(int), &nsx);
    clSetKernelArg(kernel, 8, sizeof(int), &nsy);
    clSetKernelArg(kernel, 9, sizeof(int), &nlx);
    clSetKernelArg(kernel,10, sizeof(int), &nly);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = countl;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(commands);
//----------------------------------------------------------------------------
// Output Array
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * countl, lensed_map, 0, NULL, NULL );
//----------------------------------------------------------------------------
// Free the Memory in Device
    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(input3);
    clReleaseMemObject(output);
//----------------------------------------------------------------------------

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

	float ysc1 = 0.0;
	float ysc2 = 0.0;
	float dsi = 0.03;

	int nsx = 256;
	int nsy = 256;
	int counts = nsx*nsy;
	int nlx = 256;
	int nly = 256;
	int countl = nlx*nly;

    float *lensed_map = (float *)malloc(sizeof(float)*countl);
    float *posy1 = (float *)malloc(sizeof(float)*countl);
    float *posy2 = (float *)malloc(sizeof(float)*countl);
    float *source_map = (float *)malloc(sizeof(float)*counts);
    int correct;

    int i = 0;
    for(i = 0; i < counts; i++) {
		source_map[i] = rand() / (float)RAND_MAX;
	}
    for(i = 0; i < countl; i++) {
		posy1[i] = rand() / (float)RAND_MAX;
		posy2[i] = rand() / (float)RAND_MAX;
	}


	//call_kernel(xi1,xi2,count,lpar,alpha1,alpha2,"./play_with.cl");
	call_kernel_icic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map,"./icic_opencl.cl");

    float *lensed_map_c = (float *)malloc(sizeof(float)*countl);
	inverse_cic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map_c);
    correct = 0;
    for(i = 0; i < countl; i++) {
		//lq_nie(xi1[i],xi2[i],lpar,&alpha1_c[i],&alpha2_c[i]);
		//printf("%f-----%f||%f-----%f\n",alpha1[i],alpha1_c[i],alpha2[i],alpha2_c[i]);
		//inverse_cic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map_c);
		printf("%f-----%f|\n",lensed_map[i],lensed_map_c[i]);
    }

	free(source_map);
	free(posy1);
	free(posy2);
	free(lensed_map);
	free(lensed_map_c);
    return 0;
}
