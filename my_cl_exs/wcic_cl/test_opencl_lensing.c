#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include "wcic.h"

void call_kernel_wcic(float *cic_in,float *x_in,float *y_in,float bsx,float bsy,int nx,int ny,int np,float *cic_out,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization
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
    kernel = clCreateKernel(program, "wcic_cl", &err);
//----------------------------------------------------------------------------
// Allocate Memory for Device
    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    input3 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nx*ny, NULL, NULL);
//----------------------------------------------------------------------------
// Copy Data to Device
    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * np, cic_in, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * np, x_in, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input3, CL_TRUE, 0, sizeof(float) * np, y_in, 0, NULL, NULL);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 4, sizeof(float), &bsx);
    clSetKernelArg(kernel, 5, sizeof(float), &bsy);
    clSetKernelArg(kernel, 6, sizeof(int), &nx);
    clSetKernelArg(kernel, 7, sizeof(int), &ny);
    clSetKernelArg(kernel, 8, sizeof(int), &np);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = np;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	printf("--------------------------%d\n", err);
    clFinish(commands);
//----------------------------------------------------------------------------
// Output Array
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * nx*ny, cic_out, 0, NULL, NULL );
	printf("--------------------------%d\n", err);
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

	float bsx = 1.0;
	float bsy = 1.0;

	int nx = 256;
	int ny = 256;
	int np = 1024*1024;

    float *cic_in = (float *)malloc(sizeof(float)*np);
    float *x_in = (float *)malloc(sizeof(float)*np);
    float *y_in = (float *)malloc(sizeof(float)*np);
    float *cic_out = (float *)malloc(sizeof(float)*nx*ny);
    int correct;

    int i = 0;
    for(i = 0; i < np; i++) {
		cic_in[i] = rand() / (float)RAND_MAX;
		x_in[i] = rand() / (float)RAND_MAX;
		y_in[i] = rand() / (float)RAND_MAX;
	}

	//call_kernel(xi1,xi2,count,lpar,alpha1,alpha2,"./play_with.cl");
	call_kernel_wcic(cic_in,x_in,y_in,bsx,bsy,nx,ny,np,cic_out,"./wcic_opencl.cl");

    float *cic_out_c = (float *)malloc(sizeof(float)*nx*ny);
	//inverse_cic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map_c);
	wcic(cic_in,x_in,y_in,bsx,bsy,nx,ny,np,cic_out_c);
    correct = 0;
    for(i = 0; i < nx*ny; i++) {
		//lq_nie(xi1[i],xi2[i],lpar,&alpha1_c[i],&alpha2_c[i]);
		//printf("%f-----%f||%f-----%f\n",alpha1[i],alpha1_c[i],alpha2[i],alpha2_c[i]);
		//inverse_cic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map_c);
		//printf("%f-----%f|\n",lensed_map[i],lensed_map_c[i]);
		printf("%f-----%f|\n",cic_out_c[i],cic_out[i]);
    }

	free(cic_in);
	free(x_in);
	free(y_in);
	free(cic_out);
	free(cic_out_c);
    return 0;
}
