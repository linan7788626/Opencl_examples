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

#define MAX_SOURCE_SIZE (0x100000)

void call_kernel_2d(float *x1,float *x2,float *x11,float *x12,float *x21,float *x22,float Dcell,int Nc,int dif_tag,char * cl_name) {

	//FILE* programHandle;
    //size_t programSize, KernelSourceSize;
    //char *programBuffer, *KernelSource;

	int err;
	int DIM = 2;
	size_t global[DIM];
	size_t local[DIM];
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_uint nd;
	cl_mem input1,input2,output1,output2,output3,output4;

	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	commands = clCreateCommandQueue(context, device_id, 0, &err);

	input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);
	input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);
	output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);
	output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);
	output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);
	output4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * Nc*Nc, NULL, NULL);


	//----------------------------------------------------------------------------
	err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * Nc*Nc, x1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * Nc*Nc, x2, 0, NULL, NULL);
//	//----------------------------------------------------------------------------
//    // get size of kernel source
//    programHandle = fopen(cl_name, "r");
//    fseek(programHandle, 0, SEEK_END);
//    programSize = ftell(programHandle);
//    rewind(programHandle);
//
//    programBuffer = (char*) malloc(programSize + 1);
//    programBuffer[programSize] = '\0';
//    fread(programBuffer, sizeof(char), programSize, programHandle);
//    fclose(programHandle);
//
//    // create program from buffer
//    program = clCreateProgramWithSource(context,1,(const char**) &programBuffer,&programSize, NULL);
//    free(programBuffer);
//
//    // read kernel source back in from program to check
//    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &KernelSourceSize);
//    KernelSource = (char*) malloc(KernelSourceSize);
//    clGetProgramInfo(program, CL_PROGRAM_SOURCE, KernelSourceSize, KernelSource, NULL);
//
//    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
//    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
//    kernel = clCreateKernel(program, "lanczos_2_cl", &err);
//---------------------------------------------------------------------
///* Load kernel source file */
	FILE * fp;
	const char fileName[] = "./dataParallel.cl";
	size_t KernelSourceSize;
	char *KernelSource;
	fp = fopen("./lanczos_2_opencl.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	KernelSource = (char *)malloc(MAX_SOURCE_SIZE);
	KernelSourceSize = fread(KernelSource, 1, MAX_SOURCE_SIZE, fp);
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "lanczos_2_cl", &err);
	fclose(fp);
//---------------------------------------------------------------------


	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output1);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &output2);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &output3);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &output4);
	clSetKernelArg(kernel, 6, sizeof(float), &Dcell);
	clSetKernelArg(kernel, 7, sizeof(int), &Nc);
	clSetKernelArg(kernel, 8, sizeof(int), &dif_tag);

	global[0] = (size_t) Nc; global[1] = (size_t) Nc;
	int ndim = 2;
	err = clEnqueueNDRangeKernel(commands, kernel, ndim, NULL, global, NULL, 0, NULL, NULL); clFinish(commands);

	err = clEnqueueReadBuffer(commands, output1, CL_TRUE, 0, sizeof(float) * Nc*Nc, x11, 0, NULL, NULL );
	err = clEnqueueReadBuffer(commands, output2, CL_TRUE, 0, sizeof(float) * Nc*Nc, x12, 0, NULL, NULL );
	err = clEnqueueReadBuffer(commands, output3, CL_TRUE, 0, sizeof(float) * Nc*Nc, x21, 0, NULL, NULL );
	err = clEnqueueReadBuffer(commands, output4, CL_TRUE, 0, sizeof(float) * Nc*Nc, x22, 0, NULL, NULL );

    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(output1);
    clReleaseMemObject(output2);
    clReleaseMemObject(output3);
    clReleaseMemObject(output4);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    //printf("nKernel source:\n\n %s \n", KernelSource);
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
	float boxsize = 1.0;
	int Nc = 1024;
	float Dcell = boxsize/(float)(Nc);
	float *A1 = (float *) malloc(Nc*Nc*sizeof(float));
	float *A2 = (float *) malloc(Nc*Nc*sizeof(float));
	float *A11 = (float *) malloc(Nc*Nc*sizeof(float));
	float *A12 = (float *) malloc(Nc*Nc*sizeof(float));
	float *A21 = (float *) malloc(Nc*Nc*sizeof(float));
	float *A22 = (float *) malloc(Nc*Nc*sizeof(float));

	float *A11_c = (float *) malloc(Nc*Nc*sizeof(float));
	float *A12_c = (float *) malloc(Nc*Nc*sizeof(float));
	float *A21_c = (float *) malloc(Nc*Nc*sizeof(float));
	float *A22_c = (float *) malloc(Nc*Nc*sizeof(float));

	int i;
	for (i = 0; i < Nc*Nc; i++) {
		A1[i] = rand() / (float)RAND_MAX;
		A2[i] = rand() / (float)RAND_MAX;
	}

	lanczos_diff_2_tag(A1,A2,A11_c,A12_c,A21_c,A22_c,Dcell,Nc,-1);
	//mat_add(n1,n2,n3,A,B,C_c);
	call_kernel_2d(A1,A2,A11,A12,A21,A22,Dcell,Nc,-1,"./lanczos_2_opencl.cl");

	for (i = 0;i<Nc*Nc;i++) {
		printf("%f-----||%f \n",A11_c[i], A11[i]);
	}
	return 0;
}
