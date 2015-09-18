#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

void call_kernel_2d(float *x1,float *x2,int n1,int n2,int n3,float *x3,char * cl_name) {

    FILE* programHandle;
    size_t programSize, KernelSourceSize;
    char *programBuffer, *KernelSource;

	int err, szA, szB, szC;
	szA = n1*n2;
	szB = n2*n3;
	szC = n1*n3;

	int DIM = 2;
	size_t global[DIM];
	size_t local[DIM];
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_uint nd;
	cl_mem input1, input2, output;

	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szA, NULL, NULL);
	input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szB, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * szA, x1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * szB, x2, 0, NULL, NULL);
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

	err  = clSetKernelArg(kernel, 0, sizeof(int), &n1);
	err |= clSetKernelArg(kernel, 1, sizeof(int), &n2);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &n3);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &input1);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &input2);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);

	global[0] = (size_t) n1; global[1] = (size_t) n3;
	int ndim = 2;
	err = clEnqueueNDRangeKernel(commands, kernel, ndim, NULL, global, NULL, 0, NULL, NULL); clFinish(commands);
	err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * szC, x3, 0, NULL, NULL );

    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(output);
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
	int n1 = 512,n2 = 256,n3 = 1024;
	float *A = (float *) malloc(n1*n2*sizeof(float));
	float *B = (float *) malloc(n2*n3*sizeof(float));
	float *C = (float *) malloc(n1*n3*sizeof(float));
	float *C_c = (float *) malloc(n1*n3*sizeof(float));

	int i;
	for (i = 0; i < n1*n2; i++) {
		A[i] = rand() / (float)RAND_MAX;
	}
	for (i = 0; i < n2*n3; i++) {
		B[i] = rand() / (float)RAND_MAX;
	}

	//mat_mul(n1,n2,n3,A,B,C_c);
	call_kernel_2d(A,B,n1,n2,n3,C,"./matrix_mul2d.cl");
	//call_kernel_2d(A,B,n1,n2,n3,C,"./matrix_mul2d_pri_only.cl");

	//for (i = 0;i<n1*n3;i++) {
	//	printf("%f-----%f \n",C_c[i], C[i]);
	//}
	return 0;
}
