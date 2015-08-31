#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#define MAX_SOURCE_SIZE (0x100000)

float call_kernel_integral(float a,float b,int div,int ncount,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem output;                      // device memory used for the output array
//----------------------------------------------------------------------------
// Setup Context of OpenCL
	int err;
    int gpu = 1;

    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
//---------------------------------------------------------------------
//* Load kernel source file */
	FILE * fp;
	const char fileName[] = "./numerical_interation_opencl.cl";
	size_t KernelSourceSize;
	char *KernelSource;
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	KernelSource = (char *)malloc(MAX_SOURCE_SIZE);
	KernelSourceSize = fread(KernelSource, 1, MAX_SOURCE_SIZE, fp);
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "comp_trap_multi", &err);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
	float *res = (float *)malloc(sizeof(float));

    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(float), &a);
    err = clSetKernelArg(kernel, 1, sizeof(float), &b);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err = clSetKernelArg(kernel, 3, sizeof(int), &ncount);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = ncount;
	local = 64;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(commands);
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float),res, 0, NULL, NULL );
//----------------------------------------------------------------------------
    clReleaseMemObject(output);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    //printf("nKernel source:\n\n %s \n", KernelSource);
    free(KernelSource);
	return res[0];
}

//float f(float x) {
//    return x*x;
//}

float f(float x) {
	return 16.0*(x-1.0)/(x*x*x*x-2*x*x*x+4*x-4);//(0,1)
}

//float f(float x) {
//	return 1.0/(x+1.0);//(0,1)
//}

float Simple_Trap(float a, float b) {
    float fA, fB;
    fA = f(a);
    fB = f(b);
    fA = f(a);
    fB = f(b);
    return ((fA + fB) * (b-a)) / 2;
}

float Comp_Trap( float a, float b,int n) {
    float Suma = 0;
    float i = 0;
	float INC = (b-a)/(float)n;
    i = a + INC;
    Suma += Simple_Trap(a,i);
    while(i < b)
    {
        i+=INC;
        Suma += Simple_Trap(i,i + INC);
    }
    return Suma;
}

float Simpson_Comp_Trap( float a, float b,int n) {
    float Suma = 0;
	float INC = (b-a)/(float)(n);
	float xt,yt;
	float y0 = f(a)+f(b);

    int i = 0;
	for (i = 1; i < n; i++) {
		xt =  a + INC*i;
		yt = pow(2.0,i%2+1)*f(xt);
		Suma = Suma + yt;
	}
	Suma = INC/3.0*(y0+Suma);
    return Suma;
}

int main(int argc, const char *argv[]) {
	float res1 = 0.0;
	float res2 = 0.0;
	float res3 = 0.0;
	float a = 0.0;
	float b = 1.0;
	int n = 128;

	res1 = Comp_Trap(a,b,n);
	res2 = Simpson_Comp_Trap(a,b,n);
	res3 = call_kernel_integral(a,b,4.0,n,"./numerical_interation_opencl.cl");
	//printf("-----%f-----%f\n", res,pow(b-a,3.0)/3.0);
	printf("-----%f-----%f-----%f----%f\n", res1,res2,res3,M_PI);

	return 0;
}
