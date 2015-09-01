#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include "intfuncs.h"
#include "allvars_SPH.h"
#include "proto.h"

#define MAX_SOURCE_SIZE (0x100000)

void call_kernel_sph(float *x1_in,float *x2_in,float *SmoothLength,float bsz,int nc,int np,float *sdens_out,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization
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
	printf("--------------------------%d\n", err);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
//---------------------------------------------------------------------
//* Load kernel source file */
	FILE * fp;
	const char fileName[] = "./sph_opencl.cl";
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
	kernel = clCreateKernel(program, "sph_cl", &err);
//----------------------------------------------------------------------------
// Allocate Memory for Device
    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    input3 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * np, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nc*nc, NULL, NULL);
//----------------------------------------------------------------------------
// Copy Data to Device
    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * np, x1_in, 0, NULL, NULL);
	printf("--------------------------%d\n", err);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * np, x2_in, 0, NULL, NULL);
	printf("--------------------------%d\n", err);
    err = clEnqueueWriteBuffer(commands, input3, CL_TRUE, 0, sizeof(float) * np, SmoothLength, 0, NULL, NULL);
	printf("--------------------------%d\n", err);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 4, sizeof(float), &bsz);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &nc);
	printf("--------------------------%d\n", err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &np);
	printf("--------------------------%d\n", err);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    //err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	//printf("--------------------------%d\n", err);
    global = np;
	printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global,NULL, 0, NULL, NULL);
    //err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,NULL,NULL, 0, NULL, NULL);
	printf("--------------------------%d\n", err);
    clFinish(commands);
//----------------------------------------------------------------------------
// Output Array
	printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * nc*nc, sdens_out, 0, NULL, NULL );
	printf("--------------------------^^^^^^^^^^%d\n", err);
	printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
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

int main(int argc, const char *argv[])
{
	float bsz = 3.0;
	int nc = 512;
	int np = 200000;
	int ngb = 12;
	long Np = (long)np;
	long Ngb = (long)ngb;
	long Nc = (long)nc;

    float *x1_in = (float *)malloc(np*sizeof(float));
    float *x2_in = (float *)malloc(np*sizeof(float));
    float *x3_in = (float *)malloc(np*sizeof(float));
    float *SmoothLength = (float *)malloc(np*sizeof(float));

    float *sdens_out = (float *)malloc(sizeof(float)*nc*nc);
    float *sdens_out_c = (float *)malloc(sizeof(float)*nc*nc);

	int i,j,sph;
	PARTICLE *particle = (PARTICLE *)malloc(np*sizeof(PARTICLE));

    //for(i = 0; i < np; i++) {
	//	particle[i].x = rand() / (float)RAND_MAX-0.5;
	//	particle[i].y = rand() / (float)RAND_MAX-0.5;
	//	particle[i].z = rand() / (float)RAND_MAX-0.5;
	//	//SmoothLength[i] = rand() / (float)RAND_MAX;
  	//}
//--------------------------------------------------------------------
	Loadin_particle_main_ascii(Np,"./lib_so_omp_norm_sph/input_files/cnfw_2e5.dat",particle);

	double SPHBoxSize = 0.0;
	sph = findHsml(particle,&Np,&Ngb,&SPHBoxSize,SmoothLength);
	if (sph == 1) {
		printf("FindHsml is failed!\n");
	}

//--------------------------------------------------------------------
	Make_cell_SPH(Nc,bsz,Np,particle,SmoothLength,sdens_out_c);
//--------------------------------------------------------------------
	for(i=0;i<np;i++) {
  	  	x1_in[i] = particle[i].x;
  	  	x2_in[i] = particle[i].y;
  	  	x3_in[i] = particle[i].z;
  	}
	call_kernel_sph(x1_in,x2_in,SmoothLength,bsz,nc,np,sdens_out,"./sph_opencl.cl");

    //for(i = 0; i < nc*nc; i++) {
	//	if (sdens_out_c[i] > 0)
	//		printf("%f-----%f|\n",sdens_out_c[i],sdens_out[i]);
    //}
//--------------------------------------------------------------------

	free(SmoothLength);
	free(particle);
	free(x1_in);
	free(x2_in);
	free(sdens_out);
	free(sdens_out_c);

	return 0;
}
