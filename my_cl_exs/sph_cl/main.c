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
#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })


//--------------------------------------------------------------------
void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}
//--------------------------------------------------------------------
void write_1_signal(char *out1,float *in1, int Nc) {
	FILE *f1;
	f1 =fopen(out1,"wb");

	int i,j,index;

	for (i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		index = i*Nc+j;
		fwrite(&in1[index],sizeof(float),1,f1);
	}
	//fwrite(in1,sizeof(float),Nc*Nc,f1);
	fclose(f1);
}

void call_kernel_sph(float *x1_in,float *x2_in,float *SmoothLength,float bsz,int nc,int np,float *sdens_out,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    //cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
//----------------------------------------------------------------------------
// Claim Variables for Device
    cl_mem input1;                      // device memory used for the input array
    cl_mem input2;                      // device memory used for the input array
    cl_mem input3;                      // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
//----------------------------------------------------------------------------
// Setup Context of OpenCL
	int err;
    int gpu = 1;
	unsigned int ndevices = 0;
    //err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, &ndevices);
	//printf("--------------------------%d\n", ndevices);
    //context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    //commands = clCreateCommandQueue(context, device_id, 0, &err);

    clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, NULL, &ndevices);
	cl_device_id devices[ndevices];
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, ndevices, devices, NULL);
	printf("--------------------------%d\n",err);

    context = clCreateContext(NULL, ndevices, devices, NULL, NULL, &err);
	//context = CL_CHECK_ERR(clCreateContext(NULL, ndevices, devices, &pfn_notify, NULL, &_err));
	printf("--------------------------%d\n",err);
    commands = clCreateCommandQueue(context, devices[1], 0, &err);
	printf("--------------------------%d\n",err);
//---------------------------------------------------------------------
//* Load kernel source file */
	int MAX_SOURCE_SIZE = 1048576;
	FILE * fp;
	const char fileName[] = "./sph_opencl.cl";
	size_t KernelSourceSize;
	char *KernelSource;
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	KernelSource = (char *)malloc(MAX_SOURCE_SIZE*sizeof(char));
	KernelSourceSize = fread(KernelSource,sizeof(char), MAX_SOURCE_SIZE, fp);
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	printf("--------------------------%d\n",err);
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "sph_cl", &err);
	printf("--------------------------%d\n",err);
//----------------------------------------------------------------------------
// Allocate Memory for Device
    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*np, NULL, NULL);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*np, NULL, NULL);
    input3 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*np, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*nc*nc, NULL, NULL);
//----------------------------------------------------------------------------
// Copy Data to Device
    clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float)*np, x1_in, 0, NULL, NULL);
	printf("--------------------------%d\n",err);
    clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float)*np, x2_in, 0, NULL, NULL);
	printf("--------------------------%d\n",err);
    clEnqueueWriteBuffer(commands, input3, CL_TRUE, 0, sizeof(float)*np, SmoothLength, 0, NULL, NULL);
	printf("--------------------------%d\n",err);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	printf("passing--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	printf("--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
	printf("--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
	printf("--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 4, sizeof(float), &bsz);
	printf("--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &nc);
	printf("--------------------------%d\n",err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &np);
	printf("passing--------------------------%d\n",err);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    //clGetKernelWorkGroupInfo(kernel, devices[1], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = np;
	local = 1000;

    //clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global,&local, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global,&local, 0, NULL, NULL);
	printf("--------------------------%d\n",err);
    //err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,&global,NULL, 0, NULL, NULL);
    clFinish(commands);
//----------------------------------------------------------------------------
// Output Array
	printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
    clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float)*nc*nc, sdens_out, 0, NULL, NULL );
	printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
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
	int nc = 1024*2;
	int np = 200000;
	int ngb = 4;
	long Np = (long)np;
	long Ngb = (long)ngb;
	//long Nc = (long)nc;

    float *x1_in = (float *)malloc(np*sizeof(float));
    float *x2_in = (float *)malloc(np*sizeof(float));
    float *x3_in = (float *)malloc(np*sizeof(float));
    float *SmoothLength = (float *)malloc(np*sizeof(float));

    float *sdens_out = (float *)malloc(sizeof(float)*nc*nc);
    float *sdens_out_c = (float *)malloc(sizeof(float)*nc*nc);

	int i,sph;
	PARTICLE *particle = (PARTICLE *)malloc(np*sizeof(PARTICLE));

    //for(i = 0; i < np; i++) {
	//	//particle[i].x = rand() / (float)RAND_MAX-0.5;
	//	//particle[i].y = rand() / (float)RAND_MAX-0.5;
	//	//particle[i].z = rand() / (float)RAND_MAX-0.5;
	//	SmoothLength[i] = 0.05;
  	//}
//--------------------------------------------------------------------
	Loadin_particle_main_ascii(Np,"./lib_so_omp_norm_sph/input_files/cnfw_2e5.dat",particle);

	double SPHBoxSize = 0.0;
	sph = findHsml(particle,&Np,&Ngb,&SPHBoxSize,SmoothLength);
	if (sph == 1) {
		printf("FindHsml is failed!\n");
	}
	free(particle);

//--------------------------------------------------------------------
	particle = (PARTICLE *)malloc(np*sizeof(PARTICLE));
	Loadin_particle_main_ascii(Np,"./lib_so_omp_norm_sph/input_files/cnfw_2e5.dat",particle);
	//Make_cell_SPH(Nc,bsz,Np,particle,SmoothLength,sdens_out_c);
//--------------------------------------------------------------------
	for(i=0;i<np;i++) {
  	  	x1_in[i] = particle[i].x;
  	  	x2_in[i] = particle[i].y;
  	  	x3_in[i] = particle[i].z;
  	}
	free(particle);
	call_kernel_sph(x1_in,x2_in,SmoothLength,bsz,nc,np,sdens_out,"./sph_opencl.cl");
	//call_kernel_sph(x1_in,x2_in,x3_in,bsz,nc,np,sdens_out,"./sph_opencl.cl");

    //for(i = 0; i < nc*nc; i++) {
	//	if (sdens_out_c[i] > 0)
	//		printf("%f-----%f|\n",sdens_out_c[i],sdens_out[i]);
    //}
	//write_1_signal("cpu_sdens.bin",sdens_out_c,nc);
	write_1_signal("gpu_sdens.bin",sdens_out,nc);
//--------------------------------------------------------------------

	free(SmoothLength);
	free(x1_in);
	free(x2_in);
	free(sdens_out);
	free(sdens_out_c);

	return 0;
}
