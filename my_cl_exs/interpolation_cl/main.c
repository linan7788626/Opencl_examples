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

void call_kernel_icubic(float *source_map,float *posy1,float *posy2, float ysc1, float ysc2, float dsi,
		int nsx,int nsy,int nlx,int nly,float *lensed_map,char * cl_name) {
//----------------------------------------------------------------------------
// Initialization
	int counts = nsx*nsy;
	int countl = nlx*nly;

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
	printf("------------------------------------%d\n", err);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
//---------------------------------------------------------------------
//* Load kernel source file */
	FILE * fp;
	const char fileName[] = "./ocl_bicubic_interpolation.cl";
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
	printf("------------------------------------%d\n", err);
	kernel = clCreateKernel(program, "inverse_bicubic_cl", &err);
	printf("------------------------------------%d\n", err);
//----------------------------------------------------------------------------
// Allocate Memory for Device
    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * counts, NULL, NULL);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * countl, NULL, NULL);
    input3 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * countl, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * countl, NULL, NULL);
//----------------------------------------------------------------------------
// Copy Data to Device
    err = clEnqueueWriteBuffer(commands, input1, CL_TRUE, 0, sizeof(float) * counts, source_map, 0, NULL, NULL);
	printf("------------------------------------%d\n", err);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * countl, posy1, 0, NULL, NULL);
	printf("------------------------------------%d\n", err);
    err = clEnqueueWriteBuffer(commands, input3, CL_TRUE, 0, sizeof(float) * countl, posy2, 0, NULL, NULL);
	printf("------------------------------------%d\n", err);
//----------------------------------------------------------------------------
// Passing Parameters into Kernel Functions
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 4, sizeof(float), &ysc1);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 5, sizeof(float), &ysc1);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 6, sizeof(float), &dsi);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &nsx);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 8, sizeof(int), &nsy);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &nlx);
	printf("------------------------------------%d\n", err);
    err = clSetKernelArg(kernel,10, sizeof(int), &nly);
	printf("------------------------------------%d\n", err);
//----------------------------------------------------------------------------
// Runing Kernel Functions
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	printf("------------------------------------%d\n", err);
    global = countl;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	printf("------------------------------------%d\n", err);
    clFinish(commands);
//----------------------------------------------------------------------------
// Output Array
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * countl, lensed_map, 0, NULL, NULL );
	printf("------------------------------------%d\n", err);
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

float CatMullRom(float x) {
    const float B = 0.0;
    const float C = 0.5;
    float f = x;
    if (f<0.0) f = -f;
    if (f<1.0) {
        return ((12-9*B-6*C)*(f*f*f)+(-18+12*B+6*C)*(f*f)+(6-2*B))/6.0;
    }
    else if (f>=1.0 && f<2.0) {
        return ((-B-6*C)*(f*f*f)+(6*B+30*C)*(f*f)+(-(12*B)-48*C)*f+8*B+24*C)/6.0;
    }
    else {
        return 0.0;
    }
}

float BSpline_func(float x) {
	float f = x;
	if (f<0.0) f = -f;

	if (f<=0 && f<=1) {
		return 2.0/3.0+0.5*pow(f,3.0)-f*f;
	}
	if (f>1 && f<=2) {
		return 1.0/6.0*pow((2-f),3.0);
	}
	return 0.0;
}

float BellFunc(float x) {
	float f = (x/2.0)*1.5; // Converting -2 to +2 to -1.5 to +1.5
	if( f>-1.5 && f<-0.5 ) {
		return(0.5*pow(f+1.5,2.0));
	}
	else if(f>-0.5 && f<0.5) {
		return 3.0/4.0-(f*f);
	}
	else if(f>0.5 && f<1.5) {
		return(0.5*pow(f-1.5,2.0));
	}
	return 0.0;
}


void  inverse_bicubic(float *img_in, float *posy1,float *posy2,float ysc1,float ysc2,float dsi,int nsx, int nsy, int nlx, int nly, float *img_out) {
	int i1,j1,i,j,k,l,ii,jj;
	int index;
	float xb1,xb2;
	float dx,dy;

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			ii = i1+k;
			jj = j1+l;
			dx = xb1-(float)(ii);
			dy = xb2-(float)(jj);
			img_out[index] = img_out[index] + CatMullRom(dx)*CatMullRom(dy)*img_in[ii*nsx+jj];
		}
    }
}

void call_factors(float p[4][4],float a[4][4]) {
		a[0][0] = p[1][1];
		a[0][1] = -.5*p[1][0] + .5*p[1][2];
		a[0][2] = p[1][0] - 2.5*p[1][1] + 2*p[1][2] - .5*p[1][3];
		a[0][3] = -.5*p[1][0] + 1.5*p[1][1] - 1.5*p[1][2] + .5*p[1][3];
		a[1][0] = -.5*p[0][1] + .5*p[2][1];
		a[1][1] = .25*p[0][0] - .25*p[0][2] - .25*p[2][0] + .25*p[2][2];
		a[1][2] = -.5*p[0][0] + 1.25*p[0][1] - p[0][2] + .25*p[0][3] + .5*p[2][0] - 1.25*p[2][1] + p[2][2] - .25*p[2][3];
		a[1][3] = .25*p[0][0] - .75*p[0][1] + .75*p[0][2] - .25*p[0][3] - .25*p[2][0] + .75*p[2][1] - .75*p[2][2] + .25*p[2][3];
		a[2][0] = p[0][1] - 2.5*p[1][1] + 2*p[2][1] - .5*p[3][1];
		a[2][1] = -.5*p[0][0] + .5*p[0][2] + 1.25*p[1][0] - 1.25*p[1][2] - p[2][0] + p[2][2] + .25*p[3][0] - .25*p[3][2];
		a[2][2] = p[0][0] - 2.5*p[0][1] + 2*p[0][2] - .5*p[0][3] - 2.5*p[1][0] + 6.25*p[1][1] - 5*p[1][2] + 1.25*p[1][3] + 2*p[2][0] - 5*p[2][1] + 4*p[2][2] - p[2][3] - .5*p[3][0] + 1.25*p[3][1] - p[3][2] + .25*p[3][3];
		a[2][3] = -.5*p[0][0] + 1.5*p[0][1] - 1.5*p[0][2] + .5*p[0][3] + 1.25*p[1][0] - 3.75*p[1][1] + 3.75*p[1][2] - 1.25*p[1][3] - p[2][0] + 3*p[2][1] - 3*p[2][2] + p[2][3] + .25*p[3][0] - .75*p[3][1] + .75*p[3][2] - .25*p[3][3];
		a[3][0] = -.5*p[0][1] + 1.5*p[1][1] - 1.5*p[2][1] + .5*p[3][1];
		a[3][1] = .25*p[0][0] - .25*p[0][2] - .75*p[1][0] + .75*p[1][2] + .75*p[2][0] - .75*p[2][2] - .25*p[3][0] + .25*p[3][2];
		a[3][2] = -.5*p[0][0] + 1.25*p[0][1] - p[0][2] + .25*p[0][3] + 1.5*p[1][0] - 3.75*p[1][1] + 3*p[1][2] - .75*p[1][3] - 1.5*p[2][0] + 3.75*p[2][1] - 3*p[2][2] + .75*p[2][3] + .5*p[3][0] - 1.25*p[3][1] + p[3][2] - .25*p[3][3];
		a[3][3] = .25*p[0][0] - .75*p[0][1] + .75*p[0][2] - .25*p[0][3] - .75*p[1][0] + 2.25*p[1][1] - 2.25*p[1][2] + .75*p[1][3] + .75*p[2][0] - 2.25*p[2][1] + 2.25*p[2][2] - .75*p[2][3] - .25*p[3][0] + .75*p[3][1] - .75*p[3][2] + .25*p[3][3];
}

float getValue (float x, float y,float a[4][4]) {
	float x2 = x * x;
	float x3 = x2 * x;
	float y2 = y * y;
	float y3 = y2 * y;

	return (a[0][0] + a[0][1]*y + a[0][2]*y2 + a[0][3]*y3) +
	       (a[1][0] + a[1][1]*y + a[1][2]*y2 + a[1][3]*y3)*x +
	       (a[2][0] + a[2][1]*y + a[2][2]*y2 + a[2][3]*y3)*x2 +
	       (a[3][0] + a[3][1]*y + a[3][2]*y2 + a[3][3]*y3)*x3;
}

void  inverse_bicubic_polygen(float *img_in, float *posy1,float *posy2,float ysc1,float ysc2,float dsi,int nsx, int nsy, int nlx, int nly, float *img_out) {
	int i1,j1,i,j,k,l;
	int index;
	float xb1,xb2;
	float dx,dy;
	float p[4][4];
	float a[4][4];

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;
		dx = xb1-(float)(i1);
		dy = xb2-(float)(j1);

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			p[k+1][l+1] = img_in[(i1+k)*nsx+(j1+l)];
		}
		call_factors(p,a);
		img_out[index] = getValue(dx,dy,a);
    }
}

int main() {

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


	call_kernel_icubic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map,"./ocl_bicubic_interpolation.cl");

    float *lensed_map_c = (float *)malloc(sizeof(float)*countl);
	inverse_bicubic(source_map,posy1,posy2,ysc1,ysc2,dsi,nsx,nsy,nlx,nly,lensed_map_c);
    correct = 0;
    for(i = 0; i < countl; i++) {
		printf("%f-----%f|\n",lensed_map[i],lensed_map_c[i]);
    }

	free(source_map);
	free(posy1);
	free(posy2);
	free(lensed_map);
	free(lensed_map_c);
	return 0;
}
