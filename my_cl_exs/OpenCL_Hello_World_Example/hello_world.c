#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;

    FILE* programHandle;
    size_t programSize, kernelSourceSize;
    char *programBuffer, *kernelSource;

    // get first available platform and gpu and create context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // get size of kernel source
    programHandle = fopen("kernel.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1,
            (const char**) &programBuffer, &programSize, NULL);
    free(programBuffer);

    // read kernel source back in from program to check
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &kernelSourceSize);
    kernelSource = (char*) malloc(kernelSourceSize);
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, kernelSourceSize, kernelSource, NULL);
    printf("nKernel source:nn%sn", kernelSource);
    free(kernelSource);

    clReleaseContext(context);
    return 0;

}

