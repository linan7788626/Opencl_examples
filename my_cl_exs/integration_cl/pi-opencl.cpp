#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <time.h>
#include <iostream>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

char * kernels = "size_t t2l()\
{\
    size_t gsx = get_global_size(0);\
    size_t gsy = get_global_size(1);\
    size_t gsz = get_global_size(2);\
    size_t i = \
        get_global_id(0)\
        + get_global_id(1) * gsx \
        + get_global_id(2) * gsx * gsy\
        ;\
    return i;\
}\
\
__kernel void init(__global float * v, int n)\
{\
    size_t i = t2l();\
    if (i > n)\
        return;\
        \
    if (i < 0)\
        return;\
        \
    float w = 1.0 / n;\
    v[i] = i * w;\
}\
\
__kernel void map(__global float * v, int n)\
{\
    size_t i = t2l();\
         \
    if (i > n)\
        return;\
        \
    if (i < 0)\
        return;\
        \
    float vv = v[i];\
    \
    float w = 1.0 / (1 + vv * vv);\
    if (i == 0)\
        ;\
    else if (i == n)\
        ;\
    else if (i % 2 == 0)\
    {\
        w = w * 2;\
    }\
    else\
    {\
        w = w * 4;\
    }\
    v[i] = w;\
}\
\
__kernel void kernel_reduce_a(__global float * a, int n, int s)\
{\
    size_t i = t2l();\
    \
    i = i * s;\
         \
    if (i >= n)\
        return;\
        \
    if (i < 0)\
        return;\
        \
    a[i] = a[i] + a[i + s/2];\
    if (i == 0)\
       a[1] = s/2;\
}\
";


void get_device_info(cl_device_id device_id, cl_device_info device_info, std::string* value, cl_int * err)
{
    size_t size = 0;

    //  Get all params for the given platform id, first query their size, then get the actual data
    *err = clGetDeviceInfo(device_id, device_info, 0, NULL, &size);
    value->resize(size);
    *err = clGetDeviceInfo(device_id, device_info, size, &((*value)[0]), NULL);
}


void get_platform_info(cl_platform_id platform_id, cl_platform_info platform_info, std::string* value, cl_int * err)
{
    ::size_t size = 0;

    //  Get all params for the given platform id, first query their size, then get the actual data
    *err = clGetPlatformInfo(platform_id, platform_info, 0, NULL, &size);
    value->resize(size);
    *err = clGetPlatformInfo(platform_id, platform_info, size, &((*value)[0]), NULL);
}


cl_program load_binary(cl_context context, cl_platform_id platform, cl_device_id device, char * file_name, cl_int * err)
{
    size_t len;

    // for now, compile inlined source.
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernels, 0, err);
    char **binaries = 0;
    if (*err)
    {
        return 0;
    }
    *err = clBuildProgram(program, 1, &device, "-cl-opt-disable", NULL, NULL);
    //char log[5000];
    //cl_int err2 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    //printf("log = %s\n", log);
    return program;
}

int the_blocksize = 256;

void l2t(int size, int max_dimensionality, size_t * tile_size, size_t * tiles)
{
    for (int j = 0; j < max_dimensionality; ++j)
        tiles[j] = 1;
    int max_threads[] = { the_blocksize, 64, 64};
    int max_blocks[] = { 65535, 65535, 65535};
    for (int j = 0; j < max_dimensionality; ++j)
        tile_size[j] = 1;

    int b = size / (max_threads[0] * max_blocks[0]);
    if (b == 0)
    {
        int b = size / max_threads[0];
        if (size % max_threads[0] != 0)
            b++;

        if (b == 1)
            max_threads[0] = size;

        // done. return the result.
        tiles[0] = b;
        tile_size[0] = max_threads[0];

        // OpenCL uses multiples of tile_size.
        tiles[0] *= tile_size[0];
        return;

    }

    int sqrt_size = sqrt((float)size / max_threads[0]);
    sqrt_size++;

    int b2 = sqrt_size / max_blocks[1];
    if (b2 == 0)
    {
        int b = sqrt_size;

        // done. return the result.
        tiles[0] = tiles[1] = b;
        tile_size[0] = max_threads[0];

        // OpenCL uses multiples of tile_size.
        tiles[0] *= tile_size[0];
//        tiles[1] *= tile_size[1];
        return;
    }
    throw;
}

void CpuInit(float * v, int n)
{
    float w = 1.0 / n;
    for (int i = 0; i <= n; ++i)
        v[i] = i * w;
}

void CpuMap(float * v, int n)
{
    for (int i = 0; i <= n; ++i)
    {
        float vv = v[i];

        float w = 1.0 / (1 + vv * vv);
        if (i == 0)
            ;
        else if (i == n)
            ;
        else if (i % 2 == 0)
        {
            w = w * 2;
        }
        else
        {
            w = w * 4;
        }
        v[i] = w;
    }
}


void cpu_reduce(float * a, int n)
{
    // note: use double here because of accumulation of truncation error due to adding small numbers to a large number.
    double total = 0;
    for (int i = 0; i <= n; ++i)
        total += a[i];
    a[0] = total;
}


// From http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious
int flog2(int v)
{
    int x = 0;
    while ((v = (v >> 1)) != 0)
    {
        x++;
    }
    return (int)x;
}

// Compute the 2 ** v.
int pow2(int v)
{
    int value = 1;
    for ( ; v > 0; v--)
        value <<= 1;
    return value;
}

void CHECK(cl_int x)
{
    if (x != 0)
    {
        std::cout << "Error " << x << "\n";
        throw new std::string("Error " + x);
    }
}

void CpuPi(int np1)
{
    int n = np1 - 1;
    struct _timeb  t1;
    struct _timeb  t2;
    std::cout << "Starting tests...\n";
    _ftime_s(&t1);
    float * a = (float*)malloc(np1 * sizeof(float));
    CpuInit(a, n);
    CpuMap(a, n);
    cpu_reduce(a, n);
    float w = *a * 4.0 / 3 / n;
    free(a);
    _ftime(&t2);
    std::cout << (double)(t2.time - t1.time + ((double)(t2.millitm - t1.millitm))/1000) << " s.\n";

    std::cout << n << " " << std::setprecision(44) << w;
    std::cout << "\n";
}

void TryPlatform(cl_platform_id platform, int np1)
{
    try {

        int n = np1 - 1;

        // Choose the appropriate platform for this linked DLL.
        cl_device_id dev_id[10];
        cl_uint num;
        cl_int err;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 10, dev_id, &num);
        CHECK(err);

        printf("devices = %d\n", num);

        for (int d = 0; d < num; ++d)
        {
            char tbuf[500];
            size_t sz;

            std::cout << "            Device [" << d << "]" << std::endl;
            std::cout << "                type                          = ";
            {
                cl_device_type type;
                err = clGetDeviceInfo(dev_id[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
                CHECK(err);

                if (type & CL_DEVICE_TYPE_DEFAULT       ) std::cout << "CL_DEVICE_TYPE_DEFAULT "    ;
                if (type & CL_DEVICE_TYPE_CPU           ) std::cout << "CL_DEVICE_TYPE_CPU "        ;
                if (type & CL_DEVICE_TYPE_GPU           ) std::cout << "CL_DEVICE_TYPE_GPU "        ;
                if (type & CL_DEVICE_TYPE_ACCELERATOR   ) std::cout << "CL_DEVICE_TYPE_ACCELERATOR ";

                std::cout << std::endl;
            }

            err = clGetDeviceInfo(dev_id[d], CL_DEVICE_NAME, sizeof(tbuf), tbuf, NULL);
            CHECK(err);
            std::cout << "                name                          = " << tbuf << std::endl;

            // Choose device.
            cl_device_id device = dev_id[d];

            // create the OpenCL context on a GPU device
            cl_context_properties props[4];
            props[0] = CL_CONTEXT_PLATFORM;
            props[1] = (cl_context_properties)platform;
            props[2] = 0;
            cl_context context = clCreateContext(props, 1, &device, NULL, NULL, &err);
            CHECK(err);

            // create the program
            cl_program program = load_binary(context, platform, device, "pi-opencl-kernels.o", &err);
            CHECK(err);

            struct _timeb  t1;
            struct _timeb  t2;
            std::cout << "Starting tests...\n";
            _ftime_s(&t1);

            // for (int c = 0; c < 7; ++c)
            {
                cl_int r1;
                cl_mem da = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*np1, NULL, &r1);
                CHECK(r1);

                size_t asize = sizeof(float);
                float * a = (float*) malloc(asize);
                {
                    size_t tile[3];
                    size_t tiles[3];
                    int max_dimensionality = 3;

                    l2t(np1, max_dimensionality, tile, tiles);

                    {
                        // execute the kernel
                        cl_kernel kernel_init = clCreateKernel(program, "init", &err);
                        CHECK(err);
                        err = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), (void *) &da);
                        CHECK(err);
                        err = clSetKernelArg(kernel_init, 1, sizeof(int), (void *)&n);
                        CHECK(err);
                        // create a command-queue
                        cl_command_queue cmd_queue = clCreateCommandQueue(context, device, 0, &err);
                        CHECK(err);
                        cl_event my_event;
                        err = clEnqueueNDRangeKernel(cmd_queue, kernel_init, max_dimensionality, NULL, tiles, tile, 0, NULL, &my_event);
                        CHECK(err);
                        err = clWaitForEvents(1, &my_event);
                        CHECK(err);
                        err = clReleaseCommandQueue(cmd_queue);
                        CHECK(err);
                        err = clReleaseKernel(kernel_init);
                        CHECK(err);
                    }

                    {
                        // execute the kernel
                        cl_kernel kernel_map = clCreateKernel(program, "map", &err);
                        CHECK(err);
                        err = clSetKernelArg(kernel_map, 0, sizeof(cl_mem), (void *) &da);
                        CHECK(err);
                        err = clSetKernelArg(kernel_map, 1, sizeof(int), (void *)&n);
                        CHECK(err);
                        cl_command_queue cmd_queue = clCreateCommandQueue(context, device, 0, &err);
                        CHECK(err);
                        cl_event my_event;
                        err = clEnqueueNDRangeKernel(cmd_queue, kernel_map, max_dimensionality, NULL, tiles, tile, 0, NULL, &my_event);
                        CHECK(err);
                        err = clWaitForEvents(1, &my_event);
                        CHECK(err);
                        err = clReleaseCommandQueue(cmd_queue);
                        CHECK(err);
                        err = clReleaseKernel(kernel_map);
                        CHECK(err);
                    }
                }

                // Reduce.
                int levels = flog2(np1);
                for (int level = 0; level < levels; ++level)
                {
                    int step = pow2(level+1);
                    int threads = np1 / step;

                    size_t tile[3];
                    size_t tiles[3];
                    int max_dimensionality = 3;

                    l2t(threads, max_dimensionality, tile, tiles);

                    // execute the kernel
                    cl_kernel kernel_reduce = clCreateKernel(program, "kernel_reduce_a", &err);
                    CHECK(err);
                    err = clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), (void *) &da);
                    CHECK(err);
                    err = clSetKernelArg(kernel_reduce, 1, sizeof(int), (void *)&n);
                    CHECK(err);
                    err = clSetKernelArg(kernel_reduce, 2, sizeof(int), (void *)&step);
                    CHECK(err);
                    cl_command_queue cmd_queue = clCreateCommandQueue(context, device, 0, &err);
                    CHECK(err);
                    cl_event my_event;
                    err = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, max_dimensionality, NULL, tiles, tile, 0, NULL, &my_event);
                    CHECK(err);
                    err = clWaitForEvents(1, &my_event);
                    CHECK(err);
                    err = clReleaseCommandQueue(cmd_queue);
                    CHECK(err);
                    err = clReleaseKernel(kernel_reduce);
                    CHECK(err);
                }

                // read output array
                cl_command_queue cmd_queue = clCreateCommandQueue(context, device, 0, &err);
                CHECK(err);
                err = clEnqueueReadBuffer(cmd_queue, da, CL_TRUE, 0, sizeof(float), a, 0, NULL, NULL);
                CHECK(err);
                err = clReleaseCommandQueue(cmd_queue);
                CHECK(err);
                err = clReleaseMemObject(da);
                CHECK(err);

                float a1 = *a;
                float a2 = a1 * 4.0;
                float a3 = a2 / 3;
                float a4 = a3 / n;
                float w = ((*a * 4.0) / 3) / n;

                _ftime(&t2);
                std::cout << (double)(t2.time - t1.time + ((double)(t2.millitm - t1.millitm))/1000) << " s.\n";
                std::cout << n << " " << std::setprecision(44) << w;
                std::cout << "\n";

            }
            err = clReleaseContext(context);
            CHECK(err);
            err = clReleaseProgram(program);
            CHECK(err);
        }
    }
    catch(...)
    {
        std::cout << "Whoops!\n";
    }
}

int main(int argc, char *argv[])
{
    // input block size
    argc--; argv++;
    if (argc)
    {
        the_blocksize = atoi(*argv);
    }

//    int np1 = pow2(24);
    int np1 = pow2(24);
    int n = np1 - 1;

    printf("np1 = %d\n", np1);

    CpuPi(np1);

    cl_int err = NULL;
    cl_platform_id plat_id[20];
    cl_uint num;

    err = clGetPlatformIDs(20, plat_id, &num);

    if (err != NULL)
        return err;

    printf("Number of platforms = %d\n", num);

    cl_uint i = 0;
    for (i = 0; i < num; ++i)
    {
        char buf[500];
        size_t sz;
        err = clGetPlatformInfo(plat_id[i], CL_PLATFORM_PROFILE, 500, buf, &sz);
        if (err != NULL)
            return err;
        printf("Platform profile: %s\n", buf);
        err = clGetPlatformInfo(plat_id[i], CL_PLATFORM_VERSION, 500, buf, &sz);
        if (err != NULL)
            return err;
        printf("Platform version: %s\n", buf);
        err = clGetPlatformInfo(plat_id[i], CL_PLATFORM_NAME, 500, buf, &sz);
        if (err != NULL)
            return err;
        printf("Platform name: %s\n", buf);
        char vendor[500];
        err = clGetPlatformInfo(plat_id[i], CL_PLATFORM_VENDOR, 500, vendor, &sz);
        if (err != NULL)
            return err;
        printf("Platform vendor: %s\n", vendor);
        err = clGetPlatformInfo(plat_id[i], CL_PLATFORM_EXTENSIONS, 500, buf, &sz);
        if (err != NULL)
            return err;
        printf("Platform extensions: %s\n", buf);

        TryPlatform(plat_id[i], np1);
    }
    return 0;
}

