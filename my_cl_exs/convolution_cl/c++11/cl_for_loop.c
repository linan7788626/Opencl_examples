#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <iterator>
#include <OpenCL/cl.hpp>
#include <OpenCL/opencl.h>

using namespace cl;

void cpu_3d_loop (int x, int y, int z) {

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                printf("CPU %d,%d,%dn", i, j, k);
            }
        }
    }

}

int main () {

    // CPU 3d loop

    int x = 4;
    int y = 3;
    int z = 2;
    cpu_3d_loop(x, y, z);
    std::cout << std::endl;

    // GPU 3d loop

    vector<Platform> platforms;
    vector<Device> devices;
    vector<Kernel> kernels;

    try {

        // create platform, context and command queue
        Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        Context context(devices);
        CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("kernels.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
            (std::istreambuf_iterator<char>()));
        Program::Sources source(1, std::make_pair(cl_string.c_str(),
            cl_string.length() + 1));

        // create program and kernel and set kernel arguments
        Program program(context, source);
        program.build(devices);
        Kernel kernel(program, "ndrange_parallelism");

        // execute kernel and wait for completion
        NDRange global_work_size(x, y, z);
        queue.enqueueNDRangeKernel(kernel, NullRange, global_work_size, NullRange);
        queue.finish();

    } catch (Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;

}
