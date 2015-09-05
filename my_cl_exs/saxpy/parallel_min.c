//
// parallel_min.c
//
// A simple OpenCL program for computing the minimal value of an integer array of 2^25 elements
// using either scalar integers or 4-vectors (for PS3, choose a smaller size). Inspired by an
// example in ATI Stream Computing OpenCL Programming Guide available at
//
// http://developer.amd.com/gpu_assets/ATI_Stream_SDK_OpenCL_Programming_Guide.pdf
//
// Using events with nonblocking memory copies.
//
// Program launches two kernels, "pmin" and "reduce", "reduce" must not start before
// "pmin" has finished. This is synchronized uisng OpenCL event objects.
//
// The program launches the kernels subsequently in a loop (running by default for 500 iterations).
// By default, also the loop iterations are sycnhronized (i.e. the kernel "pmin" of iteration i+1 must
// not start before "reduce" from iteration i has finihed). This can be changed by a command line argument,
// therefore allowing the device to schedule the execution of the kernel pairs.
//
// Tested on iMac (iMac11,1) Core i7 @ 2.8GHz, 8GB, ATI HD 4850, OS X 10.6.3               (bugs in drivers for HD 4850?)
//           iMac (iMac9,1) Core 2 Duo @ 2.93GHz, 4GB, ATI HD 4850, OS X 10.6.2            (bugs in drivers for HD 4850?)
//           MacBook Pro (MacBookPro5,5) C2D @ 2.26GHz, NVIDIA GeForce 9400M, OS X 10.6.3       (works ok)
//           MacBook Pro (MacBookPro6,2) Core i7 @ 2.66GHz, NVIDIA GeForce GT 330M, OS X 10.6.3 (works ok)
//           PS3, IBM OpenCL Development Kit for Linux on Power, Fedora 12 (works ok, but very slow)
//           Linux pc with NVidia GeForce GTX 285                          (works ok)
//
// Tested using all devices available at given platform.
//
// Usage ./parallel_min dev vecw ws debg indep_iter
//
// dev:			0 = CL_DEVICE_CPU, 1 = CL_DEVICE_GPU || CL_DEVICE_ACCELERATOR, default: dev = 1
// vecw:		1 = use uint scalar kernel, 4 = use uint4 vector kernel, default: choose by selected dev
// ws:			n = warp/wavefront size, default: n = 64
// debg:		if exists, comment off debug sections from kernel source (effect minimal)
// indp_it:	if exists, allow different loop iterations to run without synchronization (effect minimal)
//
// Test results (100 test runs with random data, seed srandom((unsigned int)getpid()))
//
//	SYNCHRONIZED LOOP ITERATIONS
//
//	Platfm  Device	succes %	failure %	Mem BW (ws 64)
//
//	iMaci7	HD4850	69		31		~35  GB/s
//	iMaci7	i7	100		0		~13  GB/s
//	iMacC2D	HD4850	68		32		~37  GB/s
//	iMacC2D	C2D	100		0		~3.7 GB/s
//	MBP	9400M	100		0		~4.3 GB/s
//	MBP	C2D	100		0		~3.8 GB/s
//	MBP	GT330M	100		0		~10  GB/s
//	MBP	i7	100		0		~7.2 GB/s
//	PC	GTX285	100		0		~62  GB/s      ~74.5 (ws 128)
//      PS3	6 SPEs	100		0		<0.5 GB/s
//      PS3	PPE	100		0		<0.1 GB/s
//
// Tuomo Rossi, spring 2010
////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define REPSTRL (1024)

// Use a static data size for simplicity (this is ok for GTX285 and HD4850)

#define DATA_SIZE (2 * 4096 * 4096)

// Should run on PS3 with this data size
//#define DATA_SIZE (2048 * 4096)

#define NUM_LOOPS (500)

#define SEPARATOR       ("----------------------------------------------------------------------\n")

// A structure collecting device information.

typedef struct {
  cl_device_type device_type;
  char vendor_name[REPSTRL];
  char device_name[REPSTRL];
  char device_version[REPSTRL];
  char device_extens[REPSTRL];
  cl_uint max_compute_units;
  cl_ulong global_mem_size;
  size_t max_workgroup_size;
  cl_device_local_mem_type local_mem_type;
  cl_ulong local_mem_size;
  cl_uint vec_width_char;
  cl_uint vec_width_short;
  cl_uint vec_width_int;
  cl_uint vec_width_long;
  cl_uint vec_width_float;
  cl_uint vec_width_double;
} Device_Report;

// A parallel min() kernel for finding the minimum value of an integer array. Accessing
// array as uint4 vectors, like in ATI Guide, but without atom_min.

// 0         1         2         3         4         5         6         7         8
// 01234567890123456789012345678901234567890123456789012345678901234567890123456789012345

char *PMin4KernelSource = 
  "                                                                                     \n" \
  "                                                                                     \n" \
  "__kernel void pmin( __global uint4 *src,                                             \n" \
  "                    __global uint  *gmin,                                            \n" \
  "                    __local  uint  *lmin,                                            \n" \
  "                    __global uint  *dbg,                                             \n" \
  "                    uint           nitems,                                           \n" \
  "                    uint           dev)                                              \n" \
  "{                                                                                    \n" \
  "   uint count  = (nitems/4) / get_global_size(0);                                    \n" \
  "   uint idx    = (dev == 0) ? get_global_id(0) * count                               \n" \
  "                            : get_global_id(0);                                      \n" \
  "   uint stride = (dev == 0) ? 1 : get_global_size(0);                                \n" \
  "                                                                                     \n" \
  "   // Private min for the values which this work-item accesses                       \n" \
  "                                                                                     \n" \
  "   uint pmin   = (uint) -1;                                                          \n" \
  "                                                                                     \n" \
  "   for (int n = 0; n < count; n++, idx += stride) {                                  \n" \
  "     pmin = min(pmin,src[idx].x);                                                    \n" \
  "     pmin = min(pmin,src[idx].y);                                                    \n" \
  "     pmin = min(pmin,src[idx].z);                                                    \n" \
  "     pmin = min(pmin,src[idx].w);                                                    \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   // Reduce values within the work-group into local memory                          \n" \
  "                                                                                     \n" \
  "   barrier(CLK_LOCAL_MEM_FENCE);                                                     \n" \
  "   if (get_local_id(0) == 0) lmin[0] = (uint) -1;                                    \n" \
  "                                                                                     \n" \
  "   for (int n = 0; n < get_local_size(0); n++) {                                     \n" \
  "     barrier(CLK_LOCAL_MEM_FENCE);                                                   \n" \
  "     if (get_local_id(0) == n) lmin[0] = min(lmin[0],pmin);                          \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   barrier(CLK_LOCAL_MEM_FENCE);                                                     \n" \
  "                                                                                     \n" \
  "   // Write to __global gmin which will contain the work-group minima                \n" \
  "                                                                                     \n" \
  "   if (get_local_id(0) == 0) gmin[get_group_id(0)] = lmin[0];                        \n" \
  "                                                                                     \n" \
  "   // Collect debug information                                                      \n" \
  "                                                                                     \n" \
  " %s if (get_global_id(0) == 0) {                                                     \n" \
  " %s   dbg[0] = get_num_groups(0);                                                    \n" \
  " %s   dbg[1] = get_global_size(0);                                                   \n" \
  " %s   dbg[2] = count;                                                                \n" \
  " %s   dbg[3] = stride;                                                               \n" \
  " %s }                                                                                \n" \
  "}                                                                                    \n" \
  "                                                                                     \n" \
  "   // Reduce work-group minima                                                       \n" \
  "                                                                                     \n" \
  "__kernel void reduce( __global uint  *gmin)                                          \n" \
  "{                                                                                    \n" \
  "   for (int n = 0; n < get_global_size(0); n++) {                                    \n" \
  "     barrier(CLK_GLOBAL_MEM_FENCE);                                                  \n" \
  "     if (get_global_id(0) == n) gmin[0] = min(gmin[0],gmin[n]);                      \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   barrier(CLK_GLOBAL_MEM_FENCE);                                                    \n" \
  "                                                                                     \n" \
  "}                                                                                    \n" \
  "\n";

// A parallel min() kernel for finding the minimum value of an integer array. Accessing
// array as uint scalars, like in ATI Guide, but without atom_min.

// 0         1         2         3         4         5         6         7         8
// 01234567890123456789012345678901234567890123456789012345678901234567890123456789012345

char *PMin1KernelSource = 
  "                                                                                     \n" \
  "                                                                                     \n" \
  "__kernel void pmin( __global uint  *src,                                             \n" \
  "                    __global uint  *gmin,                                            \n" \
  "                    __local  uint  *lmin,                                            \n" \
  "                    __global uint  *dbg,                                             \n" \
  "                    uint           nitems,                                           \n" \
  "                    uint           dev)                                              \n" \
  "{                                                                                    \n" \
  "   uint count  = nitems     / get_global_size(0);                                    \n" \
  "   uint idx    = (dev == 0) ? get_global_id(0) * count                               \n" \
  "                            : get_global_id(0);                                      \n" \
  "   uint stride = (dev == 0) ? 1 : get_global_size(0);                                \n" \
  "                                                                                     \n" \
  "   // Private min for the work-item                                                  \n" \
  "                                                                                     \n" \
  "   uint pmin   = (uint) -1;                                                          \n" \
  "                                                                                     \n" \
  "   for (int n = 0; n < count; n++, idx += stride) {                                  \n" \
  "     pmin = min(pmin,src[idx]);                                                      \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   // Reduce values within the work-group into local memory                          \n" \
  "                                                                                     \n" \
  "   barrier(CLK_LOCAL_MEM_FENCE);                                                     \n" \
  "   if (get_local_id(0) == 0) lmin[0] = (uint) -1;                                    \n" \
  "                                                                                     \n" \
  "   for (int n = 0; n < get_local_size(0); n++) {                                     \n" \
  "     barrier(CLK_LOCAL_MEM_FENCE);                                                   \n" \
  "     if (get_local_id(0) == n) lmin[0] = min(lmin[0],pmin);                          \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   barrier(CLK_LOCAL_MEM_FENCE);                                                     \n" \
  "                                                                                     \n" \
  "   // Write to __global gmin which will contain the work-group minima                \n" \
  "                                                                                     \n" \
  "   if (get_local_id(0) == 0) gmin[get_group_id(0)] = lmin[0];                        \n" \
  "                                                                                     \n" \
  "   // Collect debug information                                                      \n" \
  "                                                                                     \n" \
  " %s if (get_global_id(0) == 0) {                                                     \n" \
  " %s   dbg[0] = get_num_groups(0);                                                    \n" \
  " %s   dbg[1] = get_global_size(0);                                                   \n" \
  " %s   dbg[2] = count;                                                                \n" \
  " %s   dbg[3] = stride;                                                               \n" \
  " %s }                                                                                \n" \
  "}                                                                                    \n" \
  "                                                                                     \n" \
  "   // Reduce work-group minima                                                       \n" \
  "                                                                                     \n" \
  "__kernel void reduce( __global uint  *gmin)                                          \n" \
  "{                                                                                    \n" \
  "   for (int n = 0; n < get_global_size(0); n++) {                                    \n" \
  "     barrier(CLK_GLOBAL_MEM_FENCE);                                                  \n" \
  "     if (get_global_id(0) == n) gmin[0] = min(gmin[0],gmin[n]);                      \n" \
  "   }                                                                                 \n" \
  "                                                                                     \n" \
  "   barrier(CLK_GLOBAL_MEM_FENCE);                                                    \n" \
  "                                                                                     \n" \
  "}                                                                                    \n" \
  "\n";

// Prototype for device reporting function

cl_int report_and_mark_devices(cl_device_id *, cl_uint, int *, int *, int *, Device_Report *);

int main(int argc, char** argv)
{
  cl_int        err;                                          
  cl_device_id  device_id;                                    
  int           gpu              = 1;                         // use gpu by default
  int           pref_vw          = 1;                         // default vector width for type uint is 1 (like in GTX285)
  char         *PMinKernelSource = (char *)PMin1KernelSource; // default kernel uses therefore uint
  
  if (argc == 1) {
    printf("%s -trying to use CL_DEVICE_GPU\n", argv[0]);
  } else if (argc >= 2) {
    gpu = atoi(argv[1]);
    if (gpu != 0 && gpu != 1) {
      printf("Usage: %s 0 -use CL_DEVICE_CPU\n", argv[0]);
      printf("Usage: %s 1 -use CL_DEVICE_GPU or CL_DEVICE_ACCELERATOR\n", argv[0]); 
      return EXIT_FAILURE;
    }
    printf("%s -trying to use %s\n", argv[0], gpu ? "CL_DEVICE_GPU or CL_DEVICE_ACCELERATOR" :
	   "CL_DEVICE_CPU");
  }
  
  // Allocate space and fill our data set with "random" data
  
  int i = 0;
  unsigned int num_src_items = DATA_SIZE;
  cl_uint      *src_ptr;                                      // pointer to the source uint array
  cl_uint min = (cl_uint) -1;                                 // two's complement, all bits are 1
  
  src_ptr = (cl_uint *)malloc(num_src_items * sizeof(cl_uint));
  
  srandom((unsigned int)getpid());                            // initialize with current process ID
  
  for(i = 0; i < num_src_items; i++) {
    src_ptr[i] = (cl_uint) random();
    min = src_ptr[i] < min ? src_ptr[i] : min;
  }
  
  // Trying to identify one platform:
  
  cl_platform_id platform;
  cl_uint num_platforms;
  
  err = clGetPlatformIDs(1,&platform,&num_platforms);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get a platform id!\n");
    return EXIT_FAILURE;
  }
  
  // Trying to query platform specific information...
  
  size_t  returned_size = 0;
  cl_char platform_name[1024] = {0};
  cl_char platform_prof[1024] = {0};
  cl_char platform_vers[1024] = {0};
  cl_char platform_exts[1024] = {0};
  
  err  = clGetPlatformInfo(platform, CL_PLATFORM_NAME,       sizeof(platform_name), platform_name, &returned_size);
  err |= clGetPlatformInfo(platform, CL_PLATFORM_VERSION,    sizeof(platform_vers), platform_vers, &returned_size);
  err |= clGetPlatformInfo(platform, CL_PLATFORM_PROFILE,    sizeof(platform_prof), platform_prof, &returned_size);
  err |= clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(platform_exts), platform_exts, &returned_size);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get platform infor!\n");
    return EXIT_FAILURE;
  }
  
  printf("\nPlatform information\n");
  printf(SEPARATOR);
  printf("Platform name:       %s\n", (char *)platform_name);
  printf("Platform version:    %s\n", (char *)platform_vers);
  printf("Platform profile:    %s\n", (char *)platform_prof);
  printf("Platform extensions: %s\n", ((char)platform_exts[0] != '\0') ? (char *)platform_exts : "NONE");
  
  // Get all available devices (up to 4)
  
  cl_uint num_devices;
  cl_device_id devices[4];
  
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 4, devices, &num_devices);
  
  if (err != CL_SUCCESS) {
    printf("Failed to collect device list on this platform!\n");
    return EXIT_FAILURE;
  }
  
  printf(SEPARATOR);
  printf("\nFound %d compute devices!:\n", num_devices);
  
  // Collect and report device information, return indices of devices of type CL_DEVICE_TYPE_CPU,
  // CL_DEVICE_TYPE_GPU and CL_DEVICE_TYPE_ACCELERATOR in arrays devices and device_reports, and
  // specific information about devices in device_reports array.
  
  int a_cpu = -1, a_gpu = -1, an_accelerator = -1;
  Device_Report device_reports[4];
  
  err = report_and_mark_devices(devices, num_devices, &a_cpu, &a_gpu, &an_accelerator, device_reports);
  
  if (err != CL_SUCCESS) {
    printf("Failed to report information about the devices on this platform!\n");
    return EXIT_FAILURE;
  }
  
  // Checking for availability of the required device
  
  int the_device = -1;
  
  if (gpu == 0) {      // No accelerator or gpu, just cpu
    if (a_cpu == -1) {
      printf("No cpus available, have to quit\n");
      return EXIT_FAILURE;
    }
    the_device = a_cpu;
    device_id  = devices[the_device];
    pref_vw    = device_reports[the_device].vec_width_int;
    printf("There is a cpu, using it with preferred uint%d kernel\n", pref_vw);
  }
  else {               // Trying to find a gpu, or if that fails, an accelerator 
    if (a_gpu != -1) { 
      the_device = a_gpu;
      device_id  = devices[the_device];
      pref_vw    = device_reports[the_device].vec_width_int;
      printf("There is a gpu, using it with preferred uint%d kernel\n", pref_vw);
    } else if (an_accelerator != -1) {
      the_device = an_accelerator;
      device_id  = devices[the_device];
      pref_vw    = device_reports[the_device].vec_width_int;
      printf("No gpu but found an accelerator, using it with uint%d kernel\n", pref_vw);
    } else {
      printf("No gpu, nor an accelerator, have to quit\n");
      return EXIT_FAILURE;
    }
  }
  
  // If user wants, use another kernel
  
  if(argc >= 3) {
    pref_vw = atoi(argv[2]);
    if (pref_vw == 1 || pref_vw == 4) {	
      printf(SEPARATOR);	
      printf("\nUser wants to use uint%d kernel, overwriting automatically selected!\n\n", pref_vw);
      printf(SEPARATOR);		
    } else {
      printf("User want's an unsupported vector width %d, switching to scalar kernel\n", pref_vw);
      printf(SEPARATOR);
      pref_vw = 1;	
    }
  }
  
  size_t global_work_size;
  size_t local_work_size;
  size_t num_groups;
  cl_uint dev = 0;
  
  if (device_reports[the_device].device_type == CL_DEVICE_TYPE_GPU) {
    dev = 1;                                                                  // Kernel chooses mem. access tuned for GPU
    cl_uint ws = 64;                                                          // Optimal wavefront size according to ATI
    if (argc >= 4) {
      ws = (cl_uint) atoi(argv[3]);                                           // Change it, if user wants...
    }
    global_work_size = device_reports[the_device].max_compute_units * 7 * ws; // Seven warps/wavefronts per compute unit
    while ((num_src_items/pref_vw) % global_work_size != 0) {
      global_work_size += ws;
    }
    local_work_size = ws;
  } else {
    dev = 0;                                                                  // Kernel chooses mem. access tuned for CPU
    global_work_size = device_reports[the_device].max_compute_units * 1;      // One thread per cpu core
    local_work_size  = 1;
  }
  
  num_groups = global_work_size / local_work_size;
  
  printf(SEPARATOR);
  printf("\nUsing device %s\n", device_reports[the_device].device_name);
  printf("Computing minimum of %d elements of type uint%d\n\n", num_src_items/pref_vw, pref_vw);
  
  printf(SEPARATOR);
  printf("EXECUTION INFO: KERNEL \"pmin\"\n");
  printf(SEPARATOR);
  
  printf("\nGlobal work size is %d work-items\n", (int)global_work_size);
  printf("\nThe %d work-items are decomposed into %d groups each containing %d work items.\n",
	 (int)global_work_size, (int)num_groups, (int)local_work_size);
  printf("\nWithin each %d groups, its %d work items first compute the private minimum of %d elements of type uint%d.\n",
	 (int)num_groups, (int)local_work_size, (int)(num_src_items/pref_vw/global_work_size), pref_vw);
  printf("\nAt this stage, each work item accesses global memory with the stride of %d uint%d elements.\n",
	 device_reports[the_device].device_type == CL_DEVICE_TYPE_GPU ? (int)global_work_size : 1, pref_vw);
  
  if (device_reports[the_device].device_type == CL_DEVICE_TYPE_GPU) {
    printf("This will lead to coalesced memory access of parallel threads in GPU.\n");
  } else {
    printf("This will lead to good cache usage in CPU.\n");
  }
  
  printf("\nThe groupwise %d private minima are then reduced into work-group wide local minimun in __local element lmin[0].\n",
	 (int)local_work_size);
  printf("\nThe local minima of %d work-groups are collected into __global array gmin.\n" \
	 "This is the output array of kernel pmin.\n\n", (int)num_groups);
  
  printf(SEPARATOR);
  printf("EXECUTION INFO: KERNEL \"reduce\"\n");
  printf(SEPARATOR);
  
  printf("\nGlobal work size is %d work-items, input to kernel reduce is __global array gmin\n" \
	 "with %d elements of type uint.", (int)num_groups, (int)num_groups);
  printf("\n%d work-items are decomposed into work-groups automatically by the system.\n" \
	 "Work-group size is unknown to us.\n", (int)num_groups);
  printf("\n%d values in gmin are reduced to the element gmin[0].\n", (int)num_groups);
  printf("gmin[0] is the output of kernel reduce, and it will then contain the global minimum of the original array.\n");
  
  // If user wants, comment off the debug info reporting section in kernel source for checking
  // the effect to performance (seems that the effect is minimal).
  
  int debug = 1;
  char debug_str[3] = "  ";
  
  if(argc >= 5) {
    debug = 0;
    debug_str[0] = debug_str[1] = '/';
  }
  
  if (pref_vw == 1) {
    PMinKernelSource = (char *)malloc((strlen(PMin1KernelSource) + 1) * sizeof(char));
    sprintf(PMinKernelSource, PMin1KernelSource, debug_str, debug_str, debug_str, debug_str, debug_str, debug_str);
  }
  
  if (pref_vw == 4) {
    PMinKernelSource = (char *)malloc((strlen(PMin4KernelSource) + 1) * sizeof(char));
    sprintf(PMinKernelSource, PMin4KernelSource, debug_str, debug_str, debug_str, debug_str, debug_str, debug_str);
  }
  
  // We have a compute device of required type! Next, create a compute context on it.
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nCreating a compute context for the required device\n");
  
  cl_context context;                 
  
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  
  if (!context) {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }
  
  // Creating a command queue for the selected device within context
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nCreating a command queue\n");
  
  cl_command_queue commands;
  
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  
  if (!commands) {
    printf("Error: Failed to create a command queue!\n");
    return EXIT_FAILURE;
  }
  
  // Create the compute program object for our context and load the source code from the source buffer
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nCreating the compute program from source\n");
  printf("%s\n", PMinKernelSource);
  
  cl_program program;                

  program = clCreateProgramWithSource(context, 1, (const char **) &PMinKernelSource, NULL, &err);
  
  if (!program) {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }
  
  // Build the program executable
  
  printf(SEPARATOR);
  printf("\nCompiling the program executable\n");
  
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n");
    
    // See page 98...
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }
  
  // Create the compute kernel objects in the program we wish to run
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nCreating the compute kernels from program executable\n");
  
  cl_kernel pmin, reduce;                   

  pmin   = clCreateKernel(program, "pmin", &err);
  reduce = clCreateKernel(program, "reduce", &err);
  
  if (!pmin || !reduce) {
    printf("Error: Failed to create compute kernels!\n");
    exit(1);
  }
  
  // Create buffer objects for the input, output and debug buffers in device memory
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nCreating the input, output and debug buffers in device memory\n");
  printf("Allocating %d bytes for src_buf, %d bytes for dst_buf and %d bytes for dbg_buf. In total %d bytes.\n",
	 (int)(sizeof(cl_uint) * num_src_items), (int)(sizeof(cl_uint) * num_groups), (int)(sizeof(cl_uint) * global_work_size),
	 (int)(sizeof(cl_uint) * num_src_items + sizeof(cl_uint) * num_groups + sizeof(cl_uint) * global_work_size));
  
  printf("\n");
  printf(SEPARATOR);
  printf("\nLaunching kernels pmin and reduce %d times, synchronized loop iterations: %s\n", NUM_LOOPS,
	 argc >= 6 ? "NO" : "YES");
  
  cl_mem src_buf, dst_buf, dbg_buf;                       // device memory used for the input/output array y
  struct timeval initial_time, final_time;
  
  gettimeofday(&initial_time,NULL);	                      // take a time stamp
  
  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_uint) * num_src_items,    NULL, NULL);
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_groups,       NULL, NULL);
  dbg_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * global_work_size, NULL, NULL);
  
  if (!src_buf || !dst_buf || !dbg_buf) {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }    
  
  // Setting kernel args
  //
  // Kernel pmin
  
  err  = clSetKernelArg(pmin,   0, sizeof(void *),       (void *)&src_buf);
  err |= clSetKernelArg(pmin,   1, sizeof(void *),       (void *)&dst_buf);
  err |= clSetKernelArg(pmin,   2, sizeof(cl_uint),      (void *)NULL);
  err |= clSetKernelArg(pmin,   3, sizeof(void *),       (void *)&dbg_buf);
  err |= clSetKernelArg(pmin,   4, sizeof(unsigned int), (void *)&num_src_items);
  err |= clSetKernelArg(pmin,   5, sizeof(cl_uint),      (void *)&dev);
  
  // Kernel reduce
  
  err |= clSetKernelArg(reduce, 0, sizeof(void *),       (void *)&dst_buf);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    exit(1);
  }
  
  // Allocating memory for results in the  host memory.
  
  cl_uint *dst_ptr, *dbg_ptr;
  
  dst_ptr = (cl_uint *) malloc(num_groups*sizeof(cl_uint));
  dbg_ptr = (cl_uint *) malloc(global_work_size*sizeof(cl_uint));
  
  if (!dst_ptr || !dbg_ptr) {
    printf("Error: Failed to alloccate memory for buffers in the host address space!\n");
    exit(1);
  }
  
  // Nonblocking write to device memory, monitored by an event
  
  cl_event ev_memcpy;
  
  err = clEnqueueWriteBuffer(commands, src_buf, CL_FALSE, 0, sizeof(cl_uint) * num_src_items, src_ptr, 0, NULL, &ev_memcpy);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    exit(1);
  }
	
  // Launching two kernels successively NUM_LOOPS times, syncronizing using events, first waiting write to finish
  
  cl_event ev_pmin, ev_reduce;
  int nloops = NUM_LOOPS - 1;
  
  clEnqueueNDRangeKernel(commands, pmin, 1, NULL, &global_work_size, &local_work_size, 1, &ev_memcpy, &ev_pmin);
  
  if (argc >= 6) {                     // No synchronization between different loop iterations
    while (nloops--) {
      clEnqueueNDRangeKernel(commands, reduce, 1, NULL, &num_groups,       NULL,             1, &ev_pmin, NULL);
      clEnqueueNDRangeKernel(commands, pmin,   1, NULL, &global_work_size, &local_work_size, 0, NULL,     &ev_pmin);
    }
  } else {
    while (nloops--) {                 // Synchronization also between different loop iterations
      clEnqueueNDRangeKernel(commands, reduce, 1, NULL, &num_groups,       NULL,             1, &ev_pmin,   &ev_reduce);
      clEnqueueNDRangeKernel(commands, pmin,   1, NULL, &global_work_size, &local_work_size, 1, &ev_reduce, &ev_pmin);
    }	
  }
  
  clEnqueueNDRangeKernel(commands, reduce, 1, NULL, &num_groups, NULL, 1, &ev_pmin, &ev_reduce);
  
  err  = clEnqueueReadBuffer(commands, dst_buf, CL_FALSE, 0, num_groups*sizeof(cl_uint),       dst_ptr, 1, &ev_reduce, &ev_memcpy);
  err |= clEnqueueReadBuffer(commands, dbg_buf, CL_TRUE,  0, global_work_size*sizeof(cl_uint), dbg_ptr, 1, &ev_memcpy, NULL);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array! %d\n", err);
    exit(1);
  }
  
  err  = clFinish(commands);
  
  if (err != CL_SUCCESS) {
    printf("Error: Failed to finish execution of commands! %d\n", err);
    exit(1);
  }
  
  // Take another time stamp, compute performance metrics and print.
  
  gettimeofday(&final_time, NULL);
  
  if (debug){
    printf("Used %d groups, %d threads with count %d and stride %d\n", dbg_ptr[0], dbg_ptr[1], dbg_ptr[2], dbg_ptr[3]);
  }
  
  long long exec_time = ((long long)final_time.tv_sec * 1000000 + final_time.tv_usec) -
    ((long long)initial_time.tv_sec * 1000000 + initial_time.tv_usec);
  
  printf("\nExecution time was %llu microseconds\n", exec_time);
  
  float bandwidth = 1e-9*(float)(num_src_items*sizeof(cl_uint)*(NUM_LOOPS)) /
    ((float)exec_time/1e6);	
  
  printf("Memory bandwidth %.2f GB/sec\n", bandwidth);
  
  // Verify result
  
  if (dst_ptr[0] == min) {
    printf("Result was correct :-) OpenCL: %d, cpu: %d\n", dst_ptr[0], min);
  } else {
    printf("Result was incorrect :-( OpenCL: %d, cpu: %d\n", dst_ptr[0], min);
  }
  
  
  // Shutdown and cleanup
  
  free(src_ptr);
  free(dst_ptr);
  free(dbg_ptr);
  free(PMinKernelSource);
  clReleaseMemObject(src_buf);
  clReleaseMemObject(dst_buf);
  clReleaseMemObject(dbg_buf);
  clReleaseProgram(program);
  clReleaseKernel(pmin);
  clReleaseKernel(reduce);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  
  return 0;
}

cl_int report_and_mark_devices(cl_device_id *devices, cl_uint num_devices, int *a_cpu, int *a_gpu, int *an_accelerator,
			       Device_Report *dev_rep)
{
  int i, type_name_index = 0;
  cl_int err = 0;
  size_t returned_size;
  char type_names[3][27]={"CL_DEVICE_TYPE_CPU        " , "CL_DEVICE_TYPE_GPU        " , "CL_DEVICE_TYPE_ACCELERATOR"};
  
  for (i=0;i<num_devices;i++) {
    err = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE,                          sizeof(cl_device_type),           &(dev_rep[i].device_type),        &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR,                        REPSTRL * sizeof(char),           dev_rep[i].vendor_name,           &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_NAME,                          REPSTRL * sizeof(char),           dev_rep[i].device_name,           &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_VERSION,                       REPSTRL * sizeof(char),           dev_rep[i].device_version,        &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS,                    REPSTRL * sizeof(char),           dev_rep[i].device_extens,         &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS,             sizeof(cl_uint),                  &(dev_rep[i].max_compute_units),  &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE,               sizeof(cl_ulong),                 &(dev_rep[i].global_mem_size),    &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,           sizeof(size_t),                   &(dev_rep[i].max_workgroup_size), &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_TYPE,                sizeof(cl_device_local_mem_type), &(dev_rep[i].local_mem_type),     &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE,                sizeof(cl_ulong),                 &(dev_rep[i].local_mem_size),     &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,   sizeof(cl_uint),                  &(dev_rep[i].vec_width_char),     &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,  sizeof(cl_uint),                  &(dev_rep[i].vec_width_short),    &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,    sizeof(cl_uint),                  &(dev_rep[i].vec_width_int),      &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,   sizeof(cl_uint),                  &(dev_rep[i].vec_width_long),     &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,  sizeof(cl_uint),                  &(dev_rep[i].vec_width_float),    &returned_size);
    err|= clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint),                  &(dev_rep[i].vec_width_double),   &returned_size);
    
    if (err != CL_SUCCESS) {
      printf("Error: Failed to retrieve device info!\n");
      return EXIT_FAILURE;
    }
    
    if (dev_rep[i].device_type == CL_DEVICE_TYPE_CPU) {
      *a_cpu = i;
      type_name_index = 0;
    }
    
    if (dev_rep[i].device_type == CL_DEVICE_TYPE_GPU) {
      *a_gpu = i;
      type_name_index = 1;
    }
    
    if (dev_rep[i].device_type == CL_DEVICE_TYPE_ACCELERATOR) {
      *an_accelerator = i;
      type_name_index = 2;
    }
    
    printf("\nDevice information:\n");
    printf(SEPARATOR);
    printf("Type:               %s\n",  type_names[type_name_index]);
    printf("Vendor:             %s\n",  dev_rep[i].vendor_name);
    printf("Device:             %s\n",  dev_rep[i].device_name);
    printf("Version:            %s\n",  dev_rep[i].device_version);
    printf("Extensions:         %s\n",  dev_rep[i].device_extens);
    printf("Max compute units:  %d\n",  (int)dev_rep[i].max_compute_units);
    printf("Max workgroup size: %d\n",  (int)dev_rep[i].max_workgroup_size);
    printf("Global mem size:    %ld\n", (long)dev_rep[i].global_mem_size);
    printf("Local mem size:     %ld\n", (long)dev_rep[i].local_mem_size);
    printf("Local mem type:     %s\n",  dev_rep[i].local_mem_type == CL_LOCAL ? "DEDICATED" : "GLOBAL");
    printf(SEPARATOR);
    
    printf("\nPreferred vector widths by type:\n");
    
    printf(SEPARATOR);
    printf("Vector char:  %d\n",   (int)dev_rep[i].vec_width_char);
    printf("Vector short: %d\n",   (int)dev_rep[i].vec_width_short);
    printf("Vector int:   %d\n",   (int)dev_rep[i].vec_width_int);
    printf("Vector long:  %d\n",   (int)dev_rep[i].vec_width_long);
    printf("Vector float: %d\n",   (int)dev_rep[i].vec_width_float);
    printf("Vector dble:  %d\n",   (int)dev_rep[i].vec_width_double);
    printf(SEPARATOR);
    printf("\n");
  }
  return err;
}
