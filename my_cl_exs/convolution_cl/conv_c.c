/* Expected DEFINE's from the compilation */
/* kernsize -- the size of the filtering kernel in each direction */
/* nkernels -- the number of filters to use */
/* CONV_UNROLL -- amount of unrolling to perform */

/* Result is only defined in the area defined by [0 ... (imagesize - kernsize - CONV_UNROLL)]
   This means that the kernels are not centered around the input pixel but rather gives offsets the output image slightly.
   You need to shift it back manually afterwards if you care for this.
*/

/* Preprocessor settings to define types that can process multiple convolution kernels the same time.
 */
#if nkernels == 1
  typedef float kernf;
  typedef uchar kernuc;
#define kernstore(val,offset,arr) arr[offset]=val
#define convert_kernuc convert_uchar
#elif nkernels == 2
  typedef float2 kernf;
  typedef uchar2 kernuc;
#define kernstore vstore2
#define convert_kernuc convert_uchar2
#elif nkernels == 3
  typedef float3 kernf;
  typedef uchar3 kernuc;
#define kernstore vstore3
#define convert_kernuc convert_uchar3
#elif nkernels == 4
  typedef float4 kernf;
  typedef uchar4 kernuc;
#define kernstore vstore4
#define convert_kernuc convert_uchar4
#elif nkernels == 8
  typedef float8 kernf;
  typedef uchar8 kernuc;
#define kernstore vstore8
#define convert_kernuc convert_uchar8
#elif nkernels == 16
  typedef float16 kernf;
  typedef uchar16 kernuc;
#define kernstore vstore16
#define convert_kernuc convert_uchar16
#elif nkernels == 32
  typedef float32 kernf;
  typedef uchar32 kernuc;
#define kernstore vstore32
#define convert_kernuc convert_uchar32
#else
#error "nkernels should be one of: 1,2,3,4,8,16,32"
#endif


/* Use one of these three definitions to perform the multiply-add operation */
#define mmad(x,y,z) (x+y*z)       // Second most exact (or tied with fma), but fastest due to use of FMAC (FMA-accumulator) instruction
//#define mmad(x,y,z) mad(x,y,z)  // Undefined precision (for some cases this can be very very wrong)
//#define mmad(x,y,z) fma(x,y,z)  // Guaranteed to be the most exact

kernel void convolute(int4 imagesize, global unsigned char *input,
                      global unsigned char *output, global kernf *filterG) {
  int4 gid = (int4)(get_global_id(0)*CONV_UNROLL,  get_global_id(1),  get_global_id(2),  0);
  int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 0);
  int4 group = (int4)(get_group_id(0), get_group_id(1), get_group_id(2), 0);
  // First (?) pixel to process with this kernel
  int4 pixelid = gid;

  // Starting offset of the first pixel to process
  int imoffset = pixelid.s0 + imagesize.s0 * pixelid.s1 + imagesize.s0 * imagesize.s1 * pixelid.s2;
  int i,j;

  int dx,dy,dz;

  /* MAD performs a single convolution operation for each kernel,
     using the current 'raw' value as the input image
     'ko' as an instance of an unrolled convolution filter
     'pos' as the X-offset for each of the unrolled convolution filters
     Note that all the if statements dependent only on static values -
     meaning that they can be optimized away by the compiler
  */
#define MAD(ko,pos) {if(CONV_UNROLL>ko) {    \
      if(pos-ko >= 0 && pos-ko < kernsize) {    \
        val[ko] = mmad(val[ko],(kernf)(raw),filter[(pos-ko)+offset]);   \
      }}}
#define MADS(pos) {if(pos<kernsize) { \
    raw=input[imoffset2+pos];       \
    MAD(0,pos); MAD(1,pos); MAD(2,pos); MAD(3,pos); MAD(4,pos); MAD(5,pos); MAD(6,pos); MAD(7,pos); \
    MAD(8,pos); MAD(9,pos); MAD(10,pos); MAD(11,pos); MAD(12,pos); MAD(13,pos); MAD(14,pos); MAD(15,pos); \
    MAD(16,pos); MAD(17,pos); MAD(18,pos); MAD(19,pos); MAD(20,pos); MAD(21,pos); MAD(22,pos); MAD(23,pos); \
    MAD(24,pos); MAD(25,pos); MAD(26,pos); MAD(27,pos); MAD(28,pos); MAD(29,pos); MAD(30,pos); MAD(31,pos); \
    MAD(32,pos); MAD(33,pos); MAD(34,pos); MAD(35,pos); MAD(36,pos); MAD(37,pos); MAD(38,pos); MAD(39,pos); \
    }}

  kernf val[CONV_UNROLL];
  for(j=0;j<CONV_UNROLL;j++)
    val[j]=(kernf)(0.0);

  int localSize = get_local_size(0) * get_local_size(1) * get_local_size(2);
  local kernf filter[kernsize*kernsize*kernsize];

  /* Copy global filter to local memory */
  event_t event = async_work_group_copy(filter,filterG,kernsize*kernsize*kernsize,0);
  wait_group_events(1, &event);

  if(gid.s0 + kernsize + CONV_UNROLL > imagesize.s0 ||
     gid.s1 + kernsize > imagesize.s1 ||
     gid.s2 + kernsize > imagesize.s2) return;

  for(dz=0;dz<kernsize;dz++)
    for(dy=0;dy<kernsize;dy++)  {
      int offset = dy*kernsize*nkernels + dz*kernsize*kernsize*nkernels;
      int imoffset2 = imoffset+dy*imagesize.s0 + dz*imagesize.s0*imagesize.s1;
      unsigned char raw;

      /* kernsize + convolution_unroll < 42 */
      MADS(0); MADS(1); MADS(2); MADS(3); MADS(4); MADS(5);
      MADS(6); MADS(7); MADS(8); MADS(9); MADS(10); MADS(11);
      MADS(12); MADS(13); MADS(14); MADS(15); MADS(16); MADS(17);
      MADS(18); MADS(19); MADS(20); MADS(21); MADS(22); MADS(23);
      MADS(24); MADS(25); MADS(26); MADS(27); MADS(28); MADS(29);
      MADS(30); MADS(31); MADS(32); MADS(33); MADS(34); MADS(35);
      MADS(36); MADS(37); MADS(38); MADS(39); MADS(40); MADS(41);
    }

  for(j=0;j<CONV_UNROLL;j++) {
    kernstore( convert_kernuc(val[j]), imoffset+j, output);
  }
}
