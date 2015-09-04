inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
__kernel void wcic_cl(                      
	__const __global float* cic_in,                  
	__const __global float* x_in,                  
	__const __global float* y_in,
	__global float* cic_out,
	const float bsx,
	const float bsy,
	const int nx,
	const int ny,
	const int np)               
{                                          


	int index = get_global_id(0);               
    int ip,jp;
    float xb1,xb2;
    float dx = bsx/nx;
    float dy = bsy/ny;
    float xc = bsx/2.0f;
    float yc = bsy/2.0f;
    float wx,wy;
    float xp,yp,zp;

	xp = (x_in[index]+xc)/dx-0.5f;
	yp = (y_in[index]+yc)/dy-0.5f;
	zp = cic_in[index];
	
	ip = (int)xp;
	jp = (int)yp;
	
	wx = 1.0f-(xp-(float)ip);
	wy = 1.0f-(yp-(float)jp);

	if (ip<0||ip>(nx-2)||jp<0||jp>(ny-2)) return;

	cic_out[ip*ny+jp] += wx*wy*zp;
	cic_out[ip*ny+(jp+1)] += wx*(1.0f-wy)*zp;
	cic_out[(ip+1)*ny+jp] += (1.0f-wx)*wy*zp;
	cic_out[(ip+1)*ny+(jp+1)] += (1.0f-wx)*(1.0f-wy)*zp;
	
	//AtomicAdd(cic_out[ip*ny+jp],wx*wy*zp);
	//AtomicAdd(cic_out[ip*ny+(jp+1)],wx*(1.0f-wy)*zp);
	//AtomicAdd(cic_out[(ip+1)*ny+jp],(1.0f-wx)*wy*zp);
	//AtomicAdd(cic_out[(ip+1)*ny+(jp+1)],(1.0f-wx)*(1.0f-wy)*zp);
}
