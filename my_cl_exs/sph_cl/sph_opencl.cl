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

inline float si_weight(float x) {
	float sigma = 10.0f/(7.0f*3.141592653589793f);
	float res = 0.0f;

	if (x>=0.0f && x< 1.0f) {
		res = sigma*(1.0f-1.5f*x*x+0.75f*x*x*x);
		return res;
	}
	if (x>=1.0f && x< 2.0f) {
		res = sigma*0.25f*(2.0f-x)*(2.0f-x)*(2.0f-x);
		return res;
	}
	return res;
}

__kernel void sph_cl(                      
	__const __global float* x1_in,                  
	__const __global float* x2_in,                  
	__const __global float* SmoothLength,
	__global float* sdens_out,
	const float bsz,
	const int nc,
	const int np) {                                          

	float dsx = bsz/(float)(nc);
	float R;
	float sd_tot;

	int i_u,i_l,j_u,j_l,nbx,nby;
	int i,j,loc_i,loc_j;
	int Ncr = 32;
	float x_p,y_p,hdsl;

	int m = get_global_id(0);               

	x_p = x1_in[m]+0.5f*bsz;
	y_p = x2_in[m]+0.5f*bsz;
	hdsl = SmoothLength[m];

	if((fabs(x1_in[m]) > 0.5f*bsz) || (fabs(x2_in[m])> 0.5f*bsz)) return;
	
	i_u = (int)((x_p+2.0f*hdsl)/dsx);
	i_l = (int)((x_p-2.0f*hdsl)/dsx);
	nbx = i_u-i_l+1;

	j_u = (int)((y_p+2.0f*hdsl)/dsx);
	j_l = (int)((y_p-2.0f*hdsl)/dsx);
	nby = j_u-j_l+1;


	if (nbx <=2 && nby <=2) {
		//sdens_out[i_l*nc+j_l] += 1.0f/(dsx*dsx);
        AtomicAdd(&sdens_out[i_l*nc+j_l], 1.0f/(dsx*dsx));
		return;
	}

	else {
		if ((nbx+nby)/2 <= Ncr) {
			sd_tot = 0.0f;
			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
				sd_tot += si_weight(R/hdsl)/(hdsl*hdsl)*dsx*dsx;
			}

			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				if((loc_i>=nc)||(loc_i<0)||(loc_j>=nc)||(loc_j<0)) continue;

				R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
				//sdens_out[loc_i*nc+loc_j] += si_weight(R/hdsl)/(hdsl*hdsl)/sd_tot;
				AtomicAdd(&sdens_out[loc_i*nc+loc_j], si_weight(R/hdsl)/(hdsl*hdsl)/sd_tot);
			}
			return;
		}

		if ((nbx+nby)/2 > Ncr) {
			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				if((loc_i>=nc)||(loc_i<0)||(loc_j>=nc)||(loc_j<0)) continue;

				R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
				//sdens_out[loc_i*nc+loc_j] += si_weight(R/hdsl)/(hdsl*hdsl);
				AtomicAdd(&sdens_out[loc_i*nc+loc_j], si_weight(R/hdsl)/(hdsl*hdsl));
			}
			return;
		}
		return;
	}

	//for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
	//	loc_i = i_l + i;
	//	loc_j = j_l + j;
	//	if((loc_i>=(nc-1))||(loc_i<0)||(loc_j>=(nc-1))||(loc_j<0)) continue;

	//	R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
	//	//barrier(CLK_GLOBAL_MEM_FENCE);
	//	//sdens_out[loc_i*nc+loc_j] += si_weight(R/hdsl)/(hdsl*hdsl);
	//	AtomicAdd(&sdens_out[loc_i*nc+loc_j], si_weight(R/hdsl)/(hdsl*hdsl));
	//}
	//return;
}
