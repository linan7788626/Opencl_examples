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
	__global float* x1_in,                  
	__global float* x2_in,                  
	__global float* SmoothLength,
	__global float* sdens_out,
	const float bsz,
	const int Nc,
	const int Np) {                                          

	float dsx = bsz/(float)(Nc);
	float R;
	float sd_tot;

	int i_u,i_l,j_u,j_l,nbx,nby;
	int i,j,loc_i,loc_j;
	int Ncr = 32;
	float x_p,y_p,hdsl;

	int m = get_global_id(0);               
	if((fabs(x1_in[m]) > 0.5f*bsz) || (fabs(x2_in[m])> 0.5f*bsz)) return;

	barrier(CLK_LOCAL_MEM_FENCE);
	x_p = x1_in[m]+0.5f*bsz;
	y_p = x2_in[m]+0.5f*bsz;
	hdsl = SmoothLength[m];

	i_u = (int)((x_p+2.0f*hdsl)/dsx);
	i_l = (int)((x_p-2.0f*hdsl)/dsx);
	nbx = i_u-i_l+1;

	j_u = (int)((y_p+2.0f*hdsl)/dsx);
	j_l = (int)((y_p-2.0f*hdsl)/dsx);
	nby = j_u-j_l+1;

	//if (nbx <=2 && nby <=2) {
	//	sdens_out[i_l*Nc+j_l] += 1.0f/(dsx*dsx);
	//	return;
	//}
	//else {
	//	//if ((nbx+nby)/2 <= Ncr) {
	//	//	sd_tot = 0.0f;
	//	//	for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
	//	//		loc_i = i_l + i;
	//	//		loc_j = j_l + j;
	//	//		R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
	//	//		sd_tot += si_weight(R/hdsl)/(hdsl*hdsl)*dsx*dsx;
	//	//	}

	//	//	for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
	//	//		loc_i = i_l + i;
	//	//		loc_j = j_l + j;
	//	//		if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

	//	//		R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
	//	//		sdens_out[loc_i*Nc+loc_j] += si_weight(R/hdsl)/(hdsl*hdsl)/sd_tot;
	//	//	}
	//	//	return;
	//	//}

	//	//if ((nbx+nby)/2 > Ncr) {
	//	//	for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
	//	//		loc_i = i_l + i;
	//	//		loc_j = j_l + j;
	//	//		if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

	//	//		R=sqrt(pow((loc_i+0.5f)*dsx-x_p,2)+pow((loc_j+0.5f)*dsx-y_p,2));
	//	//		sdens_out[loc_i*Nc+loc_j] += si_weight(R/hdsl)/(hdsl*hdsl);
	//	//	}
	//	//	return;
	//	//}
	//	return;
	//}
	return;
}
