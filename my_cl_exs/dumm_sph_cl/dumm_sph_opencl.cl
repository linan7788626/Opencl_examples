__kernel void sph_cl(                      
	__const __global float * g1_in,                  
	__const __global float * g2_in,                  
	__const __global float * x1_in,                  
	__const __global float * x2_in,                  
	__const __global float * SmoothLength,
	__global float * sdens_out,
	const float bsz,
	const int nc,
	const int np) {                                          

	int index = get_global_id(0);               
	int i;
	float R;
	float x;
	float sigma = 10.0f/(7.0f*3.141592653589793f);
	float res;
	res = 0.0f;
	float tmp;
	tmp = 0.0f;
	float hdsl;
	hdsl = 0.0f;

	for (i = 0;i<np;i++) {
		R = sqrt((x1_in[i]-g1_in[index])*(x1_in[i]-g1_in[index])+(x2_in[i]-g2_in[index])*(x2_in[i]-g2_in[index]));
		hdsl = SmoothLength[i];
		x = R/hdsl;

		if (x>2.0f) {
			continue;
		}
		else {
			if (x>=0.0f && x< 1.0f) {
				tmp = sigma*(1.0f-1.5f*x*x+0.75f*x*x*x);
			}
			if (x>=1.0f && x< 2.0f) {
				tmp = sigma*0.25f*(2.0f-x)*(2.0f-x)*(2.0f-x);
			}
		}
		res = res + tmp/(hdsl*hdsl); 
		res = res + 1.0f; 
	}
	sdens_out[index] = res;
}
