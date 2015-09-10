__kernel void sph_cl(                      
	__global float * sdens_out,
	const int nc,
	const int np) {                                          

	int index = get_global_id(0);               
	int i;
	local float res;
	res = 0.0f;

	for (i = 0;i<np;i++) {
		res = res + 1.0f; 
	}
	sdens_out[index] = res;
}
