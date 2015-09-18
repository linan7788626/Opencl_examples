__kernel void mmul(
	const int n1,
	const int n2,
	const int n3,
	__global float *A,
	__global float *B,
	__global float *C) {
	//__local float *Bwrk) {

	int k;
	int j;

	int i = get_global_id(0);
	int iloc = get_local_id(0);
	int nloc = get_local_size(0);
	float tmp;
	float Awrk[1024];
	__local float Bwrk[1024];
	for (k = 0; k < n2; k++) {
		Awrk[k] = A[i*n2+k];
	}

	for (j = 0; j < n3; j++) {
		for (k=iloc; k<n2; k+=nloc)
		Bwrk[k] = B[k*n2+j];
		barrier(CLK_LOCAL_MEM_FENCE);

		tmp = 0.0f;
		for (k = 0; k<n2; k++) {
			tmp += Awrk[k]*B[k*n3+j];
		}

		C[i*n3+j] += tmp;
	}
}
