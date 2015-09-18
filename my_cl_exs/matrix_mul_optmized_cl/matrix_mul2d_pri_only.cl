__kernel void mmul(
	const int n1,
	const int n2,
	const int n3,
	__global float *A,
	__global float *B,
	__global float *C) {

	int k;

	int i = get_global_id(0);
	int j = get_global_id(1);
	float tmp;
	float Awrk[1024];
	for (k = 0; k < n2; k++) {
		Awrk[k] = A[i*n2+k];
	}

	tmp = 0.0f;
	for (k = 0; k<n2; k++) {
		tmp += Awrk[k]*B[k*n3+j];
	}
	C[i*n3+j] += tmp;
}
