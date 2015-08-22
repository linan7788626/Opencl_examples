__kernel void mmul(
	__global float *A,
	__global float *B,
	const int n1,
	const int n2,
	const int n3,
	__global float *C) {
	int k;
	int i = get_global_id(0);
	//int j = get_global_id(1);
	float tmp = 0.0f;
	//if (i < n1 && j < n3) {
	if (i < n1 * n3) {
		//for (k = 0; k<n2; k++) {
		//	tmp += A[i*n2+k]*B[k*n3+j];
		//}
		//C[i*n3+j] += tmp;
		C[i] = A[i]+B[i];
	}
}
