inline float deg2rad(float pha) {
	float res = 0;
	res = pha*3.141592653589793/180.0;
	return res;
}

float nie_lq(float xi1,float xi2,float xc1,float xc2,float q,float rc,float re,float pha) {

    float phirad = deg2rad(pha);
    float cosa = cos(phirad);
    float sina = sin(phirad);
	float phi,a1,a2;
	float xt1,xt2;

	xt1 = (xi1-xc1)*cosa+(xi2-xc2)*sina;
	xt2 = (xi2-xc2)*cosa-(xi1-xc1)*sina;
	phi = sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc);

	a1 = sqrt(q)/sqrt(1.0-q*q)*atan(sqrt(1.0-q*q)*xt1/(phi+rc/q));
	a2 = sqrt(q)/sqrt(1.0-q*q)*atanh(sqrt(1.0-q*q)*xt2/(phi+rc*q));

	//alpha1[i] = (a1*cosa-a2*sina)*re;
	//alpha2[i] = (a2*cosa+a1*sina)*re;
	return 1.0;
}

__kernel void opencl_lq(
    __global float * xi1,
	__global float * alpha1,
	const int count)
{
    int i = get_global_id(0);
	if (i < count) {
		alpha1[i] = 1.0;
	}
}
