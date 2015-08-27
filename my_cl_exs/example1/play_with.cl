inline float deg2rad(float pha) {
	float res = 0;
	res = pha*3.141592653589793f/180.0f;
	return res;
}

inline float nie_lq_alpha1(float xi1,float xi2,float xc1,float xc2,float q,float rc,float re,float pha) {

    float phirad = deg2rad(pha);
    float cosa = cos(phirad);
    float sina = sin(phirad);
	float phi,a1,a2;
	float xt1,xt2;

	xt1 = (xi1-xc1)*cosa+(xi2-xc2)*sina;
	xt2 = (xi2-xc2)*cosa-(xi1-xc1)*sina;
	phi = sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc);

	a1 = sqrt(q)/sqrt(1.0f-q*q)*atan(sqrt(1.0f-q*q)*xt1/(phi+rc/q));
	a2 = sqrt(q)/sqrt(1.0f-q*q)*atanh(sqrt(1.0f-q*q)*xt2/(phi+rc*q));

	//alpha1[i] = (a1*cosa-a2*sina)*re;
	//alpha2[i] = (a2*cosa+a1*sina)*re;
	float res = 0;
	res =  (a1*cosa-a2*sina)*re;
	return res;
}

inline float nie_lq_alpha2(float xi1,float xi2,float xc1,float xc2,float q,float rc,float re,float pha) {

    float phirad = deg2rad(pha);
    float cosa = cos(phirad);
    float sina = sin(phirad);
	float phi,a1,a2;
	float xt1,xt2;

	xt1 = (xi1-xc1)*cosa+(xi2-xc2)*sina;
	xt2 = (xi2-xc2)*cosa-(xi1-xc1)*sina;
	phi = sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc);

	a1 = sqrt(q)/sqrt(1.0f-q*q)*atan(sqrt(1.0f-q*q)*xt1/(phi+rc/q));
	a2 = sqrt(q)/sqrt(1.0f-q*q)*atanh(sqrt(1.0f-q*q)*xt2/(phi+rc*q));

	//alpha1[i] = (a1*cosa-a2*sina)*re;
	//alpha2[i] = (a2*cosa+a1*sina)*re;
	float res = 0;
	res = (a2*cosa+a1*sina)*re;
	return res;
}

inline float add(float a,float b)		   
{										   
   return sin(a+b);                             
}                                          
__kernel void square(                      
   __global float* input1,                  
   __global float* input2,                  
   __global float* lpar,                  
   __global float* output1,                 
   __global float* output2,                 
   const int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count) {

		//output1[i] = nie_lq_alpha1(input1[i],input2[i],lpar[0],lpar[1],lpar[2],lpar[3],lpar[4],lpar[5]); 
		//output2[i] = nie_lq_alpha2(input1[i],input2[i],lpar[0],lpar[1],lpar[2],lpar[3],lpar[4],lpar[5]); 
		float xc1 = lpar[0];
		float xc2 = lpar[1];
		float q = lpar[2];
		float rc = lpar[3];
		float re = lpar[4];
		float pha = lpar[5];
		
    	float phirad = deg2rad(pha);
    	float cosa = cos(phirad);
    	float sina = sin(phirad);
		float phi,a1,a2;
		float xt1,xt2;

		xt1 = (input1[i]-xc1)*cosa+(input2[i]-xc2)*sina;
		xt2 = (input2[i]-xc2)*cosa-(input1[i]-xc1)*sina;
		phi = sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc);

		a1 = sqrt(q)/sqrt(1.0-q*q)*atan(sqrt(1.0-q*q)*xt1/(phi+rc/q));
		a2 = sqrt(q)/sqrt(1.0-q*q)*atanh(sqrt(1.0-q*q)*xt2/(phi+rc*q));

		output1[i] = (a1*cosa-a2*sina)*re;
		output2[i] = (a2*cosa+a1*sina)*re;
	}
}
