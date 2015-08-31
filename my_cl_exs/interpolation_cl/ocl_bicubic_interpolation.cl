inline float CatMullRom(float x) {
    const float B = 0.0f;
    const float C = 0.5f;
    float f = x;
    if (f<0.0f) f = -f;
    if (f<1.0f) {
        return ((12.0f-9.0f*B-6.0f*C)*(f*f*f)+(-18.0f+12.0f*B+6*C)*(f*f)+(6.0f-2.0f*B))/6.0f;
    }
    else if (f>=1.0f && f<2.0f) {
        return ((-B-6.0f*C)*(f*f*f)+(6.0f*B+30.0f*C)*(f*f)+(-(12.0f*B)-48.0f*C)*f+8.0f*B+24.0f*C)/6.0f;
    }
    else {
        return 0.0f;
    }
}

inline float BSpline_func(float x) {
	float f = x;
	if (f<0.0f) f = -f;

	if (f<=0.0f && f<=1.0f) {
		return 2.0f/3.0f+0.5f*pow(f,3.0f)-f*f;
	}
	if (f>1 && f<=2) {
		return 1.0f/6.0f*pow((2.0f-f),3.0f);
	}
	return 0.0f;
}

inline float BellFunc(float x) {
	float f = (x/2.0f)*1.5f; // Converting -2 to +2 to -1.5 to +1.5
	if( f>-1.5f && f<-0.5f ) {
		return(0.5f*pow(f+1.5f,2.0f));
	}
	else if(f>-0.5f && f<0.5f) {
		return 3.0f/4.0f-(f*f);
	}
	else if(f>0.5f && f<1.5f) {
		return(0.5f*pow(f-1.5f,2.0f));
	}
	return 0.0f;
}

__kernel void inverse_bicubic_cl(                      
	__global float* img_in,                  
	__global float* posy1,                  
	__global float* posy2,
	__global float* img_out,
	const float ysc1,
	const float ysc2,
	const float dsi,
	const int nsx,
	const int nsy,
	const int nlx,
	const int nly)               
{                                          

	int i1,j1,i,j,ii,jj;
	int index = get_global_id(0);               
	float xb1,xb2;
	float dx,dy;

	xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0f-0.5f;
	xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0f-0.5f;

	i1 = (int)xb1;
	j1 = (int)xb2;

	if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) {
		img_out[index] = 0.0f;
	};
	for (i=-1;i<3;i++) for (j=-1;j<3;j++) {
		ii = i1+i;
		jj = j1+j;
		dx = xb1-(float)(ii);
		dy = xb2-(float)(jj);
		img_out[index] = img_out[index] + CatMullRom(dx)*CatMullRom(dy)*img_in[ii*nsx+jj];
	}
}
