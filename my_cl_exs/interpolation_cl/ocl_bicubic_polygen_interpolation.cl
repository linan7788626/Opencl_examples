inline float CatMullRom(float x) {
    const float B = 0.0f;
    const float C = 0.5f;
    float f = x;
    if (f<0.0) f = -f;
    if (f<1.0) {
        return ((12.0f-9.0f*B-6.0f*C)*(f*f*f)+(-18.0f+12.0f*B+6*C)*(f*f)+(6.0f-2.0f*B))/6.0f;
    }
    else if (f>=1.0f && f<2.0f) {
        return ((-B-6.0f*C)*(f*f*f)+(6.0f*B+30.0f*C)*(f*f)+(-(12.0f*B)-48.0f*C)*f+8.0f*B+24.0f*C)/6.0f;
    }
    else {
        return 0.0f;
    }
}


void  inverse_bicubic(float *img_in, float *posy1,float *posy2,float ysc1,float ysc2,float dsi,int nsx, int nsy, int nlx, int nly, float *img_out) {
	int i1,j1,i,j,k,l,ii,jj;
	int index;
	float xb1,xb2;
	float dx,dy;

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			ii = i1+k;
			jj = j1+l;
			dx = xb1-(float)(ii);
			dy = xb2-(float)(jj);
			img_out[index] = img_out[index] + CatMullRom(dx)*CatMullRom(dy)*img_in[ii*nsx+jj];
		}
    }
}

inline float BSpline_func(float x) {
	float f = x;
	if (f<0.0) f = -f;

	if (f<=0 && f<=1) {
		return 2.0/3.0+0.5*pow(f,3.0)-f*f;
	}
	if (f>1 && f<=2) {
		return 1.0/6.0*pow((2-f),3.0);
	}
	return 0.0;
}

float BellFunc(float x)
{
	float f = (x/2.0)*1.5; // Converting -2 to +2 to -1.5 to +1.5
	if( f>-1.5 && f<-0.5 ) {
		return(0.5*pow(f+1.5,2.0));
	}
	else if(f>-0.5 && f<0.5) {
		return 3.0/4.0-(f*f);
	}
	else if(f>0.5 && f<1.5) {
		return(0.5*pow(f-1.5,2.0));
	}
	return 0.0;
}

void call_factors(float p[4][4],float a[4][4]) {
		a[0][0] = p[1][1];
		a[0][1] = -.5*p[1][0] + .5*p[1][2];
		a[0][2] = p[1][0] - 2.5*p[1][1] + 2*p[1][2] - .5*p[1][3];
		a[0][3] = -.5*p[1][0] + 1.5*p[1][1] - 1.5*p[1][2] + .5*p[1][3];
		a[1][0] = -.5*p[0][1] + .5*p[2][1];
		a[1][1] = .25*p[0][0] - .25*p[0][2] - .25*p[2][0] + .25*p[2][2];
		a[1][2] = -.5*p[0][0] + 1.25*p[0][1] - p[0][2] + .25*p[0][3] + .5*p[2][0] - 1.25*p[2][1] + p[2][2] - .25*p[2][3];
		a[1][3] = .25*p[0][0] - .75*p[0][1] + .75*p[0][2] - .25*p[0][3] - .25*p[2][0] + .75*p[2][1] - .75*p[2][2] + .25*p[2][3];
		a[2][0] = p[0][1] - 2.5*p[1][1] + 2*p[2][1] - .5*p[3][1];
		a[2][1] = -.5*p[0][0] + .5*p[0][2] + 1.25*p[1][0] - 1.25*p[1][2] - p[2][0] + p[2][2] + .25*p[3][0] - .25*p[3][2];
		a[2][2] = p[0][0] - 2.5*p[0][1] + 2*p[0][2] - .5*p[0][3] - 2.5*p[1][0] + 6.25*p[1][1] - 5*p[1][2] + 1.25*p[1][3] + 2*p[2][0] - 5*p[2][1] + 4*p[2][2] - p[2][3] - .5*p[3][0] + 1.25*p[3][1] - p[3][2] + .25*p[3][3];
		a[2][3] = -.5*p[0][0] + 1.5*p[0][1] - 1.5*p[0][2] + .5*p[0][3] + 1.25*p[1][0] - 3.75*p[1][1] + 3.75*p[1][2] - 1.25*p[1][3] - p[2][0] + 3*p[2][1] - 3*p[2][2] + p[2][3] + .25*p[3][0] - .75*p[3][1] + .75*p[3][2] - .25*p[3][3];
		a[3][0] = -.5*p[0][1] + 1.5*p[1][1] - 1.5*p[2][1] + .5*p[3][1];
		a[3][1] = .25*p[0][0] - .25*p[0][2] - .75*p[1][0] + .75*p[1][2] + .75*p[2][0] - .75*p[2][2] - .25*p[3][0] + .25*p[3][2];
		a[3][2] = -.5*p[0][0] + 1.25*p[0][1] - p[0][2] + .25*p[0][3] + 1.5*p[1][0] - 3.75*p[1][1] + 3*p[1][2] - .75*p[1][3] - 1.5*p[2][0] + 3.75*p[2][1] - 3*p[2][2] + .75*p[2][3] + .5*p[3][0] - 1.25*p[3][1] + p[3][2] - .25*p[3][3];
		a[3][3] = .25*p[0][0] - .75*p[0][1] + .75*p[0][2] - .25*p[0][3] - .75*p[1][0] + 2.25*p[1][1] - 2.25*p[1][2] + .75*p[1][3] + .75*p[2][0] - 2.25*p[2][1] + 2.25*p[2][2] - .75*p[2][3] - .25*p[3][0] + .75*p[3][1] - .75*p[3][2] + .25*p[3][3];
}

float getValue (float x, float y,float a[4][4]) {
	float x2 = x * x;
	float x3 = x2 * x;
	float y2 = y * y;
	float y3 = y2 * y;

	return (a[0][0] + a[0][1]*y + a[0][2]*y2 + a[0][3]*y3) +
	       (a[1][0] + a[1][1]*y + a[1][2]*y2 + a[1][3]*y3)*x +
	       (a[2][0] + a[2][1]*y + a[2][2]*y2 + a[2][3]*y3)*x2 +
	       (a[3][0] + a[3][1]*y + a[3][2]*y2 + a[3][3]*y3)*x3;
}

void  inverse_bicubic_polygen(float *img_in, float *posy1,float *posy2,float ysc1,float ysc2,float dsi,int nsx, int nsy, int nlx, int nly, float *img_out) {
	int i1,j1,i,j,k,l;
	int index;
	float xb1,xb2;
	float dx,dy;
	float p[4][4];
	float a[4][4];

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;
		dx = xb1-(float)(i1);
		dy = xb2-(float)(j1);

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			p[k+1][l+1] = img_in[(i1+k)*nsx+(j1+l)];
		}
		call_factors(p,a);
		img_out[index] = getValue(dx,dy,a);
    }
}
