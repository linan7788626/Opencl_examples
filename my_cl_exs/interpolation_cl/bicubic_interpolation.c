#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double CatMullRom(double x) {
    const double B = 0.0;
    const double C = 0.5;
    double f = x;
    if (f<0.0) f = -f;
    if (f<1.0) {
        return ((12-9*B-6*C)*(f*f*f)+(-18+12*B+6*C)*(f*f)+(6-2*B))/6.0;
    }
    else if (f>=1.0 && f<2.0) {
        return ((-B-6*C)*(f*f*f)+(6*B+30*C)*(f*f)+(-(12*B)-48*C)*f+8*B+24*C)/6.0;
    }
    else {
        return 0.0;
    }
}


void inverse_cic(double *source_map, double *posy1, double *posy2, double ysc1, double ysc2,double dsi, int nsx, int nsy, int nlx, int nly, double *lensed_map) {

	int i1,j1,i,j;
	int index;
	double xb1,xb2;
	double ww1,ww2,ww3,ww4,wx,wy;

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(double)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(double)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;

		wx = 1.-(xb1-(double)(i1));
		wy = 1.-(xb2-(double)(j1));

		ww1 = wx*wy;
		ww2 = wx*(1.0-wy);
		ww3 = (1.0-wx)*wy;
		ww4 = (1.0-wx)*(1.0-wy);

		if (i1<0||i1>nsx-2||j1<0||j1>nsy-2) continue;

		lensed_map[index] = ww1*source_map[i1*nsx+j1]
						  + ww2*source_map[i1*nsx+j1+1]
						  + ww3*source_map[(i1+1)*nsx+j1]
						  + ww4*source_map[(i1+1)*nsx+j1+1];
	}
}

void  inverse_bicubic(double *img_in, double *posy1,double *posy2,double ysc1,double ysc2,double dsi,int nsx, int nsy, int nlx, int nly, double *img_out) {
	int i1,j1,i,j,k,l,ii,jj;
	int index;
	double xb1,xb2;
	double dx,dy;

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(double)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(double)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			ii = i1+k;
			jj = j1+l;
			dx = xb1-(double)(ii);
			dy = xb2-(double)(jj);
			img_out[index] = img_out[index] + CatMullRom(dx)*CatMullRom(dy)*img_in[ii*nsx+jj];
		}
    }
}

double BSpline_func(double x) {
	double f = x;
	if (f<0.0) f = -f;

	if (f<=0 && f<=1) {
		return 2.0/3.0+0.5*pow(f,3.0)-f*f;
	}
	if (f>1 && f<=2) {
		return 1.0/6.0*pow((2-f),3.0);
	}

	return 0.0;
}

double BellFunc(double x)
{
	double f = (x/2.0)*1.5; // Converting -2 to +2 to -1.5 to +1.5
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

void call_factors(double p[4][4],double a[4][4]) {
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

double getValue (double x, double y,double a[4][4]) {
	double x2 = x * x;
	double x3 = x2 * x;
	double y2 = y * y;
	double y3 = y2 * y;

	return (a[0][0] + a[0][1]*y + a[0][2]*y2 + a[0][3]*y3) +
	       (a[1][0] + a[1][1]*y + a[1][2]*y2 + a[1][3]*y3)*x +
	       (a[2][0] + a[2][1]*y + a[2][2]*y2 + a[2][3]*y3)*x2 +
	       (a[3][0] + a[3][1]*y + a[3][2]*y2 + a[3][3]*y3)*x3;
}

void  inverse_bicubic_polygen(double *img_in, double *posy1,double *posy2,double ysc1,double ysc2,double dsi,int nsx, int nsy, int nlx, int nly, double *img_out) {
	int i1,j1,i,j,k,l;
	int index;
	double xb1,xb2;
	double dx,dy;
	double p[4][4];
	double a[4][4];

	for(i=0;i<nlx;i++) for(j=0;j<nly;j++) {

		index = i*nlx+j;

		xb1 = (posy1[index]-ysc1)/dsi+(double)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(double)nsy/2.0-0.5;

		i1 = (int)xb1;
		j1 = (int)xb2;
		dx = xb1-(double)(i1);
		dy = xb2-(double)(j1);

		if (i1<1||i1>nsx-3||j1<1||j1>nsy-3) continue;

		for (k=-1;k<3;k++) for (l=-1;l<3;l++) {
			p[k+1][l+1] = img_in[(i1+k)*nsx+(j1+l)];
		}
		call_factors(p,a);
		img_out[index] = getValue(dx,dy,a);
    }
}

int main() {
	return 0;
}
