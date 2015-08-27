#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void wcic(float *cic_in,float *x_in,float *y_in,float bsx,float bsy,int nx,int ny,int np,float *cic_out) {
    float dx = bsx/nx;
    float dy = bsy/ny;
    float xc = bsx/2.0;
    float yc = bsy/2.0;
    float wx,wy;
    float xp,yp,zp;

    int i;
    int ip,jp;

    for (i=0;i<np;i++) {
        xp = (x_in[i]+xc)/dx-0.5;
        yp = (y_in[i]+yc)/dy-0.5;
		zp = cic_in[i];

        ip = (int)xp;
        jp = (int)yp;
        wx = 1.0-(xp-(float)ip);
        wy = 1.0-(yp-(float)jp);

		if (ip<0||ip>(nx-2)||jp<0||jp>(ny-2)) continue;

        cic_out[ip*ny+jp] += wx*wy*zp;
        cic_out[ip*ny+(jp+1)] += wx*(1.0-wy)*zp;
        cic_out[(ip+1)*ny+jp] += (1.0-wx)*wy*zp;
        cic_out[(ip+1)*ny+(jp+1)] += (1.0-wx)*(1.0-wy)*zp;
    }
}
