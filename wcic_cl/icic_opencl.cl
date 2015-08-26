__kernel void icic_cl(                      
   __global float* source_map,                  
   __global float* posy1,                  
   __global float* posy2,                  
   const float ysc1,
   const float ysc2,
   const float dsi,
   const int nsx,
   const int nsy,
   const int nlx,
   const int nly,
   __global float* lensed_map)               
{                                          

    int i1,j1;
    int index;
    float xb1,xb2;
    float ww1,ww2,ww3,ww4,wx,wy;

	int index = get_global_id(0);               
	if(index < nlx*nly) {
		
		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0-0.5;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0-0.5;
		
		i1 = (int)xb1;
		j1 = (int)xb2;
		
		wx = 1.-(xb1-(float)(i1));
		wy = 1.-(xb2-(float)(j1));
		
		ww1 = wx*wy;
		ww2 = wx*(1.0-wy);
		ww3 = (1.0-wx)*wy;
		ww4 = (1.0-wx)*(1.0-wy);
		
		if (i1<0||i1>nsx-2||j1<0||j1>nsy-2) {
			lensed_map[index] = 0.0;
		};
		else {
			lensed_map[index] = ww1*source_map[i1*nsx+j1]
			                  + ww2*source_map[i1*nsx+j1+1]
			                  + ww3*source_map[(i1+1)*nsx+j1]
			                  + ww4*source_map[(i1+1)*nsx+j1+1];
		}
	}
}
