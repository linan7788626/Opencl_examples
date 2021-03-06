__kernel void icic_cl(                      
	__const __global float* source_map,                  
	__const __global float* posy1,                  
	__const __global float* posy2,
	__global float* lensed_map,
	const float ysc1,
	const float ysc2,
	const float dsi,
	const int nsx,
	const int nsy,
	const int nlx,
	const int nly)               
{                                          

    int i1,j1;
    float xb1,xb2;
    float ww1,ww2,ww3,ww4,wx,wy;
	int index = get_global_id(0);               
	if(index < nlx*nly) {
		
		xb1 = (posy1[index]-ysc1)/dsi+(float)nsx/2.0f-0.5f;
		xb2 = (posy2[index]-ysc2)/dsi+(float)nsy/2.0f-0.5f;
		
		i1 = (int)xb1;
		j1 = (int)xb2;
		
		wx = 1.0f-(xb1-(float)(i1));
		wy = 1.0f-(xb2-(float)(j1));
		
		ww1 = wx*wy;
		ww2 = wx*(1.0f-wy);
		ww3 = (1.0f-wx)*wy;
		ww4 = (1.0f-wx)*(1.0f-wy);

		if (i1<0||i1>nsx-2||j1<0||j1>nsy-2) {
			lensed_map[index] = 0.0f;
		};
		lensed_map[index] = ww1*source_map[i1*nsx+j1]
		                  + ww2*source_map[i1*nsx+j1+1]
		                  + ww3*source_map[(i1+1)*nsx+j1]
		                  + ww4*source_map[(i1+1)*nsx+j1+1];
	}
}
