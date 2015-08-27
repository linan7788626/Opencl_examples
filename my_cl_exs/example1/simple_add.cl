inline float add(float a,float b)		   
{										   
   return a+b;                             
}                                          
__kernel void square(                      
   __global float* input1,                  
   __global float* input2,                  
   __global float* output,                 
   const int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)                           
       output[i] = add(input1[i],input2[i]); 
}
