inline float add(float a,float b)		   
{										   
   return a+b;                             
}                                          
__kernel void square(                      
   __global float* input,                  
   __global float* output,                 
   const unsigned int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)                           
       //output[i] = input[i] * input[i];  
       output[i] = add(input[i],input[i]); 
}
