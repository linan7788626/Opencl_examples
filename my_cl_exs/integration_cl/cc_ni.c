#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//float f(float x) {
//    return x*x;
//}

float f(float x) {
	return 16.0*(x-1.0)/(x*x*x*x-2*x*x*x+4*x-4);//(0,1)
}

//float f(float x) {
//	return 1.0/(x+1.0);//(0,1)
//}

float Simple_Trap(float a, float b) {
    float fA, fB;
    fA = f(a);
    fB = f(b);
    fA = f(a);
    fB = f(b);
    return ((fA + fB) * (b-a)) / 2;
}

float Comp_Trap( float a, float b,int n) {
    float Suma = 0;
    float i = 0;
	float INC = (b-a)/(float)n;
    i = a + INC;
    Suma += Simple_Trap(a,i);
    while(i < b)
    {
        i+=INC;
        Suma += Simple_Trap(i,i + INC);
    }
    return Suma;
}

float Simpson_Comp_Trap( float a, float b,int n) {
    float Suma = 0;
	float INC = (b-a)/(float)(n);
	float xt,yt;
	float y0 = f(a)+f(b);

    int i = 0;
	for (i = 1; i < n; i++) {
		xt =  a + INC*i;
		yt = pow(2.0,i%2+1)*f(xt);
		Suma = Suma + yt;
	}
	Suma = INC/3.0*(y0+Suma);
    return Suma;
}

int main(int argc, const char *argv[]) {
	float res1 = 0.0;
	float res2 = 0.0;
	float a = 0.0;
	float b = 1.0;
	int n = 100;

	res1 = Comp_Trap(a,b,n);
	res2 = Simpson_Comp_Trap(a,b,n);
	//printf("-----%f-----%f\n", res,pow(b-a,3.0)/3.0);
	printf("-----%f-----%f-----%f\n", res1,res2,M_PI);

	return 0;
}
