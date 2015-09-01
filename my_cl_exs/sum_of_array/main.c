#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float reduce_sum(float* input, int length) {
	float accumulator = input[0];
	int i = 1;
	for(i = 1; i < length; i++)
		accumulator += input[i];
	return accumulator;
}
