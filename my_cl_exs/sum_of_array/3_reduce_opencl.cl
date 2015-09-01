__kernel
void reduce(__global float* buffer,
            __const int block,
            __const int length,
            __global float* result) {

	int global_index = get_global_id(0) * block;
	float accumulator = INFINITY;
	int upper_bound = (get_global_id(0) + 1) * block;
	if (upper_bound > length) upper_bound = length;
	while (global_index < upper_bound) {
		float element = buffer[global_index];
		accumulator = (accumulator < element) ? accumulator : element;
		global_index++;
	}
	result[get_group_id(0)] = accumulator;
}
