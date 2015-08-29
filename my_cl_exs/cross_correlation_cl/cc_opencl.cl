__kernel void autocorrel1D(__global double *Vol_IN, __global double *Vol_AUTOCORR, int size)
{

    int j, gid = get_global_id(0);
    double sum = 0.0;

    for (j = 0; j < size; j++) {
            if ((gid+j) < size)
            {
               sum += Vol_IN[j] * Vol_IN[gid+j];
            }
            else
               continue;
               }

    Vol_AUTOCORR[gid] = sum;
}
