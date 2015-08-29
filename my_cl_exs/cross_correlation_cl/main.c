void autocorr(float *restrict A, float *restrict C, int N)
{
      int i, j;
      float sum;
      #pragma acc region
      {
        for (i = 0; i < N; i++) {
                        sum = 0.0;
                for (j = 0; j < N; j++) {
                    if ((i+j) < N)
                      sum += A[j] * A[i+j];
                    else
                      continue;
               }
            C[i] = sum;
       }
       }
}
