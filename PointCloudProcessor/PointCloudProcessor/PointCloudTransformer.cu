#include "PointCloudTransformer.h"
#include "math_functions.h"

__global__ void LLA2ECEF_GPUKernel(float *h_lla_data, float *d_ecef_data,
                                   float eq_radius, float ecc) {
  int phi_idx = blockIdx.x * POSITION_DIM;
  int lambda_idx = phi_idx + 1;
  int h_idx = phi_idx + 2;

  float phi = h_lla_data[phi_idx];
  float lambda = h_lla_data[lambda_idx];
  float h = h_lla_data[h_idx];

  float N_phi = eq_radius / sqrtf(1 - powf(ecc * sinf(phi), 2.0f));
  float xy_p0 = (h + N_phi) * cosf(lambda);

  if (threadIdx.x == 0)
    d_ecef_data[phi_idx] = xy_p0 * cosf(phi);
  else if (threadIdx.x == 1)
    d_ecef_data[lambda_idx] = xy_p0 * sinf(phi);
  else
    d_ecef_data[h_idx] = (h + (1 - ecc * ecc) * N_phi) * sinf(lambda);
}

void LLA2ECEF_GPU(float *h_lla_data, float *d_ecef_data, float eq_radius,
                  float ecc, int num_samples, int sample_size) {
  LLA2ECEF_GPUKernel <<< num_samples, sample_size >>> (h_lla_data, d_ecef_data,
                                                       eq_radius, ecc);
}