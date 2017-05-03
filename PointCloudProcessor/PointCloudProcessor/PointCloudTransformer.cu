#include "PointCloudTransformer.h"
#include "math_functions.h"

__global__ void LLA2NEmU_GPUKernel(float *h_lla_data, float *d_out_coord_data,
                                  float eq_radius, float ecc,
                                  float x0, float y0, float z0) {
  int phi_idx = blockIdx.x * POSITION_DIM;
  int lambda_idx = phi_idx + 1;
  int h_idx = phi_idx + 2;

  float phi = h_lla_data[phi_idx];
  float lambda = h_lla_data[lambda_idx];
  float h = h_lla_data[h_idx];

  float sin_phi = sinf(phi);
  float cos_phi = cosf(phi);
  float sin_lambda = sinf(lambda);
  float cos_lambda = cosf(lambda);

  float N_phi = eq_radius / sqrtf(1 - powf(ecc * sin_phi, 2.0f));
  float xy_p0 = (h + N_phi) * cos_lambda;

  if (threadIdx.x == 0)
    d_out_coord_data[phi_idx] = (xy_p0 * cos_phi) - x0;
  else if (threadIdx.x == 1)
    d_out_coord_data[lambda_idx] = (xy_p0 * sin_phi) - y0;
  else
    d_out_coord_data[h_idx] = ((h + (1 - ecc * ecc) * N_phi) * sin_lambda) 
                              - z0;
  __syncthreads();

  if (threadIdx.x == 0)
    d_out_coord_data[lambda_idx] = (d_out_coord_data[phi_idx] * (-sin_lambda))
                                    + (d_out_coord_data[lambda_idx]
                                       * cos_lambda);
  else if (threadIdx.x == 1)
    d_out_coord_data[phi_idx] = (d_out_coord_data[phi_idx] * (-cos_lambda) 
                                 * sin_phi)
                                 + (d_out_coord_data[lambda_idx] * (-sin_phi)
                                    * sin_lambda)
                                 + (d_out_coord_data[h_idx] * cos_phi);
  else
    d_out_coord_data[h_idx] = -((d_out_coord_data[phi_idx] * cos_phi 
                                * cos_lambda)
                               + (d_out_coord_data[lambda_idx] * cos_phi
                                 * sin_lambda)
                               + (d_out_coord_data[h_idx] * sin_phi));

}

void LLA2NEmU_GPU(float *h_lla_data, float *d_nemu_data, float eq_radius,
                 float ecc, int num_samples, int sample_size,
                 cam_details &ref_cam) {
  LLA2NEmU_GPUKernel << < num_samples, sample_size >> > (h_lla_data, d_nemu_data,
                                                        eq_radius, ecc,
                                                        ref_cam.x, ref_cam.y,
                                                        ref_cam.z);
}

void NEmU2Cam_GPU(float *d_nemu_data, float *d_Rq) {
  
}