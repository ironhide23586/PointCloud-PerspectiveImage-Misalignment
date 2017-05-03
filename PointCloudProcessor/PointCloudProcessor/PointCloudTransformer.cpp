#include "PointCloudTransformer.h"

PointCloudTransformer::PointCloudTransformer(const char *filename_arg,
                                             int buff_size) {
  filename = filename_arg;
  pointcloud_fstream.open(filename);
  global_row_idx = 0;
  local_row_idx = 0;
  read_rows = 0;
  buffer_size = buff_size;
  end_reached = false;
  ellipsoidal_flattening = (EQUATORIAL_RADIUS - POLAR_RADIUS)
    / EQUATORIAL_RADIUS;
  eccentricity = std::sqrtf(ellipsoidal_flattening
                            * (2 - ellipsoidal_flattening));
  CudaSafeCall(cudaGetDeviceProperties(&cudaProp, 0));
  CudaSafeCall(cudaMallocHost((void **)&h_positions_buffer_ptr,
                              sizeof(float) * POSITION_DIM * buff_size));
  CudaSafeCall(cudaMalloc((void **)&d_positions_buffer_ptr,
                          sizeof(float) * POSITION_DIM * buff_size));
  CudaSafeCall(cudaMallocHost((void **)&intensities_buffer_ptr,
                              sizeof(float) * buff_size));
}

void PointCloudTransformer::PopulateReadBuffer() {
  int i;
  for (i = 0; i < buffer_size; i++) {
    if (!ReadNextRow()) {
      buffer_size = i;
      end_reached = true;
      break;
    }
  }
  read_rows = i;
}

void PointCloudTransformer::LLA2ECEF_CPU(float phi, float lambda, float h, float eq_radius,
                                         float ecc, float *x, float *y, float *z) {
  float sin_phi = sinf(phi);
  float cos_phi = cosf(phi);
  float sin_lambda = sinf(lambda);
  float cos_lambda = cosf(lambda);

  float N_phi = eq_radius / sqrtf(1 - powf(ecc * sin_phi, 2.0f));
  float xy_p0 = (h + N_phi) * cos_lambda;

  *x = xy_p0 * cos_phi;
  *y = xy_p0 * sin_phi;
  *z = (h + (1 - ecc * ecc) * N_phi) * sin_lambda;
}

void PointCloudTransformer::LoadCameraDetails(float cam_phi, float cam_lambda, float cam_h,
                                              float cam_Qs, float cam_Qx,
                                              float cam_Qy, float cam_Qz) {
  reference_cam.phi = cam_phi;
  reference_cam.lambda = cam_lambda;
  reference_cam.h = cam_h;
  reference_cam.Qs = cam_Qs;
  reference_cam.Qx = cam_Qx;
  reference_cam.Qy = cam_Qy;
  reference_cam.Qz = cam_Qz;
  LLA2ECEF_CPU(reference_cam.phi, reference_cam.lambda, reference_cam.h,
               EQUATORIAL_RADIUS, eccentricity, &reference_cam.x,
               &reference_cam.y, &reference_cam.z);
  h_Rq = (float *)malloc(sizeof(float) * 3 * 3);
  init_Rq(h_Rq);
  CudaSafeCall(cudaMalloc((void **)&d_Rq, sizeof(float) * 3 * 3));
  CudaSafeCall(cudaMemcpy(d_Rq, h_Rq, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice));
}

void PointCloudTransformer::ConvertLLA2ENU_GPU() {
  LLA2NEmU_GPU(h_positions_buffer_ptr, d_positions_buffer_ptr, 
               EQUATORIAL_RADIUS, eccentricity, buffer_size, POSITION_DIM, reference_cam);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(h_positions_buffer_ptr, d_positions_buffer_ptr,
                          sizeof(float) * POSITION_DIM * buffer_size,
                          cudaMemcpyDeviceToHost));
}

void PointCloudTransformer::ConvertENU2CamCoord_GPU() {
  NEmU2Cam_GPU(d_positions_buffer_ptr, d_Rq);
  CudaCheckError();
}

std::vector<float> PointCloudTransformer::split(const std::string &s,
                                                char delim) {
  std::vector<float> elems;
  split_(s, delim, elems);
  return elems;
}

bool PointCloudTransformer::ReadNextRow() {
  std::vector<float> point_vect;
  if (pointcloud_fstream.is_open()) {
    if (std::getline(pointcloud_fstream, read_line)) {
      if (local_row_idx >= buffer_size) {
        local_row_idx = 0;
      }
      point_vect = split(read_line, ' ');
      for (int i = 0; i < point_vect.size(); i++) {
        if (i < POSITION_DIM) {
          h_positions_buffer_ptr[local_row_idx * POSITION_DIM + i]
            = point_vect[i];
          //std::cout << h_positions_buffer_ptr[local_row_idx * POSITION_DIM + i] << ", ";
        }
        else {
          intensities_buffer_ptr[local_row_idx] = point_vect[i];
          //std::cout << "---->" << intensities_buffer_ptr[local_row_idx];
        }
      }
      //std::cout << endl;
      local_row_idx++;
      global_row_idx++;
      return true;
    }
    else {
      pointcloud_fstream.close();
      return false;
    }
  }
  return false;
}

void PointCloudTransformer::init_Rq(float *h_Rq_ptr) {
  h_Rq_ptr[0] = std::powf(reference_cam.Qs, 2) + std::powf(reference_cam.Qx, 2)
    - std::powf(reference_cam.Qy, 2) - std::powf(reference_cam.Qz, 2);
  h_Rq_ptr[1] = 2 * (reference_cam.Qx * reference_cam.Qy 
                 - reference_cam.Qs * reference_cam.Qz);
  h_Rq_ptr[2] = 2 * (reference_cam.Qx * reference_cam.Qz 
                 + reference_cam.Qs * reference_cam.Qy);
  h_Rq_ptr[3] = 2 * (reference_cam.Qx * reference_cam.Qy 
                 + reference_cam.Qs * reference_cam.Qz);
  h_Rq_ptr[4] = std::powf(reference_cam.Qs, 2) - std::powf(reference_cam.Qx, 2)
    + std::powf(reference_cam.Qy, 2) - std::powf(reference_cam.Qz, 2);
  h_Rq_ptr[5] = 2 * (reference_cam.Qy * reference_cam.Qz 
                 - reference_cam.Qs * reference_cam.Qx);
  h_Rq_ptr[6] = 2 * (reference_cam.Qx * reference_cam.Qz 
                 - reference_cam.Qs * reference_cam.Qy);
  h_Rq_ptr[7] = 2 * (reference_cam.Qy * reference_cam.Qz 
                 + reference_cam.Qs * reference_cam.Qx);
  h_Rq_ptr[8] = std::powf(reference_cam.Qs, 2) - std::powf(reference_cam.Qx, 2)
    - std::powf(reference_cam.Qy, 2) + std::powf(reference_cam.Qz, 2);
}

void PointCloudTransformer::split_(const std::string &s, char delim,
                                   std::vector<float> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(stod(item));
  }
}

PointCloudTransformer::~PointCloudTransformer() { }