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

void PointCloudTransformer::ConvertLLA2ECEF_GPU() {
  LLA2ECEF_GPU(h_positions_buffer_ptr, d_positions_buffer_ptr, 
               EQUATORIAL_RADIUS, eccentricity, buffer_size, POSITION_DIM);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(h_positions_buffer_ptr, d_positions_buffer_ptr,
                          sizeof(float) * POSITION_DIM * buffer_size,
                          cudaMemcpyDeviceToHost));
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