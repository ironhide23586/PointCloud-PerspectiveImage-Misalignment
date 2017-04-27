#include "PointCloudTransformer.h"

PointCloudTransformer::PointCloudTransformer(const char *filename_arg,
                                             int buff_size) {
  filename = filename_arg;
  pointcloud_fstream.open(filename);
  global_row_idx = 0;
  local_row_idx = 0;
  read_rows = 0;
  row_buffer_size = buff_size;
  end_reached = false;
  ellipsoidal_flattening = (EQUATORIAL_RADIUS - POLAR_RADIUS)
    / EQUATORIAL_RADIUS;
  eccentricity = std::sqrtf(ellipsoidal_flattening
                            * (2 - ellipsoidal_flattening));
  CudaSafeCall(cudaGetDeviceProperties(&cudaProp, 0));
  CudaSafeCall(cudaMallocHost((void **)&positions_buffer_ptr,
                              sizeof(float) * POSITION_DIM * buff_size));
  CudaSafeCall(cudaMallocHost((void **)&intensities_buffer_ptr,
                              sizeof(float) * buff_size));
}

void PointCloudTransformer::PopulateReadBuffer() {
  pointcloud_buffer.clear();
  pointcloud_buffer.resize(row_buffer_size);
  int i;
  for (i = 0; i < row_buffer_size; i++) {
    if (!ReadNextRow()) {
      row_buffer_size = i;
      pointcloud_buffer.resize(row_buffer_size);
      end_reached = true;
      break;
    }
  }
  read_rows = i;
}

void PointCloudTransformer::ConvertLLA2ECEF_GPU(std::vector<std::vector<float>>
                                                &lla_points) {
  int num_points = lla_points.size();
  int pcloud_size = lla_points[0].size();
  int i = 0;
  while (true) {
    float k = *(&lla_points[0][0] + i);
    std::cout << k << std::endl;
    i++;
  }
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
      if (local_row_idx >= row_buffer_size) {
        local_row_idx = 0;
      }
      //pointcloud_buffer[local_row_idx] = split(read_line, ' ');
      point_vect = split(read_line, ' ');
      for (int i = 0; i < point_vect.size(); i++) {
        if (i < POSITION_DIM) {
          positions_buffer_ptr[local_row_idx * POSITION_DIM + i]
            = point_vect[i];
          std::cout << positions_buffer_ptr[local_row_idx * POSITION_DIM + i] << ", ";
        }
        else {
          intensities_buffer_ptr[local_row_idx] = point_vect[i];
          std::cout << "---->" << intensities_buffer_ptr[local_row_idx];
        }
      }
      std::cout << endl;

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