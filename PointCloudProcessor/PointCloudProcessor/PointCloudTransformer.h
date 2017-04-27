#pragma once

#include <cuda.h>
#include <cublas.h>
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

#include "cuda_err_check.h"

#define EQUATORIAL_RADIUS 6378137.0f // 'a'
#define POLAR_RADIUS 6356752.3142f // 'b'

#define POSITION_DIM 3

using namespace std;

void LLA2ECEF_GPU(float *h_lla_data, float *d_ecef_data, float eq_radius,
                  float ecc, int num_samples, int sample_size);

class PointCloudTransformer {
public:

  const char *filename;
  std::ifstream pointcloud_fstream;
  std::string read_line;
  //std::vector<std::vector<float>> pointcloud_buffer;
  float *h_positions_buffer_ptr, *d_positions_buffer_ptr;
  float *intensities_buffer_ptr;
  
  cudaDeviceProp cudaProp;

  int global_row_idx;
  int local_row_idx;
  int buffer_size;
  bool end_reached;

  float ellipsoidal_flattening; // 'f'
  float eccentricity; // 'e'

  int read_rows;

  PointCloudTransformer(const char *filename_arg, int buff_size = 5);
  void PopulateReadBuffer();
  
  void ConvertLLA2ECEF_GPU();

  std::vector<float> static split(const std::string &s,
                                  char delim);


private:
  bool ReadNextRow();
  void static split_(const std::string &s, char delim,
                     std::vector<float> &elems);

  ~PointCloudTransformer();
};

