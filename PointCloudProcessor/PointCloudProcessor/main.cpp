#include <iostream>

#include "PointCloudTransformer.h"

using namespace std;

void print_h_var(float *h_v, int r, int c, bool print_elem = true) {
  std::cout << "-------------------------" << std::endl;
  float mini = h_v[0], maxi = h_v[0];
  float sum = 0.0f;
  int mini_idx = 0, maxi_idx = 0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        //std::cout << h_v[j + i * c] << "\t";
        printf("%.9f\t", h_v[j + i * c]);
      if (h_v[j + i * c] < mini) {
        mini = h_v[j + i * c];
        mini_idx = j + i * c;
      }
      if (h_v[j + i * c] > maxi) {
        maxi = h_v[j + i * c];
        maxi_idx = j + i * c;
      }
      sum += h_v[j + i * c];
    }
    if (print_elem)
      std::cout << std::endl;
  }
  std::cout << "Shape = (" << r << ", " << c << ")" << std::endl;
  std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
  std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
  std::cout << "Average of all elements = " << sum / (r * c) << std::endl;
  // std::cout << std::endl;
}

int main() {
  PointCloudTransformer *pcl = new PointCloudTransformer("final_project_point_cloud.fuse", 400000);
  pcl->PopulateReadBuffer();
  //pcl->PopulateReadBuffer();
  //print_h_var(pcl->h_positions_buffer_ptr, pcl->buffer_size, 3);
  pcl->ConvertLLA2ECEF_GPU();
  //print_h_var(pcl->h_positions_buffer_ptr, pcl->buffer_size, 3);
  return 0;
}