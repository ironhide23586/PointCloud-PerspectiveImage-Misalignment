#include <iostream>

#include "PointCloudTransformer.h"

using namespace std;

int main() {
  PointCloudTransformer *pcl = new PointCloudTransformer("final_project_point_cloud.fuse");
  pcl->PopulateReadBuffer();
  pcl->ConvertLLA2ECEF_GPU(pcl->pointcloud_buffer);
  return 0;
}