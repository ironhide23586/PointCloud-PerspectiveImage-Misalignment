#include <iostream>

#include "PointCloudTransformer.h"

using namespace std;

int main() {
  PointCloudTransformer *pcl = new PointCloudTransformer("final_project_point_cloud.fuse");
  pcl->PopulateReadBuffer();
  return 0;
}