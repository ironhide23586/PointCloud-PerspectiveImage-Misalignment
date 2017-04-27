#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

class PointCloudTransformer {
public:

  const char *filename;
  std::ifstream pointcloud_fstream;
  std::string read_line;
  std::vector<std::vector<float>> pointcloud_buffer;

  int global_row_idx;
  int local_row_idx;
  int row_buffer_size;
  bool end_reached;

  int read_rows;
  //int parsed_rows;

  PointCloudTransformer(const char *filename_arg, int buff_size = 5);
  void PopulateReadBuffer();
  std::vector<float> static split(const std::string &s,
                                  char delim);

private:
  bool ReadNextRow();
  
  void static split_(const std::string &s, char delim,
                     std::vector<float> &elems);

  ~PointCloudTransformer();
};

