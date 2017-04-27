#include "PointCloudTransformer.h"

PointCloudTransformer::PointCloudTransformer(const char *filename_arg,
                                             int buff_size) {
  filename = filename_arg;
  pointcloud_fstream.open(filename);
  global_row_idx = 0;
  local_row_idx = 0;
  read_rows = 0;
  //parsed_rows = 0;
  row_buffer_size = buff_size;
  end_reached = false;
}

bool PointCloudTransformer::ReadNextRow() {
  if (pointcloud_fstream.is_open()) {
    if (std::getline(pointcloud_fstream, read_line)) {
      if (local_row_idx >= row_buffer_size) {
        local_row_idx = 0;
      }
      pointcloud_buffer[local_row_idx] = split(read_line, ' ');
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


void PointCloudTransformer::PopulateReadBuffer() {
  //raw_row_buffer.clear();
  pointcloud_buffer.clear();
  //raw_row_buffer.resize(row_buffer_size);
  pointcloud_buffer.resize(row_buffer_size);
  int i;
  for (i = 0; i < row_buffer_size; i++) {
    if (!ReadNextRow()) {
      row_buffer_size = i;
      //raw_row_buffer.resize(row_buffer_size);
      pointcloud_buffer.resize(row_buffer_size);
      end_reached = true;
      break;
    }
  }
  read_rows = i;
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

std::vector<float> PointCloudTransformer::split(const std::string &s,
                                                char delim) {
  std::vector<float> elems;
  split_(s, delim, elems);
  return elems;
}

PointCloudTransformer::~PointCloudTransformer() { }
