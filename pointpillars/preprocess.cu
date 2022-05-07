/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/

/**
* @author Ye xiubo
* Contact:github.com/speshowBUAA
* @date 2022/01/05
*/

// headers in STL
#include <stdio.h>
// headers in local files
#include "common.h"
#include "preprocess.h"
#include <cuda_runtime.h>

#define PFNDIM 64

// step 1: 统计coor每个位置上的点云数目
// dev_pillar_point_feature_in_coors将点云按照大小顺序排好，
// pillar_count_histo 储存每一个位置上的点云数目
__global__ void make_pillar_histo_kernel_mean_coors(
    const float* dev_points, float* dev_points_sum_x, float* dev_points_sum_y, float* dev_points_sum_z,
    int* pillar_count_histo, int* dev_ith_to_pointidx, const int num_points,
    const int grid_x_size,  const int grid_y_size, const int grid_z_size, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size,
    const int num_point_feature, int* valid_num_point) {
  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1);
    int count = atomicAdd(valid_num_point, 1);
    float x = dev_points[th_i * num_point_feature + 0];
    float y = dev_points[th_i * num_point_feature + 1];
    float z = dev_points[th_i * num_point_feature + 2];
    atomicAdd(&dev_points_sum_x[y_coor * grid_x_size + x_coor], x);
    atomicAdd(&dev_points_sum_y[y_coor * grid_x_size + x_coor], y);
    atomicAdd(&dev_points_sum_z[y_coor * grid_x_size + x_coor], z);
    dev_ith_to_pointidx[th_i] = count;
  }
}

// dev_pfe_gather_feature_ 加入全部点云中心3维坐标特征 再加全部体素中心3维坐标
__global__ void gather_point_feature_kernel_coors(
  const float* dev_points, const int num_point_feature, const int num_points,
  const float min_x_range, const float min_y_range, const float min_z_range, 
  const float pillar_x_size,  const float pillar_y_size, const float pillar_z_size, 
  const int grid_x_size, const int grid_y_size, const int grid_z_size,
  float* dev_points_sum_x, float* dev_points_sum_y, float* dev_points_sum_z,
  float* dev_pfe_gather_feature_, int* dev_points_coors, int* pillar_count_histo,
  int* dev_ith_to_pointidx){

  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  if (dev_ith_to_pointidx[th_i] == -1) {
    return;
  }
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) 
  {
    unsigned int pointidx = dev_ith_to_pointidx[th_i];
    dev_points_coors[pointidx * 4 + 0] = 0;  // batch idx
    dev_points_coors[pointidx * 4 + 1] = 0;  // z
    dev_points_coors[pointidx * 4 + 2] = y_coor;
    dev_points_coors[pointidx * 4 + 3] = x_coor;

    int num_gather_feature = 9;    // mmdet3d dv 是9
    int num_points_at_this_coors = pillar_count_histo[y_coor * grid_x_size + x_coor];

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 0] 
    =  dev_points[th_i * num_point_feature + 0]; 

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 1]  
    =  dev_points[th_i * num_point_feature + 1];

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 2]  
    =  dev_points[th_i * num_point_feature + 2];
    
    dev_pfe_gather_feature_[pointidx * num_gather_feature + 3]  =  0.0f;

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 4]
    =  dev_points[th_i * num_point_feature + 0] - dev_points_sum_x[y_coor * grid_x_size + x_coor] / num_points_at_this_coors;

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 5]
    =  dev_points[th_i * num_point_feature + 1] - dev_points_sum_y[y_coor * grid_x_size + x_coor] / num_points_at_this_coors;

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 6]
    =  dev_points[th_i * num_point_feature + 2] - dev_points_sum_z[y_coor * grid_x_size + x_coor] / num_points_at_this_coors;

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 7]
    =  dev_points[th_i * num_point_feature + 0] - (x_coor * pillar_x_size + (pillar_x_size/2 + min_x_range));

    dev_pfe_gather_feature_[pointidx * num_gather_feature + 8]
    =  dev_points[th_i * num_point_feature + 1] - (y_coor * pillar_y_size + (pillar_y_size/2 + min_y_range));
  }  
}

__global__ void make_pillar_index_kernel(int* pillar_count_histo, int* dev_counter, const int grid_x_size,
                                        int* dev_coor_to_pillaridx, int* dev_num_points_per_pillar) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int num_points_at_this_pillar = pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomicAdd(dev_counter, 1);
  dev_num_points_per_pillar[count] = num_points_at_this_pillar;
  dev_coor_to_pillaridx[y * grid_x_size + x] = count;
}

__global__ void  make_pfn_pillar_scatter(const float* pfn_feature, int* dev_points_coors, int* pillar_count_histo, const int grid_x_size, int* dev_coor_to_pillaridx_, float* dev_pillar_pfn_feature_scatter_, int* dev_pillar_coors_x, int* dev_pillar_coors_y)
{
  int ith_point  = blockIdx.x;
  int axis = threadIdx.x;
  int x_coor = dev_points_coors[ith_point * 4 + 3];
  int y_coor = dev_points_coors[ith_point * 4 + 2];
  int num_points_at_this_pillar = pillar_count_histo[y_coor * grid_x_size + x_coor];
  if (num_points_at_this_pillar == 0) {
    return;
  }
  int ith_pillar = dev_coor_to_pillaridx_[y_coor * grid_x_size + x_coor];
  dev_pillar_coors_y[ith_pillar] = y_coor;
  dev_pillar_coors_x[ith_pillar] = x_coor;
  // mean 
  // dev_pillar_pfn_feature_scatter_[ith_pillar*PFNDIM + axis] += pfn_feature[ith_point*PFNDIM + axis] / num_points_at_this_pillar;
  
  //max 
  if (dev_pillar_pfn_feature_scatter_[ith_pillar*PFNDIM + axis] < pfn_feature[ith_point*PFNDIM + axis]) {
    dev_pillar_pfn_feature_scatter_[ith_pillar*PFNDIM + axis] = pfn_feature[ith_point*PFNDIM + axis];
  }
}

PreprocessPointsCuda::PreprocessPointsCuda(
    const int num_threads,
    const int num_point_feature,
    const int grid_x_size, const int grid_y_size,
    const int grid_z_size, const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const float min_x_range, const float min_y_range,
    const float min_z_range)
    : num_threads_(num_threads),
      num_point_feature_(num_point_feature),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_histo_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_sum_x_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_sum_y_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_sum_z_),
        grid_y_size_ * grid_x_size_ * sizeof(int))); 
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_coor_to_pillaridx_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_num_points_per_pillar_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_counter_), sizeof(int)));
    }

PreprocessPointsCuda::~PreprocessPointsCuda() {
    GPU_CHECK(cudaFree(dev_pillar_count_histo_));
    GPU_CHECK(cudaFree(dev_points_sum_x_));
    GPU_CHECK(cudaFree(dev_points_sum_y_));
    GPU_CHECK(cudaFree(dev_points_sum_z_));
    GPU_CHECK(cudaFree(dev_counter_));
    GPU_CHECK(cudaFree(dev_coor_to_pillaridx_));
    GPU_CHECK(cudaFree(dev_num_points_per_pillar_));
  }

void PreprocessPointsCuda::DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points, int* dev_ith_to_pointidx, float* dev_pfe_gather_feature, int* dev_points_coors) {
    int num_block = DIVUP(in_num_points , num_threads_);
    gather_point_feature_kernel_coors<<<num_block , num_threads_>>>(
      dev_points, num_point_feature_, in_num_points,
      min_x_range_, min_y_range_, min_z_range_, 
      pillar_x_size_, pillar_y_size_, pillar_z_size_, 
      grid_x_size_, grid_y_size_, grid_z_size_,
      dev_points_sum_x_, dev_points_sum_y_, dev_points_sum_z_,
      dev_pfe_gather_feature, dev_points_coors, dev_pillar_count_histo_, dev_ith_to_pointidx);
}

void PreprocessPointsCuda::Rangefilter(const float* dev_points, const int in_num_points, int* valid_num_point, int* dev_ith_to_pointidx)
{
  // initialize paraments
  GPU_CHECK(cudaMemset(dev_pillar_count_histo_, 0 , grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_points_sum_x_, 0 , grid_y_size_ * grid_x_size_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_points_sum_y_, 0 , grid_y_size_ * grid_x_size_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_points_sum_z_, 0 , grid_y_size_ * grid_x_size_ * sizeof(float)));

  int num_block = DIVUP(in_num_points , num_threads_);
  make_pillar_histo_kernel_mean_coors<<<num_block , num_threads_>>>(
      dev_points, dev_points_sum_x_, dev_points_sum_y_, dev_points_sum_z_,
      dev_pillar_count_histo_, dev_ith_to_pointidx, in_num_points, grid_x_size_, grid_y_size_,
      grid_z_size_, min_x_range_, min_y_range_, min_z_range_, pillar_x_size_,
      pillar_y_size_, pillar_z_size_, num_point_feature_, valid_num_point);
}

void PreprocessPointsCuda::GetPillarCount(int* host_pillar_count)
{
  GPU_CHECK(cudaMemset(dev_coor_to_pillaridx_, 0 , grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0 , grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_counter_, 0, sizeof(int)));
  make_pillar_index_kernel<<<grid_x_size_ , grid_y_size_>>>(
    dev_pillar_count_histo_, dev_counter_, grid_x_size_, dev_coor_to_pillaridx_, dev_num_points_per_pillar_);
  
  GPU_CHECK(cudaMemcpy(host_pillar_count, dev_counter_, 1 * sizeof(int),
      cudaMemcpyDeviceToHost));
}

void PreprocessPointsCuda::PfnScatter(const float* pfn_feature, int* dev_points_coors, float* dev_pillar_pfn_feature_scatter_, int* dev_pillar_coors_x, int* dev_pillar_coors_y, int in_num_points)
{
  // int num_block = DIVUP(in_num_points , num_threads_);
  // dim3 mean_block(num_threads_, PFNDIM);
  make_pfn_pillar_scatter<<<in_num_points, PFNDIM>>>(
    pfn_feature, dev_points_coors, dev_pillar_count_histo_, grid_x_size_, dev_coor_to_pillaridx_, dev_pillar_pfn_feature_scatter_, dev_pillar_coors_x, dev_pillar_coors_y);
}