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
 * @file preprocess_points_cuda.h
 * @brief GPU version of preprocess points
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
#pragma once


class PreprocessPointsCuda {
 private:
    // initializer list
    const int num_threads_;
    const int num_point_feature_;
    const int grid_x_size_;
    const int grid_y_size_;
    const int grid_z_size_;
    const float pillar_x_size_;
    const float pillar_y_size_;
    const float pillar_z_size_;
    const float min_x_range_;
    const float min_y_range_;
    const float min_z_range_;
    // end initializer list

    float* dev_points_sum_x_;
    float* dev_points_sum_y_;
    float* dev_points_sum_z_;
    float* dev_pillar_pfn_feature_scatter_;
    int* dev_pillar_count_histo_;
    int* dev_counter_;
    int* dev_coor_to_pillaridx_;
    int* dev_num_points_per_pillar_;

 public:
  /**
   * @brief Constructor
   * @param[in] num_threads Number of threads when launching cuda kernel
   * @param[in] num_point_feature Number of features in a point
   * 
   * @param[in] grid_x_size Number of pillars in x-coordinate
   * @param[in] grid_y_size Number of pillars in y-coordinate
   * @param[in] grid_z_size Number of pillars in z-coordinate
   * 
   * @param[in] pillar_x_size Size of x-dimension for a pillar
   * @param[in] pillar_y_size Size of y-dimension for a pillar
   * @param[in] pillar_z_size Size of z-dimension for a pillar
   * 
   * @param[in] min_x_range Minimum x value for point cloud
   * @param[in] min_y_range Minimum y value for point cloud
   * @param[in] min_z_range Minimum z value for point cloud
   * @details Captital variables never change after the compile
   */
   PreprocessPointsCuda(const int num_threads,
                       const int num_point_feature,
                       const int grid_x_size, const int grid_y_size, const int grid_z_size,  // grid size
                       const float pillar_x_size, const float pillar_y_size, const float pillar_z_size, //voxel size
                       const float min_x_range, const float min_y_range, const float min_z_range); // point cloud range
  ~PreprocessPointsCuda();
   
   void DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points, int* valid_num_point, float* dev_pfe_gather_feature, int* dev_points_coors);
   void Rangefilter(const float* dev_points, const int in_num_points, int* valid_num_point, int* dev_ith_to_pointidx);
   void GetPillarCount(int* host_pillar_count);
   void PfnScatter(const float* pfn_feature, int* dev_points_coors, float* dev_pillar_pfn_feature_scatter_, int* dev_pillar_coors_x, int* dev_pillar_coors_y, int in_num_points);
};