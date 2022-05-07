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
#include "pointpillars.h"

#include <chrono>
#include <iostream>
#include <cstring>

#define PFNDIM 64
#define ANCHOR_NUM 560000   // feature_map 400x400
// #define ANCHOR_NUM 140000   // feature_map 200x200

void PointPillars::InitParams()
{
    YAML::Node params = YAML::LoadFile(pp_config_);
    kPillarXSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
    kPillarYSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
    kPillarZSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
    kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
    kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
    kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
    kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
    kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
    kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
    kNumClass = params["CLASS_NAMES"].size();
    kMaxNumPillars = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"].as<int>();
    kMaxNumPointsPerPillar = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"].as<int>();
    // kNumPointFeature = 5; // 输入点云都是5 [x, y, z, i,0] multihead_pp 是5
    kNumPointFeature = 4; // [x, y, z, i] mmdet3d 是4
    kNumAnchorSize = 9;
    kNumInputBoxFeature = 7;
    kNumOutputBoxFeature = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["BOX_CODER_CONFIG"]["code_size"].as<int>();
    kBatchSize = 1;
    kNumThreads = 64;
    kNumBoxCorners = 8;
    kNmsPreMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"].as<int>();
    kNmsPostMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"].as<int>();

    // Generate secondary parameters based on above.
    kGridXSize = static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize); //400 200
    kGridYSize = static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize); //400 200
    kGridZSize = static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize); //1
    kRpnInputSize = PFNDIM * kGridYSize * kGridXSize;
}


PointPillars::PointPillars(const float score_threshold,
                           const float nms_overlap_threshold,
                           const bool use_onnx,
                           const std::string pfe_file,
                           const std::string backbone_file,
                           const std::string pp_config)
    : score_threshold_(score_threshold),
      nms_overlap_threshold_(nms_overlap_threshold),
      use_onnx_(use_onnx),
      pfe_file_(pfe_file),
      backbone_file_(backbone_file),
      pp_config_(pp_config)
{
    InitParams();
    InitTRT(use_onnx_);
    DeviceMemoryMalloc();

    // cudaStream_t stream = NULL;
    // checkCudaErrors(cudaStreamCreate(&stream));
    // TRT(pfe_file_, stream);

    preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
        kNumThreads,
        kNumPointFeature,
        kGridXSize,kGridYSize, kGridZSize,
        kPillarXSize,kPillarYSize, kPillarZSize,
        kMinXRange, kMinYRange, kMinZRange));

    scatter_cuda_ptr_.reset(new ScatterCuda(kNumThreads, kGridXSize, kGridYSize));

    const float float_min = std::numeric_limits<float>::lowest();
    const float float_max = std::numeric_limits<float>::max();
    postprocess_cuda_ptr_.reset(
      new PostprocessCuda(kNumThreads,
                          float_min, float_max, 
                          kNumClass,kNmsPreMaxsize,
                          score_threshold_, 
                          nms_overlap_threshold_,
                          kNmsPreMaxsize, 
                          kNmsPostMaxsize,
                          kNumBoxCorners, 
                          kNumInputBoxFeature,
                          kNumOutputBoxFeature));  /*kNumOutputBoxFeature*/
    
}

void PointPillars::DeviceMemoryMalloc() {
    // for pillars 
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&host_pillar_count_), 1 * sizeof(int)));
    // for sparse map

    // for trt inference
    // create GPU buffers and a stream

    GPU_CHECK(cudaMalloc(&rpn_buffers_[0],  (kRpnInputSize + ANCHOR_NUM * kNumAnchorSize) * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[3],  kNmsPreMaxsize * 9 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[1],  kNmsPreMaxsize * 10 * sizeof(float)));
    GPU_CHECK(cudaMalloc(&rpn_buffers_[2],  kNmsPreMaxsize * sizeof(int)));

    // for scatter kernel
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_scattered_feature_),
                        kNumThreads * kGridYSize * kGridXSize * sizeof(float)));

    // for host_pillar_count_
    host_pillar_count_ = (int *)malloc(sizeof(int));

    // for filter
    host_box_ =  new float[kNmsPreMaxsize * kNumClass * kNumOutputBoxFeature]();
    host_score_ =  new float[kNmsPreMaxsize * kNumClass * 18]();
    host_filtered_count_ = new int[kNumClass]();
}


PointPillars::~PointPillars() {
    // for pillars 
    GPU_CHECK(cudaFree(rpn_buffers_[0]));
    GPU_CHECK(cudaFree(rpn_buffers_[1]));
    GPU_CHECK(cudaFree(rpn_buffers_[2]));
    GPU_CHECK(cudaFree(rpn_buffers_[3]));

    free(host_pillar_count_);

    pfe_context_->destroy();
    backbone_context_->destroy();
    pfe_engine_->destroy();
    backbone_engine_->destroy();
    // for post process
    GPU_CHECK(cudaFree(dev_scattered_feature_));
    delete[] host_box_;
    delete[] host_score_;
    delete[] host_filtered_count_;
}



void PointPillars::SetDeviceMemoryToZero() {
    memset(host_pillar_count_, 0, sizeof(int));
    GPU_CHECK(cudaMemset(rpn_buffers_[0],       0, (kRpnInputSize + ANCHOR_NUM * kNumAnchorSize) * sizeof(float)));
    GPU_CHECK(cudaMemset(rpn_buffers_[3],       0, kNmsPreMaxsize * 9 * sizeof(float)));
    GPU_CHECK(cudaMemset(rpn_buffers_[1],       0, kNmsPreMaxsize * 10 * sizeof(float)));
    GPU_CHECK(cudaMemset(rpn_buffers_[2],       0, kNmsPreMaxsize * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_scattered_feature_,    0, kNumThreads * kGridYSize * kGridXSize * sizeof(float)));
}


void PointPillars::InitTRT(const bool use_onnx) {
    OnnxToEngine(pfe_file_, &pfe_engine_);
    EngineToTRTModel(backbone_file_, &backbone_engine_);

    if (pfe_engine_ == nullptr || backbone_engine_ == nullptr) {
        std::cerr << "Failed to load ONNX file.";
    }

    // create execution context from the engine
    pfe_context_ = pfe_engine_->createExecutionContext();
    backbone_context_ = backbone_engine_->createExecutionContext();
    if (pfe_context_ == nullptr || backbone_context_ == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context.";
    }
}

void PointPillars::OnnxToEngine(
    const std::string& model_file,  // name of the onnx model
    nvinfer1::ICudaEngine** engine_ptr)
{
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition* network =
        builder->createNetworkV2(explicit_batch);

    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
#if defined (__arm64__) || defined (__aarch64__) 
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "Enable fp16!" << std::endl;
#endif
    config->setMaxWorkspaceSize(1 << 30);
    auto input = network->getInput(0);
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{50000,9});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{70000,9});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{100000,9});
    config->addOptimizationProfile(profile);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_); 
    if (runtime == nullptr) {
        std::string msg("failed to build runtime parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (engine == nullptr) {
        std::string msg("failed to build engine parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    *engine_ptr = engine;
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
}

void PointPillars::OnnxToTRTModel(
    const std::string& model_file,  // name of the onnx model
    nvinfer1::ICudaEngine** engine_ptr) {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition* network =
        builder->createNetworkV2(explicit_batch);

    // parse onnx model
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 25);
    nvinfer1::ICudaEngine* engine =
        builder->buildEngineWithConfig(*network, *config);

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}

int PointPillars::setDynamicInputDimensions(const std::string &input_name,
                                            const int in_num_points,
                                            nvinfer1::ICudaEngine *engine,
                                            nvinfer1::IExecutionContext *context)
{
    int input_idx = engine->getBindingIndex(input_name.c_str());
    if (!engine->hasImplicitBatchDimension() && engine->getNbOptimizationProfiles() > 0) {
        context->setOptimizationProfile(0);
        auto in_dims = engine->getBindingDimensions(input_idx);
        context->setBindingDimensions(input_idx, nvinfer1::Dims2(in_num_points, in_dims.d[1]));
        if (!context->allInputDimensionsSpecified())
            std::cerr << "InputDimensionsSpecified Failed!" << std::endl;
    }
    return input_idx;
}

void PointPillars::EngineToTRTModel(
    const std::string &engine_file ,     
    nvinfer1::ICudaEngine** engine_ptr)  {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream gieModelStream; 
    gieModelStream.seekg(0, gieModelStream.beg); 

    std::ifstream cache(engine_file); 
    gieModelStream << cache.rdbuf();
    cache.close(); 
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_); 

    if (runtime == nullptr) {
        std::string msg("failed to build runtime parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg(); 

    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize); 
    gieModelStream.read((char*)modelMem, modelSize);


    std::cout << " |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> "<< std::endl;
    std::cout << " | " << engine_file << " >" <<  std::endl;
    std::cout << " |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> "<< std::endl;
    std::cout << "             (\\__/) ||                 "<< std::endl;
    std::cout << "             (•ㅅ•) ||                 "<< std::endl;
    std::cout << "             / 　 づ                    "<< std::endl;
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL); 
    if (engine == nullptr) {
        std::string msg("failed to build engine parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    *engine_ptr = engine;

    for (int bi = 0; bi < engine->getNbBindings(); bi++)
    {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
    }
}

void PointPillars::DoInference(const float* in_points_array,
                                const int in_num_points,
                                const float* in_anchor_points,
                                std::vector<float>* out_detections,
                                std::vector<int>* out_labels,
                                std::vector<float>* out_scores) 
{
    SetDeviceMemoryToZero();
    cudaDeviceSynchronize();

    // [STEP 1] : load pointcloud and anchors
    auto load_start = std::chrono::high_resolution_clock::now();

    float* dev_points;
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points),
                        in_num_points * kNumPointFeature * sizeof(float))); // in_num_points , kNumPointFeature
    GPU_CHECK(cudaMemset(dev_points, 0, in_num_points * kNumPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dev_points, in_points_array,
                        in_num_points * kNumPointFeature * sizeof(float),
                        cudaMemcpyHostToDevice));

    float* dev_anchors;
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors),
                        ANCHOR_NUM * kNumAnchorSize * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_anchors, 0, ANCHOR_NUM * kNumAnchorSize * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dev_anchors, in_anchor_points,
                        ANCHOR_NUM * kNumAnchorSize * sizeof(float),
                        cudaMemcpyHostToDevice));


    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&valid_num_points), sizeof(int)));
    GPU_CHECK(cudaMemset(valid_num_points,    0, sizeof(int)));
    
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ith_to_pointidx_),
                        in_num_points * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_ith_to_pointidx_,    -1, in_num_points * sizeof(int)));
    auto load_end = std::chrono::high_resolution_clock::now();

    // [STEP 2] : preprocess
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    preprocess_points_cuda_ptr_->Rangefilter(dev_points, in_num_points, valid_num_points, dev_ith_to_pointidx_);
    cudaDeviceSynchronize();
    // DEVICE_SAVE<int>(dev_ith_to_pointidx_,  in_num_points  , "0_dev_ith_to_pointidx_");

    int filter_num_points[1];
    GPU_CHECK(cudaMemcpy(filter_num_points, valid_num_points, 1 * sizeof(int),
        cudaMemcpyDeviceToHost));

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pfe_gather_feature_),
                        filter_num_points[0] * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pfe_gather_feature_,    0, filter_num_points[0] * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_coors_),
                        filter_num_points[0] * 4 * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_points_coors_,    0, filter_num_points[0] * 4 * sizeof(int)));

    preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(dev_points, in_num_points, dev_ith_to_pointidx_, dev_pfe_gather_feature_, dev_points_coors_);
    cudaDeviceSynchronize();
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    // DEVICE_SAVE<float>(dev_pfe_gather_feature_,  filter_num_points[0] * kNumGatherPointFeature  , "0_Model_pfe_input_gather_feature");
    // DEVICE_SAVE<int>(dev_points_coors_,  filter_num_points[0] * 4  , "0_dev_points_coors");

    // [STEP 3] : pfe forward
    cudaStream_t stream;
    GPU_CHECK(cudaStreamCreate(&stream));
    auto pfe_start = std::chrono::high_resolution_clock::now();
    int input_index = setDynamicInputDimensions("input", filter_num_points[0], pfe_engine_, pfe_context_);
    
    GPU_CHECK(cudaMalloc(&pfe_buffers_[0], filter_num_points[0] * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMalloc(&pfe_buffers_[1], filter_num_points[0] * PFNDIM *sizeof(float)));
    GPU_CHECK(cudaMemset(pfe_buffers_[0],       0, filter_num_points[0] * kNumGatherPointFeature * sizeof(float)));
    GPU_CHECK(cudaMemset(pfe_buffers_[1],       0, filter_num_points[0] * PFNDIM * sizeof(float)));

    GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[0], dev_pfe_gather_feature_,
                            filter_num_points[0] * kNumGatherPointFeature * sizeof(float), ///kNumGatherPointFeature
                            cudaMemcpyDeviceToDevice, stream));
    pfe_context_->enqueueV2(pfe_buffers_, stream, nullptr);
    cudaDeviceSynchronize();
    auto pfe_end = std::chrono::high_resolution_clock::now();
    // DEVICE_SAVE<float>(reinterpret_cast<float*>(pfe_buffers_[1]),  filter_num_points[0] * PFNDIM , "1_Model_pfe_output_buffers_[1]");

    // [STEP 4] : scatter pillar feature
    auto scatter_start = std::chrono::high_resolution_clock::now();
    preprocess_points_cuda_ptr_->GetPillarCount(host_pillar_count_);
    cudaDeviceSynchronize();

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_pfn_feature_scatter_),
      host_pillar_count_[0] * PFNDIM * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pillar_pfn_feature_scatter_, 0, host_pillar_count_[0] * PFNDIM * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_coors_x_),host_pillar_count_[0] * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_coors_y_),host_pillar_count_[0] * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pillar_coors_x_, 0, host_pillar_count_[0] * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_pillar_coors_y_, 0, host_pillar_count_[0] * sizeof(int)));
    preprocess_points_cuda_ptr_->PfnScatter(reinterpret_cast<float*>(pfe_buffers_[1]), dev_points_coors_, dev_pillar_pfn_feature_scatter_, dev_pillar_coors_x_, dev_pillar_coors_y_, filter_num_points[0]);
    cudaDeviceSynchronize();
    // DEVICE_SAVE<float>(dev_pillar_pfn_feature_scatter_,  host_pillar_count_[0] * PFNDIM , "1_pfn_scatter_buffers_[2]");
    // DEVICE_SAVE<int>(dev_pillar_coors_x_,  host_pillar_count_[0] , "1_pfn_pillar_coors_x_[2]");
    // DEVICE_SAVE<int>(dev_pillar_coors_y_,  host_pillar_count_[0] , "1_pfn_pillar_coors_y_[2]");

    scatter_cuda_ptr_->DoScatterCuda(
        host_pillar_count_[0], dev_pillar_coors_x_, dev_pillar_coors_y_,
        dev_pillar_pfn_feature_scatter_, dev_scattered_feature_);
    cudaDeviceSynchronize();
    auto scatter_end = std::chrono::high_resolution_clock::now();
    // DEVICE_SAVE<float>(dev_scattered_feature_ ,  kRpnInputSize,"2_Model_backbone_input_dev_scattered_feature");

    // [STEP 5] : backbone forward
    auto backbone_start = std::chrono::high_resolution_clock::now();
    GPU_CHECK(cudaMemcpyAsync(rpn_buffers_[0], dev_scattered_feature_,
                            kBatchSize * kRpnInputSize * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    GPU_CHECK(cudaMemcpyAsync((uint8_t *)rpn_buffers_[0] + kBatchSize * kRpnInputSize * sizeof(float), dev_anchors,
                            ANCHOR_NUM * kNumAnchorSize * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    backbone_context_->enqueueV2(rpn_buffers_, stream, nullptr);
    cudaDeviceSynchronize();
    auto backbone_end = std::chrono::high_resolution_clock::now();

    // [STEP 6]: postprocess (multihead)
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    postprocess_cuda_ptr_->DoPostprocessCuda(
        reinterpret_cast<float*>(rpn_buffers_[3]), // [box]
        reinterpret_cast<float*>(rpn_buffers_[1]), // [score]
        host_box_, 
        host_score_, 
        host_filtered_count_,
        *out_detections, *out_labels , *out_scores);
    cudaDeviceSynchronize();
    auto postprocess_end = std::chrono::high_resolution_clock::now();

    // release the stream and the buffers
    std::chrono::duration<double> load_cost = load_end - load_start;
    std::chrono::duration<double> preprocess_cost = preprocess_end - preprocess_start;
    std::chrono::duration<double> pfe_cost = pfe_end - pfe_start;
    std::chrono::duration<double> scatter_cost = scatter_end - scatter_start;
    std::chrono::duration<double> backbone_cost = backbone_end - backbone_start;
    std::chrono::duration<double> postprocess_cost = postprocess_end - postprocess_start;

    std::chrono::duration<double> pointpillars_cost = postprocess_end - preprocess_start;
    std::cout << "------------------------------------" << std::endl;
    std::cout << setiosflags(ios::left)  << setw(14) << "Module" << setw(12)  << "Time"  << resetiosflags(ios::left) << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::string Modules[] = {"Load", "Preprocess" , "Pfe" , "Scatter" , "Backbone" , "Postprocess" , "Summary"};
    double Times[] = {load_cost.count(), preprocess_cost.count() , pfe_cost.count() , scatter_cost.count() , backbone_cost.count() , postprocess_cost.count() , pointpillars_cost.count()}; 

    for (int i =0 ; i < 7 ; ++i) {
        std::cout << setiosflags(ios::left) << setw(14) << Modules[i]  << setw(8)  << Times[i] * 1000 << " ms" << resetiosflags(ios::left) << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;

    cudaStreamDestroy(stream);
    GPU_CHECK(cudaFree(dev_points));
    GPU_CHECK(cudaFree(dev_anchors));
    GPU_CHECK(cudaFree(valid_num_points));
    GPU_CHECK(cudaFree(dev_ith_to_pointidx_));
    // for pfe forward
    GPU_CHECK(cudaFree(dev_pfe_gather_feature_));
    GPU_CHECK(cudaFree(dev_points_coors_));
    
    GPU_CHECK(cudaFree(pfe_buffers_[0]));
    GPU_CHECK(cudaFree(pfe_buffers_[1]));
    GPU_CHECK(cudaFree(dev_pillar_pfn_feature_scatter_));
    GPU_CHECK(cudaFree(dev_pillar_coors_x_));
    GPU_CHECK(cudaFree(dev_pillar_coors_y_));
}
