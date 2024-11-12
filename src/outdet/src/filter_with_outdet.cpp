#include <torch/torch.h>
#include <iostream>
#include "outdet/outdet.h"
#include "outdet/util.h"
#include<string>
#include<iomanip>
#include<vector>
#include<fstream>
#include<tuple>
#include<chrono>
#include<cuda_runtime.h>
#include<dlfcn.h>
#include<filesystem>
#include<stdlib.h>
#include <assert.h>
#include <filesystem>
#include "outdet/knncuda.h"
#include <torch/script.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include<cuda_runtime.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include <cmath>
#define K 9
#define RUN_TORCH_CODE false
using namespace nvinfer1;
using namespace nvonnxparser;
class Logger : public ILogger{
    void log(Severity severity, const char* msg) noexcept override{
            if (severity <= Severity::kWARNING){
                std::cout << msg << std::endl;
            }
    }
} logger;
using namespace torch::indexing;
using namespace std::filesystem;


struct InferDeleter{
    template<typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};
template <typename T>
using TRTUniquePointer = std::unique_ptr<T, InferDeleter>;
void filter_snow(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // to advertise the filtered cloud
    ros::NodeHandle n;
    // topic name
    std::string topic = n.resolveName("desnowed_cloud");
    // publisher
    ros::Publisher cloud_pub = n.advertise<sensor_msgs::PointCloud2>(topic, 10);
    // create point clouds
    sensor_msgs::PointCloud2 out_msg;
    out_msg.is_dense = false;
    out_msg.is_bigendian = false;
    out_msg.fields.resize(4);
    out_msg.fields[0].name = "x";
    out_msg.fields[0].offset = 0;
    out_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    out_msg.fields[0].count = 1;

    out_msg.fields[1].name = "y";
    out_msg.fields[1].offset = 4;
    out_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    out_msg.fields[1].count = 1;

    out_msg.fields[2].name = "z";
    out_msg.fields[2].offset = 8;
    out_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    out_msg.fields[2].count = 1;

    out_msg.fields[3].name = "i";
    out_msg.fields[3].offset = 12;
    out_msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    out_msg.fields[3].count = 1;

    out_msg.point_step = 16; // size of point in bytes
    // run at 10Hz
    ros::Rate loop_rate(10);

    //
    int w = msg->width;
    int h = msg->height;
    int num_points = w * h;
    float points[num_points][4];
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> iter_i(*msg, "i");
    std::vector<std::array<float, 4>> front_view;
    float fov = 3.1415  / 4.0;
    for (int a = 0; a < num_points; a++, ++iter_x, ++iter_y, ++iter_z, ++iter_i){
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;
            float sq_dist = x * x + y * y;
            float angle = atan2(sqrt(y * y + z * z), x);
            float inten = *iter_i;

            if (sq_dist < 100 && angle < fov){
                front_view.push_back({x, y, z, inten});
            }
    }
    torch::Device device(torch::kCUDA, 0);
    num_points = front_view.size();
    auto mean = torch::tensor({0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928}, torch::kFloat32).unsqueeze(0);
    auto stddev = torch::tensor({25.845459,  18.93466,    1.5863657, 14.734034}, torch::kFloat32).unsqueeze(0);
    auto pt_tensor = torch::from_blob(front_view.data(), {num_points, 4}, torch::TensorOptions().dtype(torch::kFloat));
    torch::Tensor pt_xyz = pt_tensor.index({Slice(), Slice(None, 3)});
//    int index = 10;
//    ROS_INFO("Total Points : [%f %f %f %f]", front_view[index][0], front_view[index][1], front_view[index][2], front_view[index][3]);

    float *selected_data_arr = pt_tensor.data_ptr<float>();

    int num_feat = 3;
    int batch_size = num_points;
    mean = mean.repeat({batch_size, 1});
    stddev = stddev.repeat({batch_size, 1});
    float * selected_xyz = pt_xyz.data_ptr<float>();
    float *ref = (float *)malloc(batch_size * num_feat * sizeof(float));

    memcpy(ref, selected_xyz, batch_size * num_feat * sizeof(float));
//    std::cout << front_view.size() << std::endl;

    float *query = (float * )malloc(batch_size * num_feat * sizeof(float ));
    memcpy(query, selected_xyz, batch_size * num_feat * sizeof(float ));

    float *knn_dist = (float *)malloc(batch_size * batch_size * sizeof(float ));
    kNN_dist(ref, batch_size, query, batch_size, num_feat, knn_dist);

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor dist_tensor = torch::from_blob(knn_dist, {batch_size, batch_size}, options);
    dist_tensor = dist_tensor.to(device);
    auto [dist, ind] = dist_tensor.topk(K, 1, false, true);
    auto inp = pt_tensor.sub(mean).div(stddev);
    dist = torch::sqrt(dist) + 1.0;
    if (RUN_TORCH_CODE){
        OutDet model(2, 1, 3, 4, 32);
        model->to(device);
        load_cpp_weights(model);
        torch::NoGradGuard no_grad;
        model->eval();
        auto out = model->forward(inp.to(device), dist.to(device), ind.to(device));
        out = out.argmax(1);
        out = out.contiguous();
        out = out.to(torch::kCPU);
        long int * pred = out.data_ptr< long int>();
        int counter = 0;
        for (int i = 0; i < out.numel(); i++){
            if (pred[i] == 1){
                counter++;
            }
        }
        std::cout << "Number of snow points detected " << counter << std::endl;
    }
    // tensorRT code
    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    IParser* parser = createParser(*network, logger);
    std::ifstream file("/var/local/home/aburai/test_catkin/src/outdet/saved_models/outdet.onnx", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (! file.read(buffer.data(), size)){
        std::cout << "Error"<< std::endl;
        throw std::runtime_error("Cannot read file");
    }
    auto success = parser->parse(buffer.data(), buffer.size());
    if (!success){
        throw std::runtime_error("failed to parse model");
    }
    const auto numInputs  = network->getNbInputs();
//    std::cout << "Number of inputs: " << numInputs << std::endl;
    const auto input0batch = network->getInput(0)->getDimensions().d[0];
//    std::cout << "Batch Size: " << input0batch << std::endl;
    // create build config
    auto conf_succes = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!conf_succes){
        throw std::runtime_error("Cannot create build config");
    }
    // register single optimization profile
    IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    for (int32_t i =0; i < numInputs; i++){
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputF = inputDims.d[1];
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims2(1, inputF));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims2(250000, inputF));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims2(30000, inputF));
    }
    conf_succes->addOptimizationProfile(optProfile);
    // use default precision
    cudaStream_t profileStream;
    cudaStreamCreate(&profileStream);
    conf_succes->setProfileStream(profileStream);

//        IBuilderConfig* config = builder->createBuilderConfig();
    conf_succes->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1024*1024*1024);
    conf_succes->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 1024*1024*1024);
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *conf_succes)};
//        IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *conf_succes);

    // write engine to disk
    const auto enginePath = "/var/local/home/aburai/test_catkin/src/outdet/saved_models/outdet.trt";
    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

    std::shared_ptr<IRuntime> mRuntime;
    std::shared_ptr<ICudaEngine> mEngine;
    mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(logger));
    mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    // create context
//        std::shared_ptr<IExecutionContext> context
    TRTUniquePointer<IExecutionContext> context(mEngine->createExecutionContext(), InferDeleter());
    // create buffer object
    std::vector<void *> m_buffers;
    std::vector<int32_t > m_outputLengths{};
    std::vector<int32_t> m_inputDims;
    std::vector<int32_t> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
//        BufferManager buffer(mEngine);
    cudaStream_t stream;
    m_buffers.resize(mEngine->getNbIOTensors());
    auto err = cudaStreamCreate(&stream);
    // first input
    int32_t m_inputBatchSize;
    int32_t  maxBatchSize =250000;
    Dims2 inp1Dims = {batch_size, 4};
    for (int i=0; i< mEngine->getNbIOTensors(); i++){
        const auto tensorName = mEngine->getIOTensorName(i);
//            std::cout << tensorName << std::endl;
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = mEngine->getTensorIOMode(tensorName);
        const auto tensorShape = mEngine->getTensorShape(tensorName);
        const auto tensorDataType = mEngine->getTensorDataType(tensorName);

        if (tensorType == TensorIOMode::kINPUT){
            m_inputDims.emplace_back(tensorShape.d[1]);
            m_inputBatchSize = tensorShape.d[0];
//            std::cout << tensorShape.d[1] << std::endl;

        }
        else if (tensorType == TensorIOMode::kOUTPUT){
            if (tensorDataType == DataType::kFLOAT){
//                std::cout << "float out " <<tensorShape.d[1] << std::endl;
                m_outputDims.push_back(tensorShape.d[1]);
                m_outputLengths.push_back(tensorShape.d[1]);
                auto err = cudaMallocAsync(&m_buffers[i], tensorShape.d[1] * maxBatchSize * sizeof(float), stream);
                if (err != cudaSuccess){
                    std::cout << err << std::endl;
                }
            }
        }
    }


    // destroy cuda steam at the end
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
//        const auto batchSize = -1;
    Dims2 inp1Shape = Dims2(batch_size, 4);
    Dims2 inp2Shape = Dims2(batch_size, 9);
    Dims2 inp3Shape = Dims2(batch_size, 9);
    context->setInputShape("points", inp1Dims);
    context->setInputShape("dist", inp2Shape);
    context->setInputShape("indices", inp3Shape);
    m_buffers[0] = (void *)inp.to(device).data_ptr();
    m_buffers[1] = (void *)dist.to(device).data_ptr();
    m_buffers[2] = (void *)ind.to(device).data_ptr();
    // set the address of input and output buffer
    context->setTensorAddress("points", m_buffers[0]);
    context->setTensorAddress("dist", m_buffers[1]);
    context->setTensorAddress("indices", m_buffers[2]);
    context->setTensorAddress("out", m_buffers[3]);
    // infer
    auto status = context->enqueueV3(profileStream);
    float *rt_out = (float *)malloc(sizeof(float) * batch_size * 2);
    cudaMemcpyAsync(rt_out, m_buffers[3], sizeof(float) * batch_size * 2, cudaMemcpyDeviceToHost);
    torch::Tensor tensor_out = torch::from_blob(rt_out, {batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat));
    cudaStreamDestroy(profileStream);

    auto out = tensor_out.argmax(1);
    out = out.contiguous();
    out = out.to(torch::kCPU);
    long int * pred = out.data_ptr< long int>();
    int counter = 0;
    std::vector<std::array<float, 4>> filtered;
    for (int i = 0; i < out.numel(); i++){
        if (pred[i] == 1){
            counter++;
        }
        else{
            filtered.push_back(front_view[i]);
        }
    }
    ROS_INFO("Total Snow Points detected: [%d]", counter);
    // advertise the desnowed clouds
    int count = 0;

        // fill the message object with data
        int width = 1;
        int height = counter;
        int num_points_out = width * height;
        out_msg.row_step = width * out_msg.point_step;
        out_msg.height = height;
        out_msg.width = width;
        out_msg.data.resize(num_points_out * out_msg.point_step);
        int pt_counter = 0;
        for (int x = 0; x < width; x++){
            for (int y=0; y< height; y++){
                uint8_t *ptr = &out_msg.data[0] + (x + y * width) * out_msg.point_step;
                *(float *)ptr = filtered[pt_counter][0];  //x
                ptr += 4;
                *(float *)ptr = filtered[pt_counter][1]; //y
                ptr += 4;
                *(float *)ptr = filtered[pt_counter][2]; //z
                ptr += 4;
                *(float *)ptr = filtered[pt_counter][3];
                pt_counter += 1;
            }
        }
        out_msg.header.seq = count;
        out_msg.header.stamp = ros::Time::now();

        // PUBLISH  the point cloud msg
        cloud_pub.publish(out_msg);
        // unlike spin(), spinOnce will process the event and return so next message can be published
//        ros::spinOnce();
        // time control
//        loop_rate.sleep();
//        ++count;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "desnower");
    // handler for point subscriber node
    ros::NodeHandle n;
    // topic name
    std::string topic = n.resolveName("input_cloud");
    // subscriber
    ros::Subscriber sub = n.subscribe(topic, 10, filter_snow);

    // enter into loop
    ros::spin();
    // wait till roscore dies
    return 0;
}
