//
// Created by aburai on 2024-11-11.
//
#include <torch/script.h>
#include "util.h"
#include "outdet.h"
bool load_cpp_weights(OutDet model){
// load model weights
    torch::jit::script::Module cpp_module = torch::jit::load("/var/local/home/aburai/outdet_cpp/cpp_weights.pt");
// nh conv layer 1
    assert(cpp_module.hasattr("convs.0.conv1.dw") == true && "Cannot read convs.0.conv1.dw");
    model->conv1->conv1->dw = cpp_module.attr("convs.0.conv1.dw").toTensor();
//    std::cout << model->conv1->conv1->dw << std::endl;
    assert(cpp_module.hasattr("convs.0.conv1.b") == true && "Cannot read convs.0.conv1.b");
    model->conv1->conv1->b = cpp_module.attr("convs.0.conv1.b").toTensor();
    assert(cpp_module.hasattr("convs.0.conv1.weight") == true && "Cannot read convs.0.conv1.weight");
    model->conv1->conv1->weight = cpp_module.attr("convs.0.conv1.weight").toTensor();
// batch norm 1
    assert(cpp_module.hasattr("convs.0.bn1.weight") == true && "Cannot read convs.0.bn1.weight");
    model->conv1->bn1->weight = cpp_module.attr("convs.0.bn1.weight").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.bias") == true && "Cannot read convs.0.bn1.bias");
    model->conv1->bn1->bias = cpp_module.attr("convs.0.bn1.bias").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.running_mean") == true && "Cannot read convs.0.bn1.running_mean");
    model->conv1->bn1->running_mean = cpp_module.attr("convs.0.bn1.running_mean").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.running_var") == true && "Cannot read convs.0.bn1.running_var");
    model->conv1->bn1->running_var = cpp_module.attr("convs.0.bn1.running_var").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.num_batches_tracked") == true && "Cannot read convs.0.bn1.num_batches_tracked");
    model->conv1->bn1->num_batches_tracked = cpp_module.attr("convs.0.bn1.num_batches_tracked").toTensor();

// nh conv layer 2
    assert(cpp_module.hasattr("convs.0.conv2.weight") == true && "Cannot read convs.0.conv2.weight");
    model->conv1->conv2->weight = cpp_module.attr("convs.0.conv2.weight").toTensor();

//batch norm 2
    assert(cpp_module.hasattr("convs.0.bn2.weight") == true && "Cannot read convs.0.bn2.weight");
    model->conv1->bn2->weight = cpp_module.attr("convs.0.bn2.weight").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.bias") == true && "Cannot read convs.0.bn2.bias");
    model->conv1->bn2->bias = cpp_module.attr("convs.0.bn2.bias").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.running_mean") == true && "Cannot read convs.0.bn2.running_mean");
    model->conv1->bn2->running_mean = cpp_module.attr("convs.0.bn2.running_mean").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.running_var") == true && "Cannot read convs.0.bn2.running_var");
    model->conv1->bn2->running_var = cpp_module.attr("convs.0.bn2.running_var").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.num_batches_tracked") == true && "Cannot read convs.0.bn2.num_batches_tracked");
    model->conv1->bn2->num_batches_tracked = cpp_module.attr("convs.0.bn2.num_batches_tracked").toTensor();

// residual connection
    assert(cpp_module.hasattr("convs.0.downsample.weight") == true && "Cannot read convs.0.downsample.weight");
    model->conv1->downsample->weight = cpp_module.attr("convs.0.downsample.weight").toTensor();

// linear layer
    assert(cpp_module.hasattr("fc.weight") == true && "Cannot read fc.weight");
    model->fc->weight = cpp_module.attr("fc.weight").toTensor();
    assert(cpp_module.hasattr("fc.bias") == true && "Cannot read fc.bias");
    model->fc->bias = cpp_module.attr("fc.bias").toTensor();

    return true;
}