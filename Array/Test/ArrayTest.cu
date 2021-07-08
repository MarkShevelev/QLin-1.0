#include <Device/DeviceMemory.h>
#include <Device/DeviceError.h>
#include <Device/Device.h>
#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace std;

__global__ void kernel_array_sum(
    iki::device::Array<float,1> const lha,
    iki::device::Array<float,1> const rha,
    iki::device::Array<float,1> res)
{
    int idx = threadIdx.x;
    res(idx) = lha(idx) + rha(idx);
}

void init_array(iki::device::Array<float,1> arr, std::vector<float> &src) {
    cudaError_t cudaStatus;

    size_t shape_collapse[2] = {1, 1};
     if (cudaSuccess != (cudaStatus = cudaMemcpy(arr.shape, shape_collapse, 2 * sizeof(size_t), cudaMemcpyHostToDevice)))
        throw iki::device::DeviceError("Can't copy shape memory: ", cudaStatus);

    
    if (cudaSuccess != (cudaStatus = cudaMemcpy(arr.data, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice)))
        throw iki::device::DeviceError("Can't copy data memory: ", cudaStatus);
}

int main() {
    try {
    size_t size = 1024;
    size_t byte_size = size * sizeof(float) + sizeof(size_t) * 2;
    cout << "Init data..." << endl;
    iki::device::Device device(0);
    iki::device::DeviceMemory 
        lha_mem(byte_size), rha_mem(byte_size), res_mem(byte_size);

    iki::device::Array<float,1> lha(lha_mem.as<size_t>(),lha_mem.as<size_t>() + 1,(float*)(lha_mem.as<size_t>() + 2));
    iki::device::Array<float,1> rha(rha_mem.as<size_t>(),rha_mem.as<size_t>() + 1,(float*)(rha_mem.as<size_t>() + 2));
    iki::device::Array<float,1> res(res_mem.as<size_t>(),res_mem.as<size_t>() + 1,(float*)(res_mem.as<size_t>() + 2));

    vector<float> vec(1024,1.);
    init_array(lha,vec);
    init_array(rha,vec);
    init_array(res,vec); 

    cout << "Kernel call..." << endl;
    kernel_array_sum<<<1,1024>>>(lha, rha, res);
    cudaDeviceSynchronize();

    cout << "Result: " << endl;
    {
        cudaError_t cudaStatus;
        if (cudaSuccess != (cudaStatus = cudaMemcpy(&vec[0], res.data, vec.size() * sizeof(float), cudaMemcpyDeviceToHost)))
            throw iki::device::DeviceError("Can't copy data memory: ", cudaStatus);

        for(auto x : vec)
            cout << x << " ";
        cout << endl;
    }
    }
    catch(std::exception const &ex) {
        cout << ex.what() << endl;
    }


    return 0;
}