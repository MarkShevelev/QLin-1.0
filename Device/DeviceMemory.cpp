#include <Device/DeviceMemory.h>
#include <Device/DeviceError.h>

#include <cuda_runtime.h>

#include <utility>

using namespace std;

namespace iki { namespace device {
	DeviceMemory::DeviceMemory(unsigned byte_size) : device_ptr(nullptr), byte_size(byte_size) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaMalloc(&device_ptr, byte_size)))
			throw DeviceError("Can't allocate memory: ", cudaStatus);
	}

	DeviceMemory::DeviceMemory(DeviceMemory const &src): DeviceMemory(src.byte_size) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaMemcpy(device_ptr, src.device_ptr, byte_size, cudaMemcpyDeviceToDevice)))
			throw DeviceError("Can't copy device memory: ", cudaStatus);
	}

	DeviceMemory& DeviceMemory::operator=(DeviceMemory const &src) {
		DeviceMemory tmp(src);
		std::swap(tmp.device_ptr, device_ptr);
		std::swap(tmp.byte_size, byte_size);
		return *this;
	}

	DeviceMemory::DeviceMemory(DeviceMemory &&src) : device_ptr(src.device_ptr), byte_size(src.byte_size) {
		src.device_ptr = nullptr;
		src.byte_size = 0;
	}

	DeviceMemory& DeviceMemory::operator=(DeviceMemory &&src) {
		DeviceMemory tmp(move(src));
		std::swap(tmp.device_ptr, device_ptr);
		std::swap(tmp.byte_size, byte_size);
		return *this;
	}

	void DeviceMemory::swap(DeviceMemory &src) {
		std::swap(device_ptr,src.device_ptr);
		std::swap(byte_size,src.byte_size);
	}

} /*device*/ }/*iki*/