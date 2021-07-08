#include <Device/Device.h>
#include <Device/DeviceError.h>

#include <cuda_runtime.h>

using namespace std;

namespace iki { namespace device {
	Device::Device(int device) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(device)))
			throw DeviceError(cudaStatus);
	}

	Device::~Device() noexcept {
		cudaDeviceReset();
	}
} /*device*/ } /*iki*/