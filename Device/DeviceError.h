#pragma once

#include <cuda_runtime.h>

#include <string>
#include <stdexcept>

namespace iki { namespace device {
	class DeviceError final: public std::runtime_error {
	public:
		DeviceError(std::string const &additional_text, cudaError_t cudaStatus);
		DeviceError(cudaError_t cudaStatus);
	};
} /*device*/ } /*iki*/