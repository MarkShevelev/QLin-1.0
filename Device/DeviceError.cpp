#include <Device/DeviceError.h>

using namespace std;

namespace iki { namespace device {
	DeviceError::DeviceError(string const &additional_text, cudaError_t cudaStatus): 
		runtime_error(additional_text + cudaGetErrorString(cudaStatus)) {  }
	DeviceError::DeviceError(cudaError_t cudaStatus): 
		runtime_error(cudaGetErrorString(cudaStatus)) {  }
} /*device*/ } /*iki*/