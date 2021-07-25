#include <cuda_runtime.h>

#include <Array/DeviceArray.cuh>

namespace iki { namespace solver { namespace device {
	template <typename T>
	__global__ void thomson_sweep_kernel(
		iki::device::Array<T, 2u> a,
		iki::device::Array<T, 2u> b,
		iki::device::Array<T, 2u> c,
		iki::device::Array<T, 2u> d,
		iki::device::Array<T, 2u> res
	) {
		size_t x_idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (x_idx == 0 || x_idx >= a.shape[1] - 1) return;

		for (size_t y_idx = 2; y_idx != res.shape[0] - 1; ++y_idx) {
			T w = a(y_idx, x_idx) / b(y_idx - 1, x_idx);
			b(y_idx, x_idx) = fma(-w, c(y_idx - 1, x_idx), b(y_idx, x_idx));
			d(y_idx, x_idx) = fma(-w, d(y_idx - 1, x_idx), d(y_idx, x_idx));
		}
		res(res.shape[0] - 2, x_idx) = d(res.shape[0] - 2, x_idx) / b(res.shape[0] - 2, x_idx);

		for (size_t y_idx = res.shape[0] - 3; y_idx != 0; --y_idx) {
			res(y_idx, x_idx) = fma(-c(y_idx, x_idx), res(y_idx + 1, x_idx), d(y_idx, x_idx)) / b(y_idx, x_idx);
		}
	}
}/*device*/ }/*solver*/ }/*iki*/