#include <Device/DeviceMemory.h>
#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>

#include <array>

namespace iki { namespace device {
    template <typename T, size_t Dim>
    class HostManagedArray final {
    private:
        device::DeviceMemory device_memory;
        std::array<size_t,Dim> host_shape;
        std::array<size_t, Dim> host_collapse;
        
        template<typename... Shape>
        inline
        static size_t collapse_shape(size_t size, Shape... shape) {
            return size * collapse_shape(shape);
        }

        template<>
        inline
        static size_t collapse_shape(size_t size) {
            return size;
        }

    public:
        template <typename... Shape>
        HostManagedArray(Shape... shape): device_memory(sizeof(T) * collapse_shape(shape) + 2 * Dim * sizeof(size_t)) {
            static_assert(sizeof...(shape) == Dim, "Number of arguments is not equal to the array dimension");
            size_t tmp[Dim] = {shape...};
            host_shape[0] = tmp[0];
            host_collapse[Dim - 1u] = 1u;
            for (size_t idx = 1u; idx != Dim; ++idx) {
                host_shape[idx] = tmp[idx];
                host_collapse[Dim - idx - 1u] = tmp[idx] * host_collapse[Dim - idx];
            }

            {//copy shape
                cudaError_t cudaStatus;
                if (cudaSuccess != (cudaStatus = cudaMemcpy(device_memory.as<size_t>(), host_shape, Dim * sizeof(size_t), cudaMemcpyHostToDevice)))
                    throw DeviceError("Can't copy from host to device memory: ", cudaStatus);
            }

            {//copy collapse
                cudaError_t cudaStatus;
                if (cudaSuccess != (cudaStatus = cudaMemcpy(device_memory.as<size_t>() + Dim, host_collapse, Dim * sizeof(size_t), cudaMemcpyHostToDevice)))
                    throw DeviceError("Can't copy from host to device memory: ", cudaStatus);
            }
        }

        Array<T, Dim> array() const {
            return Array<T, Dim>(device_memory.as<size_t>(), device_memory.as<size_t>() + Dim, reinterpret_cast<T*>(device_memory.as<size_t>() + 2 * Dim));
        }

        std::array<size_t, Dim> get_shape() const {
            return host_shape;
        }

        std::array<size_t, Dim> get_collapse() const {
            return host_collapse;
        }
    };
} /*device*/ } /*iki*/