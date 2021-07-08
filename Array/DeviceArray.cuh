#pragma once

#include <cuda_runtime.h>

namespace iki { namespace device {
    template<size_t Dim>
    inline
    __device__ size_t get_shift_rec(size_t *ixs, size_t *collapse, size_t shift) {
        return get_shift_rec<Dim - 1>(ixs, collapse, shift + ixs[Dim - 1] * collapse[Dim - 1]);
    }

    template<>
    inline
    __device__ size_t get_shift_rec<0>(size_t *ixs, size_t *collapse, size_t shift) {
        return shift;
    }

    template<size_t Dim>
    inline
    __device__ size_t get_shift(size_t *ixs, size_t *collapse) {
        return get_shift_rec<Dim>(ixs, collapse, 0u);
    }

    template <typename T, size_t Dim>
    struct Array final {
        size_t *const shape, *const collapse; //device pointer
        T * const data; //device pointer

        __host__ __device__ Array(size_t *shape, size_t *collapse, T *data): shape(shape), collapse(collapse), data(data) {
        }

        template<typename... Ixs_t>
        __device__ T operator()(Ixs_t ... Ixs) const {
            static_assert(sizeof...(Ixs)==Dim, "Number of indexes is not equal to the array dimension");
            size_t ixs[Dim] = {Ixs...};
            return data[get_shift<Dim>(ixs,collapse)]; 
        }

        template<typename... Ixs_t>
        __device__ T& operator()(Ixs_t ... Ixs) {
            static_assert(sizeof...(Ixs)==Dim, "Number of indexes is not equal to the array dimension");
            size_t ixs[Dim] = {Ixs...};
            return data[get_shift<Dim>(ixs,collapse)]; 
        }
    };

    template <typename T>
    struct Array<T,1u> final {
        size_t *const shape, *const collapse; //device pointer
        T *const data; //device pointer

        __host__ __device__ Array(size_t *shape, size_t *collapse, T *data) : shape(shape), collapse(collapse), data(data) { }

        __device__ T operator()(size_t idx) const {
            return data[idx];
        }

        __device__ T& operator()(size_t idx) {
            return data[idx];
        }
    };

    template <typename T>
    struct Array<T, 2u> final {
        size_t *const shape, *const collapse; //device pointer
        T *const data; //device pointer

        __host__ __device__ Array(size_t *shape, size_t *collapse, T *data) : shape(shape), collapse(collapse), data(data) { }

        __device__ T operator()(size_t y_idx, size_t x_idx) const {
            return data[y_idx * collapse[0] + x_idx];
        }

        __device__ T& operator()(size_t y_idx, size_t x_idx) {
            return data[y_idx * collapse[0] + x_idx];
        }
    };
} /*device*/ } /*iki*/