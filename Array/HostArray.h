#pragma once

#include <vector>
#include <array>

namespace iki {
	template <typename T, size_t Dim>
	class HostArray final {
	private:
        size_t const full_size;
		std::array<size_t, Dim> shape;
		std::array<size_t, Dim> collapse;
		std::vector<T> host_memory;

        template<typename T, typename... Shape>
        inline
            static T full_size_calc(T size, Shape... shape) {
            return size * full_size_calc(shape...);
        }

        template<typename T>
        inline
            static T full_size_calc(T size) {
            return size;
        }

        template<size_t Dim>
        inline
            size_t get_shift_rec(size_t *ixs, size_t *collapse, size_t shift) {
            return get_shift_rec<Dim - 1>(ixs, collapse, shift + ixs[Dim - 1] * collapse[Dim - 1]);
        }

        template<>
        inline
            size_t get_shift_rec<0>(size_t *ixs, size_t *collapse, size_t shift) {
            return shift;
        }

        template<size_t Dim>
        inline
            size_t get_shift(size_t *ixs, size_t *collapse) {
            return get_shift_rec<Dim>(ixs, collapse, 0u);
        }

	public:
        template <typename... Shape>
        HostArray(Shape... shape) : full_size(full_size_calc<size_t>(shape...)), shape({ shape... }), host_memory(full_size) {
            static_assert(sizeof...(shape) == Dim, "Number of arguments is not equal to the array dimension");
            collapse[Dim - 1u] = 1u;
            for (size_t idx = 1u; idx != Dim; ++idx)
                collapse[Dim - idx - 1u] = shape[Dim - idx] * collapse[Dim - idx];
        }

        std::array<size_t, Dim> get_shape() const {
            return host_shape;
        }

        std::array<size_t, Dim> get_collapse() const {
            return host_collapse;
        }

        size_t get_full_size() const {
            return full_size;
        }

        T const* data() const {
            return host_memory.data();
        }

        T* data() {
            return host_memory.data();
        }

        template<typename... Ixs_t>
        T operator()(Ixs_t ... Ixs) const {
            static_assert(sizeof...(Ixs) == Dim, "Number of indexes is not equal to the array dimension");
            size_t ixs[Dim] = { Ixs... };
            return host_memory[get_shift<Dim>(ixs, collapse)];
        }

        template<typename... Ixs_t>
        T &operator()(Ixs_t ... Ixs) {
            static_assert(sizeof...(Ixs) == Dim, "Number of indexes is not equal to the array dimension");
            size_t ixs[Dim] = { Ixs... };
            return host_memory[get_shift<Dim>(ixs, collapse)];
        }
	};

    template <typename T>
    class HostArray<T,1u> final {
    private:
        size_t const full_size;
        std::array<size_t, 1u> shape;
        std::array<size_t, 1u> collapse;
        std::vector<T> host_memory;

    public:
        HostArray(size_t size) : full_size(size), shape({size}), host_memory(size) {
            collapse[0] = 1u;
        }

        std::array<size_t, 1u> get_shape() const {
            return shape;
        }

        std::array<size_t, 1u> get_collapse() const {
            return collapse;
        }

        size_t get_full_size() const {
            return full_size;
        }

        T const *data() const {
            return host_memory.data();
        }

        T *data() {
            return host_memory.data();
        }

        T operator()(size_t Ixs) const {
            return host_memory[Ixs];
        }

        T& operator()(size_t Ixs) {
            return host_memory[Ixs];
        }
    };

    template <typename T>
    class HostArray<T, 2u> final {
    private:
        size_t const full_size;
        std::array<size_t, 2u> shape;
        std::array<size_t, 2u> collapse;
        std::vector<T> host_memory;

    public:
        HostArray(size_t y_size, size_t x_size) : full_size(y_size * x_size), shape({ y_size, x_size }), host_memory(x_size * y_size) {
            collapse[0] = 1u;
            collapse[1] = x_size;
        }

        std::array<size_t, 2u> get_shape() const {
            return shape;
        }

        std::array<size_t, 2u> get_collapse() const {
            return collapse;
        }

        size_t get_full_size() const {
            return full_size;
        }

        T const *data() const {
            return host_memory.data();
        }

        T *data() {
            return host_memory.data();
        }

        T operator()(size_t y_idx, size_t x_idx) const {
            return host_memory[y_idx * x_size + x_idx];
        }

        T &operator()(size_t y_idx, size_t x_idx) {
            return host_memory[y_idx * x_size + x_idx];
        }
    };
}/*iki*/