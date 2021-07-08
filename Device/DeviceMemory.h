#pragma once

namespace iki { namespace device {
	class DeviceMemory final {
	public:
		DeviceMemory(unsigned byte_size);

		DeviceMemory(DeviceMemory const &src);
		DeviceMemory(DeviceMemory &&src);
		DeviceMemory& operator=(DeviceMemory const &src);
		DeviceMemory& operator=(DeviceMemory &&src);

		void* get() const {
			return device_ptr;
		}

		template <typename T>
		T* as() const {
			return reinterpret_cast<T*>(device_ptr);
		}

		size_t get_size() const {
			return byte_size;
		}

		void swap(DeviceMemory &src);

	private:
		void *device_ptr;
		unsigned byte_size;
	};
} /*device*/ } /*iki*/