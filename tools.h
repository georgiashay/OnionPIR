#ifndef NFL_TESTS_TOOLS_H
#define NFL_TESTS_TOOLS_H

#include <chrono>
#include <sys/mman.h>
#include <vector>
#include <memory_resource>
#include <unistd.h>

template <class T, size_t Align, class... Args>
T* alloc_aligned(size_t n, Args&& ... args)
{

	T* ret;
	if (posix_memalign((void**) &ret, Align, sizeof(T)*n) != 0) {
		throw std::bad_alloc();
	}
	for (size_t i = 0; i < n; i++) {
		new (&ret[i]) T(std::forward<Args>(args)...);
	}
	return ret;
}

template <class T>
void free_aligned(size_t n, T* p)
{
	for (size_t i = 0; i < n; i++) {
		p[i].~T();
	}
	free(p);
}

template <class T>
double get_time_us(T const& start, T const& end, uint32_t N)
{
  auto diff = end-start;
  return (long double)(std::chrono::duration_cast<std::chrono::microseconds>(diff).count())/N;
}

struct MmapDeleter {
    private:
        size_t size;

    public:
        MmapDeleter(size_t size) {
            this->size = size;
        }

        void operator()(const void* p) { 
             munmap(const_cast<void*>(p), size);
        }
};

class ReservedMmapAllocator: public std::pmr::memory_resource {
	private:
		int fd;
		std::size_t file_bytes;
		std::size_t mapped_bytes;
		uint8_t* mmap_region;

		void* do_allocate(std::size_t bytes, std::size_t alignment) {
			assert(alignment <= sysconf(_SC_PAGE_SIZE));

			std::size_t next_byte = mapped_bytes;
			std::size_t mask = ~(alignment - 1);
			std::size_t next_byte_masked = next_byte & mask;
			if (next_byte != next_byte_masked) {
				next_byte = next_byte_masked + alignment;
			}
			
			std::size_t bytes_left = file_bytes - next_byte;

			if (bytes_left >= bytes) {
				// Enough bytes left in file
				mapped_bytes = next_byte + bytes;
				return (void*) (mmap_region + next_byte);
			} else {
				throw std::runtime_error("Ran out of reserved mmap space");
			}

		}

		void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
			if (p == mmap_region && bytes == file_bytes) {
				munmap(p, bytes);
				
				mmap_region = nullptr;
				mapped_bytes = 0;
				file_bytes = 0;
				ftruncate(fd, 0);
			} else {
				throw std::runtime_error("Attempted to deallocate partial mmap");
			}
		}

		bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
			if (const ReservedMmapAllocator* m = dynamic_cast<const ReservedMmapAllocator*>(&other); m != nullptr) {
				return true;
			}
			return false;
		}

	public:
		//static const uint64_t PAGE_SIZE;

		ReservedMmapAllocator() {
			char file_template[] = "mmap_allocator_XXXXXX";
			fd = mkstemp(file_template);
			file_bytes = 0;
			mapped_bytes = 0;
			mmap_region = nullptr;
		}

		void reserve(std::size_t reserve_bytes) {
			assert(mmap_region == nullptr);
			assert(file_bytes == 0);
			assert(mapped_bytes == 0);

			ftruncate(fd, reserve_bytes);
			uint8_t* p = (uint8_t*) mmap(NULL, reserve_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); 

			// Reserve memory mapped region in advance to use for subsequent allocations
			if (p == MAP_FAILED) {
				// mmap failed, do not keep reservations
				throw std::runtime_error("Mmap failed for reservation in mmap allocator");
			} else {
				mmap_region = p;
				mapped_bytes = 0;
				file_bytes = reserve_bytes;
			}
		}

		~ReservedMmapAllocator() {
			close(fd);
		}
};

#endif
