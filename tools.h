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

class MmapAllocator: public std::pmr::memory_resource {
	private:
		uint8_t* reserved;
		std::size_t reserved_bytes;

		void* do_allocate(std::size_t bytes, std::size_t alignment) {
			if (reserved_bytes >= bytes) {
				// Enough bytes left in reservation

				// Get next pointer after reservation that is aligned to alignment
				uint8_t* p = reserved;
				std::size_t mask = ~(alignment - 1);
				uint8_t* p_mask = (uint8_t*)((uint64_t)p & mask);
				if (p_mask != p) {
					p = p_mask + alignment;
				}
				std::size_t extra_bytes = p - reserved;

				// Check if we still have enough bytes left after the alignment
				if (reserved_bytes >= bytes + extra_bytes) {
					// Update reservation by decreasing available bytes
					reserved = p + bytes;
					reserved_bytes -= (bytes + extra_bytes);
					return (void*) p;
				}
			}

			void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
			if (p == MAP_FAILED) {
				throw std::runtime_error("Mmap failed for mmap allocator");
			}
			return p;
		}

		void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
			munmap(p, bytes);
		}

		bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
			if (const MmapAllocator* m = dynamic_cast<const MmapAllocator*>(&other); m != nullptr) {
				return true;
			}
			return false;
		}

	public:
		MmapAllocator() {
			reserved = nullptr;
			reserved_bytes = 0;
		}

		void reserve(std::size_t reserve_bytes) {
			// Reserve memory mapped region in advance to use for subsequent allocations
			reserved = (uint8_t*) mmap(NULL, reserve_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
			if (reserved == MAP_FAILED) {
				// mmap failed, do not keep reservations
				reserved = nullptr;
				reserved_bytes = 0;
			} else {
				reserved_bytes = reserve_bytes;
			}
		}
};

#endif
