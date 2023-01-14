#ifndef NFL_TESTS_TOOLS_H
#define NFL_TESTS_TOOLS_H

#include <chrono>
#include <sys/mman.h>
#include <vector>
#include <memory_resource>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <atomic>
#include <thread>

using namespace std::chrono;

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

std::uint64_t get_page_faults();

std::uint64_t get_io_delay_ticks();

struct MmapDeleter {
    private:
        size_t size;

    public:
        MmapDeleter(size_t size);

        void operator()(const void* p);
};

class DiskTracker {
	private:
		std::uint64_t interval_ms;
		std::vector<std::uint64_t> page_faults;
		std::atomic_bool running;
		std::thread run_thread;
		std::uint64_t runtime;
		std::uint64_t io_delay_ticks;

		void loop() {
			auto start_time = high_resolution_clock::now();
			auto last_time = start_time;
			std::uint64_t last_page_faults = get_page_faults();
			std::uint64_t last_io_delay_ticks = get_io_delay_ticks();
			printf("Last io delay ticks: %ld\n", last_io_delay_ticks);

			while (running) {
				auto current_time = high_resolution_clock::now();
				std::uint64_t time_diff = duration_cast<milliseconds>(current_time - last_time).count();
				if (time_diff > interval_ms) {
					std::uint64_t current_page_faults = get_page_faults();
					std::uint64_t new_page_faults = current_page_faults - last_page_faults;
					page_faults.push_back(new_page_faults);
					last_time = current_time;
					last_page_faults = current_page_faults;
				}
			}
			if (page_faults.size() == 0) {
				std::uint64_t current_page_faults = get_page_faults();
				std::uint64_t new_page_faults = current_page_faults - last_page_faults;
				page_faults.push_back(new_page_faults);
			}

			std::uint64_t current_io_delay_ticks = get_io_delay_ticks();
			io_delay_ticks = current_io_delay_ticks - last_io_delay_ticks;

			auto end_time = high_resolution_clock::now();
			runtime = duration_cast<milliseconds>(end_time - start_time).count();
			
		}


	public:
		DiskTracker(std::uint64_t interval_ms) {
			this->interval_ms = interval_ms;
			running = false;
		}

		void start() {
			running = true;
			run_thread = std::thread([this] { loop(); });
		}

		void stop() {
			running = false;
			run_thread.join();
		}

		void print_stats() {
			printf("Going to print some stats\n");
			std::uint64_t max_page_faults = 0;
			std::uint64_t sum_page_faults = 0;

			for (int i = 0; i < page_faults.size(); i++) {
				if (page_faults[i] > max_page_faults) {
					max_page_faults = page_faults[i];
				}
				sum_page_faults += page_faults[i];
			}
			
			std::uint64_t page_size = sysconf(_SC_PAGE_SIZE);

			std::uint64_t average_bytes_per_second = (sum_page_faults * page_size * 1000)/runtime;
			std::uint64_t max_bytes_per_second = (max_page_faults * page_size * 1000)/std::min(interval_ms, runtime);
			
			printf("Disk statistics:\n");

			printf("Runtime: %ld ms\n", runtime);

			std::uint64_t io_delay_ms = (io_delay_ticks * 1000) / sysconf(_SC_CLK_TCK);
			printf("IO Delay: %ld ms\n", io_delay_ms);
			printf("IO Delay ticks: %ld\n", io_delay_ticks);

			printf("Total page faults disk tracker: %ld\n", sum_page_faults);

			std::uint64_t page_fault_bytes = sum_page_faults * page_size;
			if (page_fault_bytes< (1 << 10)) {
				printf("Page faults: %ld B\n", page_fault_bytes);
			} else if (page_fault_bytes < (1 << 20)) {
				printf("Page faults: %ld KB\n", page_fault_bytes >> 10);
			} else if (page_fault_bytes < (1 << 30)) {
				printf("Page faults: %ld MB\n", page_fault_bytes >> 20);
			} else {
				printf("Page faults: %ld GB\n", page_fault_bytes >> 30);
			}

			if (average_bytes_per_second < (1 << 10)) {
				printf("Average: %ld B/s\n", average_bytes_per_second);
			} else if (average_bytes_per_second < (1 << 20)) {
				printf("Average: %ld KB/s\n", average_bytes_per_second >> 10);
			} else if (average_bytes_per_second < (1 << 30)) {
				printf("Average: %ld MB/s\n", average_bytes_per_second >> 20);
			} else {
				printf("Average: %ld GB/s\n", average_bytes_per_second >> 30);
			}

			if (max_bytes_per_second < (1 << 10)) {
				printf("Max: %ld B/s\n", max_bytes_per_second);
			} else if (max_bytes_per_second < (1 << 20)) {
				printf("Max: %ld KB/s\n", max_bytes_per_second >> 10);
			} else if (max_bytes_per_second < (1 << 30)) {
				printf("Max: %ld MB/s\n", max_bytes_per_second >> 20);
			} else {
				printf("Max: %ld GB/s\n", max_bytes_per_second >> 30);
			}
		}

		void clear() {
			page_faults.clear();
		}
};

// class ReservedMmapAllocator: public std::pmr::memory_resource {
// 	private:
// 		int fd;
// 		std::size_t file_bytes;
// 		std::size_t mapped_bytes;
// 		uint8_t* mmap_region;

// 		void* do_allocate(std::size_t bytes, std::size_t alignment) {
// 			assert(alignment <= sysconf(_SC_PAGE_SIZE));

// 			std::size_t next_byte = mapped_bytes;
// 			std::size_t mask = ~(alignment - 1);
// 			std::size_t next_byte_masked = next_byte & mask;
// 			if (next_byte != next_byte_masked) {
// 				next_byte = next_byte_masked + alignment;
// 			}
			
// 			std::size_t bytes_left = file_bytes - next_byte;

// 			if (bytes_left >= bytes) {
// 				// Enough bytes left in file
// 				mapped_bytes = next_byte + bytes;
// 				return (void*) (mmap_region + next_byte);
// 			} else {
// 				throw std::runtime_error("Ran out of reserved mmap space");
// 			}

// 		}

// 		void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
// 			if (p == mmap_region && bytes == file_bytes) {
// 				munmap(p, bytes);
				
// 				mmap_region = nullptr;
// 				mapped_bytes = 0;
// 				file_bytes = 0;
// 				ftruncate(fd, 0);
// 			} else {
// 				throw std::runtime_error("Attempted to deallocate partial mmap");
// 			}
// 		}

// 		bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
// 			if (const ReservedMmapAllocator* m = dynamic_cast<const ReservedMmapAllocator*>(&other); m != nullptr) {
// 				return true;
// 			}
// 			return false;
// 		}

// 	public:
// 		//static const uint64_t PAGE_SIZE;

// 		ReservedMmapAllocator() {
// 			char file_template[] = "mmap_allocator_XXXXXX";
// 			fd = mkstemp(file_template);
// 			file_bytes = 0;
// 			mapped_bytes = 0;
// 			mmap_region = nullptr;
// 		}

// 		void reserve(std::size_t reserve_bytes) {
// 			assert(mmap_region == nullptr);
// 			assert(file_bytes == 0);
// 			assert(mapped_bytes == 0);

// 			ftruncate(fd, reserve_bytes);
// 			uint8_t* p = (uint8_t*) mmap(NULL, reserve_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); 

// 			// Reserve memory mapped region in advance to use for subsequent allocations
// 			if (p == MAP_FAILED) {
// 				// mmap failed, do not keep reservations
// 				throw std::runtime_error("Mmap failed for reservation in mmap allocator");
// 			} else {
// 				mmap_region = p;
// 				mapped_bytes = 0;
// 				file_bytes = reserve_bytes;
// 			}
// 		}

// 		~ReservedMmapAllocator() {
// 			close(fd);
// 		}
// };

#endif
