#include "tools.h"

std::uint64_t get_page_faults() {
	std::ifstream stat_file; 
	stat_file.open("/proc/self/stat");
 
	std::vector<std::string> stat_vector;
    std::string s;
    while (std::getline(stat_file, s, ' ')) {
        stat_vector.push_back(s);
    }

    stat_file.close();

	std::string pf_string = stat_vector[11];

	return std::stoi(pf_string);
}

std::uint64_t get_io_delay_ticks() {
	std::ifstream stat_file; 
	stat_file.open("/proc/self/stat");
 
	std::vector<std::string> stat_vector;
    std::string s;
    while (std::getline(stat_file, s, ' ')) {
        stat_vector.push_back(s);
    }

    stat_file.close();

	std::string pf_string = stat_vector[41];

	return std::stoi(pf_string);
}

// Am I sure carries are being handled correctly?
// They are definitely not being handled correctly.
// TODO: PLEASE FIX THIS
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t n, uint128_t m) {
    uint128_t prod_ = (uint128_t)a * b;

    uint64_t prod[2] {(uint64_t)prod_, (uint64_t)(prod_ >> 64)};
    uint64_t ratio[2] {(uint64_t)m, (uint64_t)(m >> 64)};
    
    uint128_t zero_zero = (uint128_t)(prod[0]) * (uint128_t)(ratio[0]);
    uint128_t zero_one = (uint128_t)(prod[0]) * ratio[1];
    uint128_t one_zero = (uint128_t)(prod[1]) * ratio[0];
    uint128_t one_one = (uint128_t)(prod[1]) * ratio[1];

    uint128_t mid_prod = (zero_zero >> 64) + zero_one + one_zero;
    uint128_t top_prod = (mid_prod >> 64) + one_one;

    uint64_t q = top_prod;

    uint64_t r = prod[0] - q * n;
    if (r >= n) {
        r -= n;
    }
    return r;
}

uint64_t addmod(uint64_t a, uint64_t b, uint64_t n) {
    uint128_t sum = (uint128_t)a + b;
    if (sum >= n) {
        sum -= n;
    }
    return sum;
}


void mul_matrix_matrix_mod(uint64_t* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_sub_rows, uint64_t A_sub_cols,
						    uint64_t* B, uint64_t B_rows, uint64_t B_cols, uint64_t B_sub_rows, uint64_t B_sub_cols,
						    uint64_t* C, uint64_t N, std::vector<uint64_t> coeff_moduli, std::vector<uint128_t> m)
{
    // Multiply a matrix of matrices by another matrix of matrices

    assert(A_cols == B_rows);
    assert(A_sub_cols == B_sub_rows);

    uint64_t poly_size = N * coeff_moduli.size();

    uint64_t C_rows = A_rows;
    uint64_t C_cols = B_cols;
    uint64_t C_sub_rows = A_sub_rows;
    uint64_t C_sub_cols = B_sub_cols;

    uint64_t A_sub_size = A_sub_rows * A_sub_cols * poly_size;
    uint64_t B_sub_size = B_sub_rows * B_sub_cols * poly_size;
    uint64_t C_sub_size = C_sub_rows * C_sub_cols * poly_size;

    uint64_t A_row_size = A_cols * A_sub_size;
    uint64_t A_cell_size = A_sub_size;
    uint64_t A_sub_row_size = A_sub_cols * poly_size;
    uint64_t A_sub_cell_size = poly_size;

    uint64_t B_row_size = B_cols * B_sub_size;
    uint64_t B_cell_size = B_sub_size;
    uint64_t B_sub_row_size = B_sub_cols * poly_size;
    uint64_t B_sub_cell_size = poly_size;

    uint64_t C_row_size = C_cols * C_sub_size;
    uint64_t C_cell_size = C_sub_size;
    uint64_t C_sub_row_size = C_sub_cols * poly_size;
    uint64_t C_sub_cell_size = poly_size;

    // Einsum
    // C_row in [0, A_rows) -> [0, M)
    // C_col in [0, B_cols) -> [0, N)
    // C_sub_row in [0, A_sub_rows) -> [0, P)
    // C_sub_col in [0, B_sub_cols) -> [0, Q)
    // cm in [0, coeff_moduli.size()) -> [0, C)
    // p in [0, N) -> [0, D)
    // i in [0, A_cols) -> [0, J)
    // j in [0, A_sub_cols) -> [0, K)


    // C_(m, n, p, q, c, d) = A_(m, j, p, k, c, d) * B_(j, n, k, q, c, d)
    // C_(C_row, C_col, C_sub_row, C_sub_col, cm, p) = A_(C_row, i, C_sub_row, j, cm, p) * B_(i, C_col, j, C_sub_col, cm, p)

    printf("Matrix MUL parameters:\n");
    printf("M: %ld\nN: %ld\nP: %ld\nQ: %ld\nJ: %ld\nK: %ld\nC: %ld\nD: %ld\n", A_rows, B_cols, A_sub_rows, B_sub_cols, A_cols, A_sub_cols, coeff_moduli.size(), N);

    // Same as B_total_cols
    uint64_t A_total_cols = A_cols * A_sub_cols;

    for (uint64_t C_row = 0; C_row < C_rows; C_row++) {
        uint64_t A_row = C_row;
        for (uint64_t C_sub_row = 0; C_sub_row < C_sub_rows; C_sub_row++) {
            uint64_t A_sub_row = C_sub_row;
            for (uint64_t C_col = 0; C_col < C_cols; C_col++) {
                uint64_t B_col = C_col;
                for (uint64_t C_sub_col = 0; C_sub_col < C_sub_cols; C_sub_col++) {
                    uint64_t B_sub_col = C_sub_col;

                    // One specific polynmomial in output
        
                    for (uint64_t cm = 0; cm < coeff_moduli.size(); cm++) {
                        for (uint64_t p = 0; p < N; p++) {
                            uint64_t cell = 0;

                            // One specific coefficient

                            // Multiply and add along a row/column of A/B
                            for (uint64_t i = 0; i < A_cols; i++) {
                                uint64_t A_col = i;
                                uint64_t B_row = i;
                                for (uint64_t j = 0; j < A_sub_cols; j++) {
                                    uint64_t A_sub_col = j;
                                    uint64_t B_sub_row = j;

                                    uint64_t A_index = (A_row * A_row_size) + (A_col * A_cell_size)
                                                        + (A_sub_row * A_sub_row_size) + (A_sub_col * A_sub_cell_size)
                                                        + cm * N + p;
                                    uint64_t B_index = (B_row * B_row_size) + (B_col * B_cell_size)
                                                        + (B_sub_row * B_sub_row_size) + (B_sub_col * B_sub_cell_size)
                                                        + cm * N + p;

                                    uint64_t A_value = A[A_index];
                                    uint64_t B_value = B[B_index];
                                    uint64_t AB = mulmod(A_value, B_value, coeff_moduli[cm], m[cm]);
                                    // printf("m: %lx\n", m[cm]);
                                    cell = addmod(cell, AB, coeff_moduli[cm]);
                                }
                            }

                            uint64_t C_index = (C_row * C_row_size) + (C_col * C_cell_size)
                                                + (C_sub_row * C_sub_row_size) + (C_sub_col * C_sub_cell_size)
                                                + cm * N + p;
                            
                            C[C_index] = cell;
                        }
                    }
                }
            }
        }
    }
}

MmapDeleter::MmapDeleter(size_t size) {
    this->size = size;
}

void MmapDeleter::operator()(const void* p) { 
    munmap(const_cast<void*>(p), size);
}

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

