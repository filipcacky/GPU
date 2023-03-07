#include <csr/matrix.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace indirection {
using value_t = int;
using vector_t = thrust::device_vector<value_t>;
using matrix_t = csr::matrix<value_t, thrust::device_vector>;

__host__ vector_t dot(const matrix_t &mx, const vector_t &v);

__host__ value_t norm(const vector_t &vector, int l);

__host__ value_t dot(const vector_t &first, const vector_t &second);

__host__ void scale(vector_t &vector, value_t scalar);

__host__ void add(vector_t &first, const vector_t &second);

__host__ void add_scaled(vector_t &first, value_t s, const vector_t &second);

__host__ void sub(vector_t &first, const vector_t &second);

__host__ void sub_scaled(vector_t &first, value_t s, const vector_t &second);

} // namespace indirection
