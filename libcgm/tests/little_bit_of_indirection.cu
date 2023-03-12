#include "little_bit_of_indirection.cuh"

#include <cgm/traits.cuh>

namespace indirection {

using matrix_algorithms =
    csr::matrix_algorithms<value_t, thrust::device_vector>;

using vector_algorithms =
    csr::vector_algorithms<value_t, thrust::device_vector>;

vector_t dot(const matrix_t &mx, const vector_t &v) {
  vector_t result;
  matrix_algorithms::dot(mx, v, result);
  return result;
}

value_t norm(const vector_t &vector, int l) {
  auto device_ptr = vector_algorithms::make_device_ptr();
  auto host_ptr = vector_algorithms::make_host_ptr();
  auto result = vector_algorithms::norm(vector, l, device_ptr, host_ptr);
  vector_algorithms::delete_device_ptr(std::move(device_ptr));
  vector_algorithms::delete_host_ptr(std::move(host_ptr));
  return result;
}

value_t dot(const vector_t &first, const vector_t &second) {
  auto device_ptr = vector_algorithms::make_device_ptr();
  auto host_ptr = vector_algorithms::make_host_ptr();
  auto result = vector_algorithms::dot(first, second, device_ptr, host_ptr);
  vector_algorithms::delete_device_ptr(std::move(device_ptr));
  vector_algorithms::delete_host_ptr(std::move(host_ptr));
  return result;
}

void scale(vector_t &vector, value_t scalar) {
  vector_algorithms::scale(vector, scalar);
}

void add(vector_t &first, const vector_t &second) {
  vector_algorithms::add(first, second);
}

void add_scaled(vector_t &first, value_t s, const vector_t &second) {
  vector_algorithms::add_scaled(first, s, second);
}

void sub(vector_t &first, const vector_t &second) {
  vector_algorithms::sub(first, second);
}

void sub_scaled(vector_t &first, value_t s, const vector_t &second) {
  vector_algorithms::sub_scaled(first, s, second);
}

} // namespace indirection
