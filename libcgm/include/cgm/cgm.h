#pragma once

#include <cmath>

#include <csr/matrix.hpp>
#include <csr/traits.hpp>

namespace cgm {

template <typename T, template <typename...> typename Storage>
std::pair<Storage<T>, size_t> solve(const csr::matrix<T, Storage> &A,
                                    const Storage<T> &b, T permissible_error,
                                    size_t max_it) {
  using vector_t = Storage<T>;
  using vector_algorithms = csr::vector_algorithms<T, Storage>;
  using matrix_algorithms = csr::matrix_algorithms<T, Storage>;

  auto device_ptr = vector_algorithms::make_device_ptr();
  auto host_ptr = vector_algorithms::make_host_ptr();

  vector_t x(b.size());

  auto r = b;
  vector_algorithms::sub(r, A.dot(x));

  auto p = r;

  T rTr = vector_algorithms::dot(r, r, device_ptr, host_ptr);

  size_t it;

  vector_t Ap(p.size());

  for (it = 0; it < max_it && rTr > permissible_error; ++it) {
    matrix_algorithms::dot(A, p, Ap);

    T a = rTr / vector_algorithms::dot(p, Ap, device_ptr, host_ptr);

    vector_algorithms::add_scaled(x, a, p);
    vector_algorithms::sub_scaled(r, a, Ap);

    T rTr2 = vector_algorithms::dot(r, r, device_ptr, host_ptr);
    T scale = rTr2 / rTr;
    rTr = rTr2;

    vector_algorithms::scale_and_add(p, scale, r);
  }

  vector_algorithms::delete_device_ptr(std::move(device_ptr));
  vector_algorithms::delete_host_ptr(std::move(host_ptr));

  return std::make_pair(std::move(x), it);
}

} // namespace cgm
