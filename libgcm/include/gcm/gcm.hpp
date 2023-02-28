#pragma once

#include <csr/matrix.hpp>
#include <csr/utils.hpp>
#include <cmath>

namespace gcm {

template <typename T, template <typename...> typename Storage>
std::pair<csr::vector<T, Storage>, size_t>
solve(const csr::matrix<T, Storage> &A, const csr::vector<T, Storage> &b,
      T permissible_error, size_t max_it) {

  csr::vector<T, Storage> x(b.width());
  csr::vector<T, Storage> Adotx = A.dot(x);
  csr::vector<T, Storage> r = b - Adotx;
  auto p = r;

  T rTr = r.dot(r);

  size_t it;

  for (it = 0; it < max_it && r.norm(2) > permissible_error; ++it) {
    csr::vector<T, Storage> Ap = A.dot(p);

    T a = rTr / p.dot(Ap);

    x = x + a * p;

    r = r - a * Ap;

    T rTr2 = r.dot(r);
    T scale = rTr2 / rTr;
    rTr = rTr2;

    p *= scale;
    p = p + r;
  }

  return std::make_pair(x, it);
}

} // namespace gcm
