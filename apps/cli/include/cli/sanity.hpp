#pragma once

#include <boost/log/trivial.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <random>
#include <type_traits>

using namespace fmt::literals;
namespace fs = std::filesystem;

namespace cli {

template <typename T> class Badge {
  friend T;
  Badge() = default;
};

#define NO_COPY(T)                                                             \
  T(const T &) = delete;                                                       \
  T &operator=(const T &) = delete

#define NO_MOVE(T)                                                             \
  T(T &&) = delete;                                                            \
  T &operator=(T &&) = delete

#define UNUSED(variable) (void)variable;

#define LOG(level)                                                             \
  BOOST_LOG_TRIVIAL(level) << fmt::format("[{}:{}] ", __FUNCTION__, __LINE__)

template <typename T> auto ensure(T &&v) {
  if (not v.has_value()) {
    throw std::logic_error("Precondition not met.");
  }
  return std::move(v.value());
}

#define var(v) #v ""_a = v

#define PURE = 0

struct const_change {};

template <typename T>
bool fuzzy_compare(T a, T b, T eps = std::numeric_limits<T>::epsilon()) {
  a = std::abs(a);
  b = std::abs(b);
  return std::abs(a - b) <= std::max(a, b) * eps;
}

template <typename T, typename V> class temporary_change {
public:
  temporary_change(T &variable, V value)
      : variable_(variable), old_value_(variable) {
    variable_ = value;
  }

  temporary_change(const_change, const T &variable, V value)
      : variable_(const_cast<T &>(variable)), old_value_(variable) {
    variable_ = value;
  }

  ~temporary_change() { variable_ = old_value_; }

  T current_value() const { return variable_; }
  V old_value() const { return old_value_; }

private:
  T &variable_;
  V old_value_;
};

} // namespace cli
