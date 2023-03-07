#define BOOST_TEST_MODULE cpu_matrix_tests
#include <boost/test/included/unit_test.hpp>
#include <csr/matrix.hpp>
#include <csr/traits.hpp>
#include <vector>

using algorithms = csr::matrix_algorithms<int, std::vector>;

BOOST_AUTO_TEST_CASE(mx_dot_vector) {
  std::vector<int> mx_data{1, 0, 0, 0, 2, 0, 0, 0, 3};
  auto mx = csr::from_dense(mx_data, {3, 3});
  std::vector<int> vector{1, 1, 1};

  std::vector<int> expected{1, 2, 3};

  std::vector<int> result;
  algorithms::dot(mx, vector, result);

  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}
