#define BOOST_TEST_MODULE cpu_vector_tests
#include <boost/test/included/unit_test.hpp>
#include <csr/matrix.hpp>
#include <csr/traits.hpp>
#include <vector>

using algorithms = csr::vector_algorithms<int, std::vector>;

BOOST_AUTO_TEST_CASE(vector_norm) {
  auto device_ptr = algorithms::make_device_ptr();
  auto host_ptr = algorithms::make_host_ptr();
  std::vector<int> v1{1, 2, 3, 4};
  const int expected = 30;

  BOOST_CHECK_EQUAL(algorithms::norm(v1, 2, device_ptr, host_ptr), expected);
}

BOOST_AUTO_TEST_CASE(vector_dot) {
  auto device_ptr = algorithms::make_device_ptr();
  auto host_ptr = algorithms::make_host_ptr();
  std::vector<int> v1{1, 2, 3, 4};
  std::vector<int> v2{5, 6, 7, 8};
  const int expected = 70;

  BOOST_CHECK_EQUAL(algorithms::dot(v1, v2, device_ptr, host_ptr), expected);
}

BOOST_AUTO_TEST_CASE(vector_scale) {
  std::vector<int> v{1, 2, 3, 4};
  std::vector<int> expected{2, 4, 6, 8};

  algorithms::scale(v, 2);

  BOOST_CHECK_EQUAL_COLLECTIONS(v.begin(), v.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_add) {
  std::vector<int> v1{1, 2, 3, 4};
  std::vector<int> v2{1, 2, 3, 4};
  std::vector<int> expected{2, 4, 6, 8};

  algorithms::add(v1, v2);

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_add_scaled) {
  std::vector<int> v1{1, 2, 3, 4};
  std::vector<int> v2{1, 2, 3, 4};
  std::vector<int> expected{3, 6, 9, 12};

  algorithms::add_scaled(v1, 2, v2);

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_sub) {
  std::vector<int> v1{1, 2, 3, 4};
  std::vector<int> v2{1, 2, 3, 4};
  std::vector<int> expected{0, 0, 0, 0};

  algorithms::sub(v1, v2);

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_sub_scaled) {
  std::vector<int> v1{2, 4, 6, 8};
  std::vector<int> v2{1, 2, 3, 4};
  std::vector<int> expected{0, 0, 0, 0};

  algorithms::sub_scaled(v1, 2, v2);

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}
