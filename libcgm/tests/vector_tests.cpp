#define BOOST_TEST_MODULE gpu_vector_tests
#include <boost/test/included/unit_test.hpp>
#include <csr/matrix.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "little_bit_of_indirection.cuh"

BOOST_AUTO_TEST_CASE(vector_norm) {
  thrust::host_vector<int> v1 = std::vector{1, 2, 3, 4, 5};
  const int expected = 55;

  BOOST_CHECK_EQUAL(indirection::norm(thrust::device_vector<int>(v1), 2),
                    expected);
}

BOOST_AUTO_TEST_CASE(vector_dot) {
  thrust::host_vector<int> v1 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> v2 = std::vector{5, 6, 7, 8};
  const int expected = 70;

  BOOST_CHECK_EQUAL(indirection::dot(thrust::device_vector<int>(v1),
                                     thrust::device_vector<int>(v2)),
                    expected);
}

BOOST_AUTO_TEST_CASE(vector_scale) {
  thrust::host_vector<int> v = std::vector{1, 2, 3, 4};
  const thrust::host_vector<int> expected = std::vector{2, 4, 6, 8};

  thrust::device_vector<int> v_device = v;

  indirection::scale(v_device, 2);

  v = v_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(v.begin(), v.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_add) {
  thrust::host_vector<int> v1 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> v2 = std::vector{1, 2, 3, 4};
  const thrust::host_vector<int> expected = std::vector{2, 4, 6, 8};

  thrust::device_vector<int> v1_device = v1;
  thrust::device_vector<int> v2_device = v2;

  indirection::add(v1_device, v2_device);

  v1 = v1_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_add_scaled) {
  thrust::host_vector<int> v1 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> v2 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> expected = std::vector{3, 6, 9, 12};

  thrust::device_vector<int> v1_device = v1;
  thrust::device_vector<int> v2_device = v2;

  indirection::add_scaled(v1_device, 2, v2_device);

  v1 = v1_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_sub) {
  thrust::host_vector<int> v1 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> v2 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> expected = std::vector{0, 0, 0, 0};

  thrust::device_vector<int> v1_device = v1;
  thrust::device_vector<int> v2_device = v2;

  indirection::sub(v1_device, v2_device);

  v1 = v1_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(vector_sub_scaled) {
  thrust::host_vector<int> v1 = std::vector{2, 4, 6, 8};
  thrust::host_vector<int> v2 = std::vector{1, 2, 3, 4};
  thrust::host_vector<int> expected = std::vector{0, 0, 0, 0};

  thrust::device_vector<int> v1_device = v1;
  thrust::device_vector<int> v2_device = v2;

  indirection::sub_scaled(v1_device, 2, v2_device);

  v1 = v1_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), expected.begin(),
                                expected.end());
}
