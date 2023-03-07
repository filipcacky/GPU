#define BOOST_TEST_MODULE gpu_matrix_tests
#include <boost/test/included/unit_test.hpp>
#include <csr/matrix.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "little_bit_of_indirection.cuh"

BOOST_AUTO_TEST_CASE(mx_dot_vector) {
  thrust::host_vector<int> mx_data = std::vector{1, 1, 1, 2, 2, 2, 3, 3, 3};
  csr::matrix<int, thrust::host_vector> mx = csr::from_dense(mx_data, {3, 3});
  thrust::host_vector<int> vector = std::vector{1, 1, 1};
  const thrust::host_vector<int> expected = std::vector{3, 6, 9};

  csr::matrix<int, thrust::device_vector> mx_device =
      mx.transfer<thrust::device_vector>();
  thrust::device_vector<int> vector_device = vector;

  thrust::device_vector<int> result_device = indirection::dot(mx_device, vector_device);

  thrust::host_vector<int> result = result_device;

  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(),
                                expected.end());
}
