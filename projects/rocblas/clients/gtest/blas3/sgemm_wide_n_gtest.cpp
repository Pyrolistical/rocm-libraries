#include "rocblas.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <vector>

TEST(SgemmWideN, WritesEveryColumn)
{
    const rocblas_int        n = 524289;
    const float              a = 2.0f;
    const std::vector<float> b(n, 1.0f);
    std::vector<float>       c(n, -7.0f);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    ASSERT_EQ(hipMalloc(&device_a, sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&device_b, n * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&device_c, n * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_a, &a, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_b, b.data(), n * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_c, c.data(), n * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    rocblas_handle handle = nullptr;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    ASSERT_EQ(rocblas_sgemm(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            1,
                            n,
                            1,
                            &alpha,
                            device_a,
                            1,
                            device_b,
                            1,
                            &beta,
                            device_c,
                            1),
              rocblas_status_success);
    ASSERT_EQ(hipMemcpy(c.data(), device_c, n * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(c[524287], 2.0f);
    EXPECT_EQ(c[524288], 2.0f);

    ASSERT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
    ASSERT_EQ(hipFree(device_a), hipSuccess);
    ASSERT_EQ(hipFree(device_b), hipSuccess);
    ASSERT_EQ(hipFree(device_c), hipSuccess);
}
