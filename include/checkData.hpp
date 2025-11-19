#pragma once

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include <cuda_runtime.h>

const float ERROR_THRESHOLD_EPSILON = 1e-6f;
const float ERROR_THRESHOLD_MSE_EPSILON = 1e-4f;

template<typename T>
inline bool checkOneData(const T data1, const T data2) {
    return data1 == data2;
}

template<>
inline bool checkOneData<float>(const float data1, const float data2) {
    constexpr float ABS_EPSILON = 1e-5f;

    const float absDiff = std::fabs(data1 - data2);
    if (absDiff < ABS_EPSILON) return true;

    const float maxVal = std::max(std::max(std::fabs(data1), std::fabs(data2)), ERROR_THRESHOLD_EPSILON);
    return (absDiff / maxVal) < ERROR_THRESHOLD_EPSILON;
}

template<>
inline bool checkOneData<double>(const double data1, const double data2) {
    constexpr double ABS_EPSILON = 1e-5;

    const double absDiff = std::fabs(data1 - data2);
    if (absDiff < ABS_EPSILON) return true;

    const double maxVal = std::max(std::max(std::fabs(data1), std::fabs(data2)),
                                   static_cast<double>(ERROR_THRESHOLD_EPSILON));
    return (absDiff / maxVal) < ERROR_THRESHOLD_EPSILON;
}


// 计算均方误差（MSE）来比较两个数据数组
template<typename T>
inline float computeMSE(const T *data1, const T *data2, size_t size) {
    double mse = 0.0;
#pragma omp parallel for reduction(+:mse)
    for (size_t i = 0; i < size; ++i) {
        double diff = static_cast<double>(data1[i]) - static_cast<double>(data2[i]);
        mse += diff * diff;
    }
    if (size == 0) return 0.0f;
    return static_cast<float>(mse / static_cast<double>(size));
}

template<typename T>
inline float computeMSE(const std::vector<T> &data1, const std::vector<T> &data2) {
    if (data1.size() != data2.size() || data1.empty()) return 0.0f;
    return computeMSE(data1.data(), data2.data(), data1.size());
}

bool checkMSE(const std::vector<float> &data1,
              const std::vector<float> &data2,
              float threshold = ERROR_THRESHOLD_MSE_EPSILON,
              const std::string &name = ""
) {
    float mse = computeMSE(data1, data2);
    if (mse > threshold) {
        printf("Error, %s failed: mse = %f, threshold = %f\n", name.c_str(), mse, threshold);
        return false;
    }
    return true;
}

template<typename T>
inline bool checkDataFunction(const size_t num, const T *data1, const T *data2, size_t &numError) {
    bool isCorrect = true;

    printf("|---------------------------check data---------------------------|\n");
    printf("| Data size : %ld\n", num);
    printf("| Error threshold epsilon : %f\n", ERROR_THRESHOLD_EPSILON);
    printf("| Checking results...\n");

    size_t errors = 0;
    for (int idx = 0; idx < num; ++idx) {
        const T oneData1 = data1[idx];
        const T oneData2 = data2[idx];
        if (!checkOneData(oneData1, oneData2)) {
            ++errors;
            if (errors < 10) {
                printf("| Error : idx = %d, data1 = %f, data2 = %f, difference = %f\n",
                       idx,
                       static_cast<float>(oneData1),
                       static_cast<float>(oneData2),
                       static_cast<float>(oneData1 - oneData2));
            }
        }
    }
    numError = errors;
    if (errors > 0) {
        printf("| No Pass! Inconsistent data! %zu errors! Error rate : %2.2f%%\n",
               errors, static_cast<float>(errors) / static_cast<float>(num) * 100);
        isCorrect = false;
    } else {
        printf("| Pass! Result validates successfully.\n");
    }

    printf("|----------------------------------------------------------------|\n");

    return isCorrect;
}

template<typename T>
inline bool checkData(const std::vector<T> &hostData1, const std::vector<T> &hostData2) {
    if (hostData1.size() != hostData2.size()) {
        return false;
    }
    size_t numError;
    return checkDataFunction(hostData1.size(), hostData1.data(), hostData2.data(), numError);
}

template<typename T>
inline bool checkData(const std::vector<T> &hostData1, const std::vector<T> &hostData2, size_t &numError) {
    if (hostData1.size() != hostData2.size()) {
        return false;
    }
    return checkDataFunction(hostData1.size(), hostData1.data(), hostData2.data(), numError);
}

template<typename T>
bool checkData(const std::vector<T> &hostData1, const dev::vector<T> &devData2) {
    std::vector<T> hostData2;
    d2h(hostData2, devData2);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const std::vector<T> &hostData1, const dev::vector<T> &devData2, size_t &numError) {
    std::vector<T> hostData2;
    d2h(hostData2, devData2);
    return checkData(hostData1, hostData2, numError);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2) {
    std::vector<T> hostData1;
    d2h(hostData1, devData1);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const dev::vector<T> &devData2) {
    std::vector<T> hostData1 = d2h(devData1);
    std::vector<T> hostData2 = d2h(devData2);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2, size_t &numError) {
    std::vector<T> hostData1;
    d2h(hostData1, devData1);
    return checkData(hostData1, hostData2, numError);
}
