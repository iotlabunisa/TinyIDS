//
// Created by Gennaro on 07/02/2024.
//

#ifndef SIMPLEFFNN_ARRAYUTILS_H
#define SIMPLEFFNN_ARRAYUTILS_H

template<typename T>
inline void getMinMax(T *arr, int len, T *min, T *max) {
    *min = arr[0];
    *max = arr[0];
    for (int i = 0; i < len; ++i) {
        if (arr[i] > *max) {
            *max = arr[i];
        } else if (arr[i] < *min) {
            *min = arr[i];
        }
    }
}

template<typename T>
inline bool in(T *arr, int len, T val) {
    for (int i = 0; i < len; ++i) {
        if (arr[i] == val)
            return true;
    }
    return false;
}

template<typename T>
inline bool notIn(T *arr, int len, T val) {
    return !in(arr, len, val);
}

template<typename T>
inline int getMin(T *arr, int len, T *min) {
    *min = arr[0];
    int index = -1;
    for (int i = 0; i < len; ++i) {
        if (arr[i] < *min) {
            *min = arr[i];
            index = i;
        }
    }
    return index;
}

template<typename T>
inline int getMax(T *arr, int len, T *max) {
    *max = arr[0];
    int index = -1;
    for (int i = 0; i < len; ++i) {
        if (arr[i] > *max) {
            *max = arr[i];
            index = i;
        }
    }
    return index;
}

template<typename T>
inline T getMean( T *arr,  int len) {
    T sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i];
    }
    return sum / len;
}

#endif //SIMPLEFFNN_ARRAYUTILS_H
