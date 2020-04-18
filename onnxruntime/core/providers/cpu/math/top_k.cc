/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/providers/cpu/math/top_k.h"
#include "core/providers/common.h"
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;
namespace onnxruntime {

template <typename T>
struct GreaterValueCmp {
  using DataType = T;
  bool operator()(const pair<T, int64_t>& lhs, const pair<T, int64_t>& rhs) {
    return (lhs.first > rhs.first ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

template <typename T>
struct LesserValueCmp {
  using DataType = T;
  bool operator()(const pair<T, int64_t>& lhs, const pair<T, int64_t>& rhs) {
    return (lhs.first < rhs.first ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

// Static helpers that implement the core logic for each of the 'TopK' operator flavor

// Selects the top k elements (largest or smallest based on template parameter)
template <class Comparator>
static void select_top_k(
    const ConstEigenMatrixMapRowMajor<typename Comparator::DataType>& raw_data,
    int64_t row_num, int64_t num_blocks,
    int64_t block_slice, int64_t inter_block_offset, const unsigned k,
    bool sort_top_k,
    vector<pair<typename Comparator::DataType, int64_t>>& data_holder) {
  data_holder.clear();
  data_holder.reserve(num_blocks);

  for (int64_t l = 0; l < num_blocks; ++l) {
    data_holder.push_back({raw_data(row_num, l * block_slice + inter_block_offset), l});
  }

  // find the top k (largest or smallest) elements in the data holder - O(n) average. O(n*n) worst case.
  // See https://en.wikipedia.org/wiki/Quickselect
  Comparator comparator;
  nth_element(data_holder.begin(), data_holder.begin() + (k - 1), data_holder.end(), comparator);

  // sort the top k elements if needed - O (k log k)
  if (sort_top_k) {
    std::sort(data_holder.begin(), data_holder.begin() + k, comparator);
  }

  // the data_holder now contains the top k elements in the first k indices
}

template <class _Ty, class _Pr = less<typename std::vector<_Ty>::value_type>>
class reusable_priority_queue : public std::priority_queue<_Ty, std::vector<_Ty>, _Pr> {
 public:
  reusable_priority_queue() = default;

  const std::vector<_Ty>& unsorted_results() const {
    return std::priority_queue<_Ty, std::vector<_Ty>, _Pr>::c;
  }

  void reset(size_t size) {
    std::priority_queue<_Ty, std::vector<_Ty>, _Pr>::c.clear();
    std::priority_queue<_Ty, std::vector<_Ty>, _Pr>::c.reserve(size);
  }
};

// Given an input tensor 'input' and metadata values - 'k' and 'axis_parsed',
// this method will extract the sorted top k largest/smallest elements and place them in the output tensor 'values'
// along with the metadata output 'indices'
template <class Comparator>
static void extract_top_k_elements(const Tensor* input, const TensorShape& input_shape, Tensor* values,
                                   Tensor* indices, const TensorShape& output_shape, const unsigned k, bool sorted,
                                   const unsigned axis_parsed, concurrency::ThreadPool* threadpool) {
  // Cache some values that will be used in the implementation below
  const int64_t rows = input_shape.SizeToDimension(static_cast<size_t>(axis_parsed));
  const int64_t cols = input->Shape().Size() / rows;
  auto input_map =
      ConstEigenMatrixMapRowMajor<typename Comparator::DataType>(
          static_cast<const typename Comparator::DataType*>(input->template Data<typename Comparator::DataType>()), rows, cols);

  // Use Eigen maps to allow indexing into the 2d tensors like Values_map(i,j)
  const int64_t reduced_cols = output_shape.SizeFromDimension(static_cast<size_t>(axis_parsed));
  auto values_map = EigenMatrixMapRowMajor<typename Comparator::DataType>(
      values->template MutableData<typename Comparator::DataType>(), rows, reduced_cols);
  auto indices_map = EigenMatrixMapRowMajor<int64_t>(indices->template MutableData<int64_t>(), rows, reduced_cols);

  // This is basically the number of elements within each of the "k" rows
  const int64_t block_slice = reduced_cols / k;
  const int64_t num_blocks = input_shape[axis_parsed];

  int64_t tp_threads = threadpool != nullptr ? threadpool->NumThreads() : 1;
  int64_t num_threads = std::min(tp_threads, rows);

  // rough attempt to make sure there's enough work for each thread
  auto total_work = rows * block_slice * num_blocks;
  while (num_threads > 1 && (total_work / num_threads) < 4 * 1024) {
    --num_threads;
  }

  std::function<void(std::ptrdiff_t batch)> find_top_k;

  // from testing various batch sizes relative to k, the following works well as a selector.
  // tested with following combinations
  //   batch_size = [ 8, 16, 32, 64, 128, 256, 512, 1024, 2048 ]
  //            k = [ 1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 128 ]
  bool use_priority_queue = k < 4 || (std::log2(k) / std::log2(num_blocks)) <= 0.70;

  if (use_priority_queue) {
    find_top_k =
        [num_threads, rows, block_slice, num_blocks, k, sorted,
         &input_map, &values_map, &indices_map](std::ptrdiff_t batch) {
          int64_t start_row = static_cast<int64_t>(batch * rows / num_threads);
          int64_t end_row = static_cast<int64_t>((batch + 1) * rows / num_threads);

          reusable_priority_queue<pair<typename Comparator::DataType, int64_t>, Comparator> heap;
          Comparator comparer;

          for (int64_t i = start_row; i < end_row; ++i) {
            for (int64_t j = 0; j < block_slice; ++j) {
              heap.reset(k);

              int64_t l = 0;
              // add first k items
              for (; l < k && l < num_blocks; ++l) {
                heap.push({input_map(i, l * block_slice + j), l});
              }

              for (; l < num_blocks; ++l) {
                std::pair<typename Comparator::DataType, int64_t> item(input_map(i, l * block_slice + j), l);
                if (comparer(item, heap.top())) {
                  heap.pop();
                  heap.push(item);
                }
              }

              if (sorted) {
                // Extract these k elements and place them in the results placeholder
                for (l = 0; l < k; ++l) {
                  const auto& elem = heap.top();
                  auto col_index = (k - l - 1) * block_slice + j;
                  values_map(i, col_index) = elem.first;
                  indices_map(i, col_index) = elem.second;
                  heap.pop();
                }
              } else {
                const auto& results = heap.unsorted_results();
                for (l = 0; l < k; ++l) {
                  const auto& elem = results[l];
                  auto col_index = l * block_slice + j;
                  values_map(i, col_index) = elem.first;
                  indices_map(i, col_index) = elem.second;
                }
              }
            }
          }
        };
  } else {
    find_top_k =
        [num_threads, rows, block_slice, num_blocks, k, sorted,
         &input_map, &values_map, &indices_map](std::ptrdiff_t batch) {
          int64_t start_row = static_cast<int64_t>(batch * rows / num_threads);
          int64_t end_row = static_cast<int64_t>((batch + 1) * rows / num_threads);

          // we re-use a single data_holder for performance. avoids allocating memory on each iteration
          vector<pair<typename Comparator::DataType, int64_t>> data_holder;

          for (int64_t i = start_row; i < end_row; ++i) {
            for (int64_t j = 0; j < block_slice; ++j) {
              select_top_k<Comparator>(input_map, i, num_blocks, block_slice, j, k, sorted, data_holder);

              // Insert the top 'k' (largest or smallest) elements into the final output buffers
              for (int64_t l = 0; l < k; ++l) {
                const auto& elem = data_holder[l];
                auto col_index = l * block_slice + j;
                values_map(i, col_index) = elem.first;
                indices_map(i, col_index) = elem.second;
              }
            }
          }
        };
  }

  if (num_threads == 1) {
    find_top_k(0);
  } else {
    // we want to re-use the priority_queue in each lambda as much as possible, so the lambda does multiple rows.
    // the alternative would be to use TryBatchParallelFor with the lambda doing one row.
    threadpool->SimpleParallelFor(num_threads, find_top_k);
  }
}

// Wrapper over core TopK implementation
template <typename T>
static Status TopKImpl(OpKernelContext* p_op_kernel_context, const Tensor* input, const int axis, const unsigned k,
                       bool largest = true, bool sorted = true) {
  const TensorShape& input_shape = input->Shape();
  // Will return axis_ as is if positive or fixes it in case it is negative
  const auto axis_parsed = HandleNegativeAxis(axis, static_cast<int64_t>(input_shape.NumDimensions()));
  // Check to ensure k is within the bounds of what is available in that specific axis
  if (input_shape[axis_parsed] < k) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k argument [", k,
                           "] should not be greater than specified axis dim value [", input_shape[axis_parsed], "]");
  }

  // Resize output tensors to be the same shape as the input except
  // for the specified dimension ((i.e.) axis_parsed), which will be of size k. E.x. for an input tensor
  // of shape [3, 4, 5] and k=2 with axis_parsed=1, both of the outputs will be shape [3, 2, 5]
  TensorShape output_shape = input_shape;
  output_shape[axis_parsed] = k;
  auto* values = p_op_kernel_context->Output(0, output_shape);
  auto* indices = p_op_kernel_context->Output(1, output_shape);

  if (values == nullptr || indices == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "output count mismatch, expected 2 outputs to be present for TopK operator");
  }

  // no-op - no output buffers to fill - return silently
  if (k == 0) {
    return Status::OK();
  }

  auto* threadpool = p_op_kernel_context->GetOperatorThreadPool();

  if (largest) {
    extract_top_k_elements<GreaterValueCmp<T>>(input, input_shape, values, indices, output_shape, k, sorted,
                                               gsl::narrow_cast<unsigned>(axis_parsed), threadpool);
  } else {
    extract_top_k_elements<LesserValueCmp<T>>(input, input_shape, values, indices, output_shape, k, sorted,
                                              gsl::narrow_cast<unsigned>(axis_parsed), threadpool);
  }

  return Status::OK();
}

// Opset ver - 1 to 9
template <>
TopK<9, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t k_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_temp).IsOK());
  ORT_ENFORCE(k_temp > 0);
  k_ = gsl::narrow_cast<unsigned>(k_temp);

  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 1 to 9
template <>
Status TopK<9, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "input count mismatch, expected 1 input - the tensor to be processed");
  }

  return TopKImpl<float>(p_op_kernel_context, X, axis_, k_);
}

// Opset ver - 10
template <>
TopK<10, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 10
template <>
Status TopK<10, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* Y = p_op_kernel_context->Input<Tensor>(1);
  if (X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "input count mismatch, expected 2 inputs - "
                           "the tensor to be processed and a tensor containing k value");
  }

  const vector<int64_t>& y_shape = Y->Shape().GetDims();
  if (y_shape.size() != 1 || y_shape[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k tensor should be a 1D tensor of size 1");
  }

  auto parsed_input_k = Y->template Data<int64_t>()[0];
  if (parsed_input_k < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "value of k must not be negative");
  }

  return TopKImpl<float>(p_op_kernel_context, X, axis_, gsl::narrow_cast<unsigned>(parsed_input_k));
}

// Opset ver - 11

static void TopkOpset11ConstructorCommon(const OpKernelInfo& op_kernel_info,
                                         int& axis, bool& largest, bool& sorted) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis = gsl::narrow_cast<int>(axis_temp);

  int64_t largest_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("largest", &largest_temp).IsOK());
  largest = largest_temp == 1 ? true : false;

  int64_t sorted_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("sorted", &sorted_temp).IsOK());
  sorted = sorted_temp == 1 ? true : false;
}

template <>
TopK<11, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <>
TopK<11, int64_t>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  TopkOpset11ConstructorCommon(op_kernel_info, axis_, largest_, sorted_);
}

template <typename T>
static Status ComputeImplOpset11(OpKernelContext* p_op_kernel_context, int axis, bool is_largest, bool is_sorted) {
  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* Y = p_op_kernel_context->Input<Tensor>(1);
  if (X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "input count mismatch, expected 2 inputs - "
                           "the tensor to be processed and a tensor containing k value");
  }

  const vector<int64_t>& y_shape = Y->Shape().GetDims();
  if (y_shape.size() != 1 || y_shape[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "k tensor should be a 1D tensor of size 1");
  }

  auto parsed_input_k = Y->template Data<int64_t>()[0];
  if (parsed_input_k < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "value of k must not be negative");
  }

  return TopKImpl<T>(p_op_kernel_context, X, axis, gsl::narrow_cast<unsigned>(parsed_input_k), is_largest, is_sorted);
}

// Opset ver - 11
template <>
Status TopK<11, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset11<float>(p_op_kernel_context, axis_, largest_, sorted_);
}

template <>
Status TopK<11, int64_t>::Compute(OpKernelContext* p_op_kernel_context) const {
  return ComputeImplOpset11<int64_t>(p_op_kernel_context, axis_, largest_, sorted_);
}

// Register necessary kernels
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(TopK, 1, 9,
                                   KernelDefBuilder()
                                       .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                   TopK<9, float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(TopK, 10, 10,
                                   KernelDefBuilder()
                                       .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
                                   TopK<10, float>);

#define REGISTER_TOPK_TYPED_KERNEL(OPSET, TYPE)                                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(TopK,                                                             \
                                 OPSET,                                                            \
                                 TYPE,                                                             \
                                 KernelDefBuilder()                                                \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())     \
                                     .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                 TopK<OPSET, TYPE>);

REGISTER_TOPK_TYPED_KERNEL(11, float);
REGISTER_TOPK_TYPED_KERNEL(11, int64_t);

}  // namespace onnxruntime
