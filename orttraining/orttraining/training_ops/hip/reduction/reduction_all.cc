// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/hip/reduction/reduction_all.h"

namespace onnxruntime {
namespace hip {

#define REGISTER_REDUCE_ALL_KERNEL_TYPED(Name, TIn, TOut)                                                                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                                \
      Name,                                                                                                                                     \
      kOnnxDomain,                                                                                                                              \
      9,                                                                                                                                        \
      TIn##_##TOut,                                                                                                                             \
      kHipExecutionProvider,                                                                                                                   \
      KernelDefBuilder().TypeConstraint("TIn", DataTypeImpl::GetTensorType<TIn>()).TypeConstraint("TOut", DataTypeImpl::GetTensorType<TOut>()), \
      Name<TIn, TOut>);

template <typename TIn, typename TOut>
Status ReduceAllL2<TIn, TOut>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<TIn>::MappedType HipTIn;
  typedef typename ToHipType<TOut>::MappedType HipTOut;

  // Get Input tensor count.
  const auto total_tensor_count = ctx->InputCount();
  // We only have one tensor per group so
  // grouped_tensor_pointers[i] always contains only one element.
  std::vector<std::vector<void*>> grouped_tensor_pointers(total_tensor_count);
  std::vector<int> tensor_sizes(total_tensor_count);

  for (int i = 0; i < total_tensor_count; ++i) {
    const Tensor* input = ctx->Input<Tensor>(i);
    const auto size = input->Shape().Size();
    ORT_ENFORCE(size <= std::numeric_limits<int>::max(), "Number of reduced elements (",
                size, ") exceeds the max allowed value (", std::numeric_limits<int>::max(), ").");
    grouped_tensor_pointers[i] = {const_cast<TIn*>(input->Data<TIn>())};
    tensor_sizes[i] = static_cast<int>(size);
  }

  // Allocate output tensor.
  Tensor* output = ctx->Output(0, {});
  HipTOut* p_output = reinterpret_cast<HipTOut*>(output->template MutableData<TOut>());
  ORT_ENFORCE(hipMemset(p_output, 0, sizeof(HipTOut)) == hipSuccess);

  typedef MultiTensorReduceL2<HipTIn, HipTOut> TFunctor;
  TFunctor functor;

  // Check if all values are finite and write true to deviceOutput.
  // Otherwise, false will be written.
  launch_multi_tensor_functor<1, TFunctor, HipTOut*>(
      2048 * 32, tensor_sizes, grouped_tensor_pointers, functor, p_output);

  // *p_output is the squared sum of all elements.
  // Let's take a sqrt to get the actual L2-norm.
  ScalarSqrt(p_output, p_output);

  return Status::OK();
}

REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, MLFloat16)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, MLFloat16)

}  // namespace hip
}  // namespace onnxruntime