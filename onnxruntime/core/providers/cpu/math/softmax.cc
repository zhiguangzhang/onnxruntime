// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_config.h"
//Ignore a wired warning in gcc 7.4.0. The latest gcc doesn't generate this warning
#ifdef __GNUC__
#ifdef HAS_MAYBE_UNINITIALIZED
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#include "core/providers/cpu/math/softmax.h"

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/eigen_common_wrapper.h"

namespace onnxruntime {
template class Softmax<float>;
template class Softmax<double>;

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Softmax, 1, 11, float,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                         Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(LogSoftmax, 1, 11, float,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                         Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Softmax, 1, 11, double,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                                         Softmax<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(LogSoftmax, 1, 11, double,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                                         Softmax<double>);

}  // namespace onnxruntime
