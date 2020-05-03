// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion) \
  ONNX_CPU_OPERATOR_KERNEL(                                             \
      alias, sinceVersion,                                              \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

namespace functors{
template <>
void Sigmoid<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  float* output_ptr = output + first;
  MlasComputeLogistic(input + first, output_ptr, static_cast<size_t>(len));
}

template <>
void Tanh<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  float* output_ptr = output + first;
  MlasComputeTanh(input + first, output_ptr, static_cast<size_t>(len));
}
}

}  // namespace onnxruntime
