// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cross_entropy.h"
#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/loss/reduction_type.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2>
class SoftmaxCrossEntropyLoss final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLoss(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLoss);
};

template <typename T1, typename T2>
class SoftmaxCrossEntropyLossGrad final : public LossBase {
 public:
  explicit SoftmaxCrossEntropyLossGrad(const OpKernelInfo& info) : LossBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxCrossEntropyLossGrad);
};

}  // namespace contrib
}  // namespace onnxruntime