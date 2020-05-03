// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/threadpool.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
using concurrency::ThreadPool;

namespace functors {

inline common::Status GetFloatParam(const std::string& name, const onnxruntime::NodeAttributes& attributes,
                                    float& out) {
  auto attr = attributes.find(name);
  if (attr == attributes.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name:'", name, "'is defined.");
  }
  if (attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT) {
    return Status(ONNXRUNTIME, FAIL, "Attibute name and type don't match");
  }
  out = attr->second.f();
  return common::Status::OK();
}

// Like the std::transform
// T should be float or double
template <typename T>
struct ElementWiseRangedTransform {
  using DataType = T;
  const T* input = nullptr;
  T* output = nullptr;
  // Run an unary function through the range [input + first, input + last) -> [output + first, output + last)
  // Thread safe
  virtual void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const = 0;
  // Thread safe
  virtual float Cost() const = 0;
  virtual ElementWiseRangedTransform<T>* Copy() const = 0;
  virtual ~ElementWiseRangedTransform() = 0;
  static Status Create(const std::string& type, const onnxruntime::NodeAttributes& attributes,
                       ElementWiseRangedTransform<T>** out);
};

template <typename T>
ElementWiseRangedTransform<T>::~ElementWiseRangedTransform() {
}
#define ORT_GET_ATTR_AND_RETURN(X)                                 \
  float X;                                                         \
  Status Init(const onnxruntime::NodeAttributes& attributes) {     \
    return (GetFloatParam(#X, attributes, X));                     \
  }                                                                \
  ElementWiseRangedTransform<T>* Copy() const final {              \
    using T1 = typename std::remove_pointer<decltype(this)>::type; \
    using T2 = typename std::remove_const<T1>::type;               \
    return new T2(*this);                                          \
  }

#define ORT_GET_ATTR_AND_RETURN_2(X, Y)                            \
  float X;                                                         \
  float Y;                                                         \
  Status Init(const onnxruntime::NodeAttributes& attributes) {     \
    ORT_RETURN_IF_ERROR(GetFloatParam(#X, attributes, X));         \
    ORT_RETURN_IF_ERROR(GetFloatParam(#Y, attributes, Y));         \
    return Status::OK();                                           \
  }                                                                \
  ElementWiseRangedTransform<T>* Copy() const final {              \
    using T1 = typename std::remove_pointer<decltype(this)>::type; \
    using T2 = typename std::remove_const<T1>::type;               \
    return new T2(*this);                                          \
  }

template <typename T>
struct Elu : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 30.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * (xm.exp() - 1));
  }
};

template <typename T>
struct HardSigmoid : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 0.5f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (((T)alpha * xm + (T)beta).cwiseMin(1.0f)).cwiseMax(0.0f);
  }
};

template <typename T>
struct LeakyRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 25.0f;
  }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * xm);
  }
};

template <typename T>
struct Softplus : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 15.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > 0).select(xm + ((-xm).exp() + 1.0f).log(), ((xm).exp() + 1.0f).log());
  }
};

template <typename T>
struct Relu : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseMax(0);
  }
};

template <typename T>
struct Sigmoid : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 2.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
  }
};

template <>
void Sigmoid<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct Softsign : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (1 + xm.abs()).inverse() * xm;
  }
};

template <typename T>
struct Tanh : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.tanh();
  }
};
template <>
void Tanh<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct ThresholdedRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > (T)alpha).select(xm, 0);
  }
};

template <typename T>
struct Selu : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN_2(alpha, gamma);

  float Cost() const final {
    return 4.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (T)gamma * (xm.cwiseMax(0.0f) + ((T)alpha * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
  }
};

template <typename T>
struct ParametricSoftplus : public ElementWiseRangedTransform<T> {
  ORT_GET_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 15.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (T)alpha *
         (xm * (T)beta > 0)
             .select(xm * (T)beta + ((-xm * (T)beta).exp() + 1.0f).log(), ((xm * (T)beta).exp() + 1.0f).log());
  }
};
}  // namespace functors

template <typename F>
class ElementWiseKernel final : public OpKernel {
 public:
  explicit ElementWiseKernel(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(f_.Init(info.node().GetAttributes()));
  }

  Status Compute(OpKernelContext* context) const override {
    using T = typename F::DataType;
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    if (input_size == 0)
      return Status::OK();
    ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());
    F f = f_;
    f.input = X->template Data<T>();
    f.output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(input_size),
                               {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), f.Cost()}, f);
    return Status::OK();
  }

 private:
  F f_;
};

#define DEFINE_ELE_KERNEL(X, id)               \
  template <typename T>                        \
  using X = ElementWiseKernel<functors::X<T>>; \
  static constexpr int ELEMENTWISE_KERNEL_TYPE_##X = id;

DEFINE_ELE_KERNEL(Elu, 1);
DEFINE_ELE_KERNEL(HardSigmoid, 2);
DEFINE_ELE_KERNEL(LeakyRelu, 3);
DEFINE_ELE_KERNEL(ParametricSoftplus, 4);
DEFINE_ELE_KERNEL(Softplus, 5);
DEFINE_ELE_KERNEL(Relu, 6);
DEFINE_ELE_KERNEL(Sigmoid, 7);
DEFINE_ELE_KERNEL(Softsign, 8);
DEFINE_ELE_KERNEL(Tanh, 9);
DEFINE_ELE_KERNEL(ThresholdedRelu, 10);
DEFINE_ELE_KERNEL(Selu, 11);

#define CREATE_ELE_KERNEL(X)                  \
  if (type == #X) {                           \
    functors::X<T>* p = new functors::X<T>(); \
    p->Init(attributes);                      \
    *out = p;                                 \
    return Status::OK();                      \
  }

namespace functors {
template <typename T>
Status ElementWiseRangedTransform<T>::Create(const std::string& type, const NodeAttributes& attributes,
                                             ElementWiseRangedTransform<T>** out) {
  CREATE_ELE_KERNEL(Elu);
  CREATE_ELE_KERNEL(HardSigmoid);
  CREATE_ELE_KERNEL(LeakyRelu);
  CREATE_ELE_KERNEL(ParametricSoftplus);
  CREATE_ELE_KERNEL(Softplus);
  CREATE_ELE_KERNEL(Relu);
  CREATE_ELE_KERNEL(Sigmoid);
  CREATE_ELE_KERNEL(Softsign);
  CREATE_ELE_KERNEL(Tanh);
  CREATE_ELE_KERNEL(ThresholdedRelu);
  CREATE_ELE_KERNEL(Selu);
  return Status(ONNXRUNTIME, FAIL, "unknown kernel type");
}
}  // namespace functors

}  // namespace onnxruntime
