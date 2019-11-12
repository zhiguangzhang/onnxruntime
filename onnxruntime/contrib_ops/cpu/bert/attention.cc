// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/schema.h"
#include "core/util/eigen_common_wrapper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/softmax.h"
#include "core/providers/cpu/tensor/transpose.h"

#define ATTENTION_IMPL_B

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)

AttentionBase::AttentionBase(const OpKernelInfo& info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
}

Status AttentionBase::CheckInputs(const OpKernelContext* context) const {
  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Input 1 - weights     : (hidden_size, 3 * hidden_size)
  //   Input 2 - bias        : (3 * hidden_size)
  //   Input 3 - mask_index  : (batch_size)
  //   Output                : (batch_size, sequence_length, hidden_size)

  const Tensor* input = context->Input<Tensor>(0);
  const auto dims = input->Shape().GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 3 dimensions, got ", dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int hidden_size = static_cast<int>(dims[2]);
  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 dimension 2 should be divisiable by value of the num_heads attribute.");
  }

  const Tensor* weights = context->Input<Tensor>(1);
  const auto weights_dims = weights->Shape().GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 is expected to have 2 dimensions, got ", weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }
  if (weights_dims[1] != 3 * weights_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 1 should be 3 times of dimension 0");
  }

  const Tensor* bias = context->Input<Tensor>(2);
  const auto bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 is expected to have 1 dimension, got ", bias_dims.size());
  }
  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 dimension 0 should have same length as dimension 1 of input 1");
  }

  const Tensor* mask_index = context->Input<Tensor>(3);
  const auto mask_dims = mask_index->Shape().GetDims();
  if (mask_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 3 is expected to have 1 dimension, got ", mask_dims.size());
  }
  if (static_cast<int>(mask_dims[0]) != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Inputs 3 and 0 shall have same length at dimension 0");
  }

  return Status::OK();
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionBase(info) {}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);

  const auto dims = input->Shape().GetDims();
  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  int hidden_size = static_cast<int>(dims[2]);
  int head_size = hidden_size / num_heads_;

  TensorShape output_shape(dims);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  const bool is_profiler_enabled = context->Profiler() && context->Profiler()->IsEnabled();
  TimePoint timepoint;

  if (is_profiler_enabled) {
    timepoint = context->Profiler()->StartTime();
  }

#if defined(ATTENTION_IMPL_A)

  // STEP.1: gemm_data(BS, 3NH) = input(BS, NH) x weights(NH, 3NH) + bias(3NH)

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  auto gemm_data = allocator->Alloc(batch_size * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto gemm_data_mat = EigenMatrixMapRowMajor<T>(reinterpret_cast<T*>(gemm_data), m, n);
  gemm_data_mat.rowwise() = ConstEigenVectorMap<T>(bias->template Data<T>(), n).transpose();

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_1_of_6_broadcast",
                                               timepoint);
    timepoint = context->Profiler()->StartTime();
  }

  math::Gemm<T>(
      CblasNoTrans,
      CblasNoTrans,
      m,
      n,
      k,
      1.0f,
      input->template Data<T>(),
      weights->template Data<T>(),
      1.0f,
      reinterpret_cast<T*>(gemm_data),
      tp);

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_1_of_6_gemm",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

  // STEP.2: gemm_data_transposed(3, B, N, S, H) = transpose gemm_data(B, S, 3, N, H)
  auto gemm_data_transposed = allocator->Alloc(batch_size * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_transposed_buffer(gemm_data_transposed, BufferDeleter(allocator));

  Tensor gemm_data_tensor{input->DataType(), TensorShape{batch_size, sequence_length, 3, num_heads_, head_size}, gemm_data, allocator->Info()};
  Tensor gemm_data_transposed_tensor{input->DataType(), TensorShape{3, batch_size, num_heads_, sequence_length, head_size}, gemm_data_transposed, allocator->Info()};

  static const std::vector<size_t> transpose_permutations{2, 0, 3, 1, 4};
  ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(transpose_permutations, gemm_data_tensor, gemm_data_transposed_tensor));

  T* Q = reinterpret_cast<T*>(gemm_data_transposed);
  T* K = Q + (batch_size * hidden_size * sequence_length);
  T* V = K + (batch_size * hidden_size * sequence_length);

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_2_of_6",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

#elif defined(ATTENTION_IMPL_B)

  // STEP.1: gemm_data(BS, 3NH) = input(BS, NH) x weights(NH, 3NH) + bias(3NH)
  auto gemm_data = allocator->Alloc(batch_size * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));
  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;
  // auto Q = allocator->Alloc(batch_size * sequence_length * hidden_size * element_size);
  // auto K = allocator->Alloc(batch_size * sequence_length * hidden_size * element_size);
  // auto V = allocator->Alloc(batch_size * sequence_length * hidden_size * element_size);
  // BufferUniquePtr Q_buffer(Q, BufferDeleter(allocator));
  // BufferUniquePtr K_buffer(K, BufferDeleter(allocator));
  // BufferUniquePtr V_buffer(V, BufferDeleter(allocator));

  T* QKV[3] = {reinterpret_cast<T*>(Q), reinterpret_cast<T*>(K), reinterpret_cast<T*>(V)};

  // broadcast 3NH -> (3.B.N.S.H)
  {
    int loop_len = 3 * batch_size * num_heads_ * sequence_length;

    if (false && tp != nullptr) {
      tp->ParallelFor(loop_len, [num_heads = num_heads_,
                                                               bias_data = bias->template Data<T>(),
                                                               sequence_length,
                                                               batch_size,
                                                               head_size,
                                                               hidden_size,
                                                               QKV](int32_t i) {
        int seq_index = i % sequence_length;
        int rest = i / sequence_length;
        int head_index = rest % num_heads;
        rest /= num_heads;
        int batch_index = rest % batch_size;
        rest /= batch_size;
        int qkv_index = rest;

        T* data_dest = QKV[qkv_index] + ((batch_index * num_heads + head_index) * sequence_length + seq_index) * head_size;
        const T* data_src = bias_data + qkv_index * hidden_size + head_index * head_size;
        memcpy(data_dest, data_src, head_size * sizeof(T));
      });
    } else {
      const T* data_src = bias->template Data<T>();
      for (int qkv_index = 0; qkv_index < 3; qkv_index++) {
        T* data_dest = QKV[qkv_index];
        for (int batch_index = 0; batch_index < batch_size; batch_index++) {
          for (int head_index = 0; head_index < num_heads_; head_index++) {
            for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
              memcpy(data_dest, data_src, head_size * element_size);
              data_dest += head_size;
            }
            data_src += head_size;
          }
          data_src -= hidden_size;
        }
        data_src += hidden_size;
      }
    }
  }

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_1_of_6_broadcast",
                                               timepoint);
    timepoint = context->Profiler()->StartTime();
  }

  // gemm
  {
    int loop_len = 3 * batch_size * num_heads_;

    if (tp != nullptr) {
      tp->ParallelFor(loop_len, [num_heads = num_heads_,
                                 sequence_length,
                                 head_size,
                                 hidden_size,
                                 input_data = input->template Data<T>(),
                                 weights_data = weights->template Data<T>(),
                                 QKV](int32_t i) {
        int batch_index = (i / 3) / num_heads;
        int head_index = (i / 3) % num_heads;
        int qkv_index = i % 3;

        int input_offset = batch_index * sequence_length * hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        T* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads + head_index) * (sequence_length * head_size);

        //                   original           transposed            iteration
        // A: input          (BxSxNxH)          (B.)S x NH            S x NH
        // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

        math::GemmEx(CblasNoTrans,                                        // TransA = no
                     CblasNoTrans,                                        // TransB = no
                     sequence_length,                                     // M      = S
                     head_size,                                           // N      = H
                     hidden_size,                                         // K      = NH
                     1.0f,                                                // alpha
                     input_data + input_offset,                           // A
                     hidden_size,                                         // lda    = NH
                     weights_data + weights_offset,                       // B
                     3 * hidden_size,                                     // ldb    = 3NH
                     1.0f,                                                // beta
                     qkv_dest + qkv_offset,                               // C
                     head_size,                                           // ldc
                     reinterpret_cast<concurrency::ThreadPool*>(nullptr)  // use single-thread
        );
      });
    } else {
      const T* weights_data = weights->template Data<T>();
      for (int qkv_index = 0; qkv_index < 3; qkv_index++) {
        const T* input_data = input->template Data<T>();
        T* qkv_data = QKV[qkv_index];
        for (int batch_index = 0; batch_index < batch_size; batch_index++) {
          for (int head_index = 0; head_index < num_heads_; head_index++) {
            math::GemmEx(CblasNoTrans,                                        // TransA = no
                         CblasNoTrans,                                        // TransB = no
                         sequence_length,                                     // M      = S
                         head_size,                                           // N      = H
                         hidden_size,                                         // K      = NH
                         1.0f,                                                // alpha
                         input_data,                                          // A
                         hidden_size,                                         // lda    = NH
                         weights_data,                                        // B
                         3 * hidden_size,                                     // ldb    = 3NH
                         1.0f,                                                // beta
                         qkv_data,                                            // C
                         head_size,                                           // ldc
                         reinterpret_cast<concurrency::ThreadPool*>(nullptr)  // use single-thread
            );

            weights_data += head_size;
            qkv_data += sequence_length * head_size;
          }
          input_data += sequence_length * hidden_size;
          weights_data -= head_size * num_heads_;
        }
        weights_data += hidden_size;
      }
    }
  }

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_1_of_6_gemm",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

#endif

  // STEP.3: scratch(B, N, S, S) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S, H -> B, N, H, S) + 1 x mask_index(B -> B, 1, 1, 1)
  auto scratch_data = allocator->Alloc(batch_size * num_heads_ * sequence_length * sequence_length * element_size);
  BufferUniquePtr scratch_buffer(scratch_data, BufferDeleter(allocator));

  {
    memset(scratch_data, 0, batch_size * num_heads_ * sequence_length * sequence_length * element_size);
    auto mask_data = mask_index->template Data<int>();
    int size_each_batch = num_heads_ * sequence_length;
    T* p_current_data = reinterpret_cast<T*>(scratch_data);
    for (int b_i = 0; b_i < batch_size; b_i++) {
      int mask = mask_data[b_i];
      for (int n_i = 0; n_i < size_each_batch; n_i++) {
        for (int m_i = mask; m_i < sequence_length; m_i++) {
          p_current_data[m_i] = static_cast<T>(-10000.0);
        }
        p_current_data += sequence_length;
      }
    }
  }

  {
    float alpha = 1.0f / sqrt(static_cast<float>(head_size));

    if (tp != nullptr) {
      tp->ParallelFor(batch_size * num_heads_, [this, sequence_length, head_size, alpha, Q, K, scratch_data, tp](int32_t i) {
        math::Gemm<T>(
            CblasNoTrans,
            CblasTrans,
            sequence_length,
            sequence_length,
            head_size,
            alpha,
            Q + sequence_length * head_size * i,
            K + sequence_length * head_size * i,
            1.0,
            reinterpret_cast<T*>(scratch_data) + sequence_length * sequence_length * i,
            tp);
      });
    } else {
      int offset_Q = 0;
      int offset_Q_increment = sequence_length * head_size;
      int offset_scratch = 0;
      int offset_scratch_increment = sequence_length * sequence_length;

      for (int b_i = 0; b_i < batch_size; b_i++) {
        for (int n_i = 0; n_i < num_heads_; n_i++) {
          math::Gemm<T>(
              CblasNoTrans,
              CblasTrans,
              sequence_length,
              sequence_length,
              head_size,
              alpha,
              Q + offset_Q,
              K + offset_Q,
              1.0,
              reinterpret_cast<T*>(scratch_data) + offset_scratch,
              tp);
          offset_Q += offset_Q_increment;
          offset_scratch += offset_scratch_increment;
        }
      }
    }
  }

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_3_of_6",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

  // STEP.4: P(B, N, S, S) = Softmax(scratch)
  auto p_data = allocator->Alloc(batch_size * num_heads_ * sequence_length * sequence_length * element_size);
  BufferUniquePtr p_buffer(p_data, BufferDeleter(allocator));

  {
    int N = batch_size * num_heads_ * sequence_length;
    int D = sequence_length;

    Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> X_tensor(
        reinterpret_cast<T*>(scratch_data), N, D);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> Y_tensor(
        reinterpret_cast<T*>(p_data), N, D);
#ifndef USE_OPENMP
    if (tp == nullptr)
#endif
      ComputeSoftMax(Eigen::DefaultDevice(), X_tensor, Y_tensor, false);
#ifndef USE_OPENMP
    else
      ComputeSoftMax(Eigen::ThreadPoolDevice(&tp->GetHandler(), tp->NumThreads()), X_tensor, Y_tensor, false);
#endif
  }

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_4_of_6",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

  // STEP.5: out_tmp(B, N, S, H) = P(B, N, S, S) x V(B, N, S, H)
  auto out_tmp_data = allocator->Alloc(batch_size * num_heads_ * sequence_length * head_size * element_size);
  BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

  {
    if (tp != nullptr) {
      tp->ParallelFor(batch_size * num_heads_, [this, sequence_length, head_size, p_data, out_tmp_data, V, tp](int32_t i) {
        math::MatMul<T>(
            sequence_length,
            head_size,
            sequence_length,
            reinterpret_cast<T*>(p_data) + sequence_length * sequence_length * i,
            V + sequence_length * head_size * i,
            reinterpret_cast<T*>(out_tmp_data) + sequence_length * head_size * i,
            tp);
      });
    } else {
      int offset_p = 0;
      int offset_p_increment = sequence_length * sequence_length;
      int offset_V = 0;
      int offset_V_increment = sequence_length * head_size;

      for (int b_i = 0; b_i < batch_size; b_i++) {
        for (int n_i = 0; n_i < num_heads_; n_i++) {
          math::MatMul<T>(
              sequence_length,
              head_size,
              sequence_length,
              reinterpret_cast<T*>(p_data) + offset_p,
              V + offset_V,
              reinterpret_cast<T*>(out_tmp_data) + offset_V,
              tp);
          offset_p += offset_p_increment;
          offset_V += offset_V_increment;
        }
      }
    }
  }

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_5_of_6",
                                               timepoint);

    timepoint = context->Profiler()->StartTime();
  }

  // STEP.6: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
  Tensor out_tmp_tensor{input->DataType(), TensorShape{batch_size, num_heads_, sequence_length, head_size}, out_tmp_data, allocator->Info()};
  Tensor output_tensor{input->DataType(), TensorShape{batch_size, sequence_length, num_heads_, head_size}, output->template MutableData<T>(), allocator->Info()};

  static const std::vector<size_t> transpose_out_permutations{0, 2, 1, 3};
  ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(transpose_out_permutations, out_tmp_tensor, output_tensor));

  if (is_profiler_enabled) {
    context->Profiler()->EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                               "Custom_Attention_STEP_6_of_6",
                                               timepoint);
  }

  return Status::OK();
}  // namespace contrib

}  // namespace contrib
}  // namespace onnxruntime
