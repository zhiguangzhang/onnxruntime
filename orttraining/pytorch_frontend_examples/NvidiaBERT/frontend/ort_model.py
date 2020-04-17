import io
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import helper
import torch
import torch.nn
import torch.onnx
import onnxruntime as ort
import pdb

class model_loss_cls(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(model_loss_cls, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn

    def forward(self, input, target):
        preds = self.model_(input)
        return self.loss_fn_(preds, target)


def delete_input_with_name(input, name):
    index = 0
    for i in input:
        if i.name == name:
            del input[index]
            break
        index = index + 1


# reference:
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
# https://pytorch.org/docs/stable/tensors.html
# also must map to types accepted by:
# MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type)
def dtype_torch_to_numpy(torch_dtype):
    if torch_dtype == torch.float64 or torch_dtype == torch.double:
        return np.float64
    elif torch_dtype == torch.float32 or torch_dtype == torch.float:
        return np.float32
    elif torch_dtype == torch.float16 or torch_dtype == torch.half:
        return np.float16
    elif torch_dtype == torch.int64 or torch_dtype == torch.long:
        return np.longlong
    elif torch_dtype == torch.int32 or torch_dtype == torch.int:
        return np.int32
    elif torch_dtype == torch.int16 or torch_dtype == torch.short:
        return np.int16


class ORTModel():
    def __init__(self, model_path, loss_output_name, prediction_name, dynamic_axes, none_train_weights=()):
        super(ORTModel, self).__init__()

        model = onnx.load(model_path)

        self.input_names = [_.name for _ in model.graph.input]
        print(self.input_names)
        self.ort_parameters = ort.TrainingParameters()
        self.ort_parameters.loss_output_name = loss_output_name
        self.ort_parameters.enable_mix_precision = False

        self.torch_params = {}
        self.output_types = {}
        trainable_weights_idx = []
        idx = 0
        for initializer in model.graph.initializer:
            if initializer.name not in none_train_weights:
                tmp = torch.from_numpy(numpy_helper.to_array(initializer)).cuda()
                if tmp.dtype != torch.float32:
                    print("None float parameter found: name: %s, dtype: %s" % (initializer.name, tmp.dtype))
                torch_tensor = torch.nn.Parameter(torch.from_numpy(numpy_helper.to_array(initializer)).cuda())
                delete_input_with_name(model.graph.input, initializer.name)
                model.graph.input.extend(
                    [helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)])
                self.torch_params[initializer.name] = torch_tensor
                trainable_weights_idx.append(idx)
            idx += 1

        trainable_weights_idx.sort(reverse=True)
        for idx in trainable_weights_idx:
            del model.graph.initializer[idx]

        for output in model.graph.output:
            self.output_types[output.name] = output.type.tensor_type

        self.ort_parameters.weights_to_train = set(self.torch_params.keys())

        self.session = ort.TrainingSession(model.SerializeToString(), self.ort_parameters)

        self.train_io_binding = self.session.io_binding()
        self.eval_io_binding = self.session.io_binding()
        self.grad_buffers = {}
        for param in self.torch_params.keys():
            torch_tensor = self.torch_params[param]

            self.train_io_binding.bind_input(param, torch_tensor.device.type, torch_tensor.device.index,
                                             dtype_torch_to_numpy(torch_tensor.dtype), list(torch_tensor.size()),
                                             torch_tensor.data_ptr())
            self.eval_io_binding.bind_input(param, torch_tensor.device.type, torch_tensor.device.index,
                                            dtype_torch_to_numpy(torch_tensor.dtype), list(torch_tensor.size()),
                                            torch_tensor.data_ptr())
            grad_buffer = torch.zeros(torch_tensor.size(), dtype=torch.float32, device='cuda')
            self.grad_buffers[param] = grad_buffer
            # hardcode grad name
            self.train_io_binding.bind_output(param + "_grad",
                                              grad_buffer.device.type,
                                              grad_buffer.device.index,
                                              np.float32,
                                              list(grad_buffer.size()),
                                              grad_buffer.data_ptr())
        self.prediction_name = prediction_name
        self.dynamic_axes = dynamic_axes

    def parameters(self):
        return list(self.torch_params.values())

    def named_parameters(self):
        return list(self.torch_params.items())

    def state_dict(self):
        return self.torch_params

    def load_state_dict(self, state_dict, strict=False):
        for name, param in self.torch_params.items():
            input_param = state_dict[name]
            param.data.copy_(input_param.data)

    def _resolve_dims(self, input_shapes):
        result = {}
        for name in input_shapes:
            if name in self.dynamic_axes.keys():
                shape = input_shapes[name]
                dynamic_dims = self.dynamic_axes[name]
                for index in dynamic_dims.keys():
                    result[dynamic_dims[index]] = shape[index]
        return result

    def train(self):
        return None

    def run(self, inputs, fetches=None):
        if fetches is None:
            fetches = [self.ort_parameters.loss_output_name]

        input_shapes = {}
        feeds = dict(zip(self.input_names, inputs))
        for input_name in feeds:
            input = feeds[input_name]
            self.train_io_binding.bind_input(input_name, input.device.type, input.device.index,
                                         dtype_torch_to_numpy(input.dtype),
                                         list(input.size()), input.data_ptr())
            input_shapes[input_name] = list(input.size())

        dynamic_dims = self._resolve_dims(input_shapes)

        # only loss output is fetched for now.
        torch_outputs = {}
        for fetch_name in fetches:
            dims = self.output_types[fetch_name].shape.dim
            shape = [dynamic_dims[_.dim_param] if _.dim_param else _.dim_value for _ in dims]
            torch_tensor = torch.zeros(shape, device='cuda', dtype=torch.float32)
            self.train_io_binding.bind_output(fetch_name, torch_tensor.device.type, torch_tensor.device.index,
                                              np.float32, list(torch_tensor.size()), torch_tensor.data_ptr())
            torch_outputs[fetch_name] = torch_tensor

        self.session.run_with_iobinding(self.train_io_binding)

        for name, param in self.torch_params.items():
            grad_buffer = self.grad_buffers[name]
            if param.grad is None:
               param.grad = torch.zeros(grad_buffer.size(), device='cuda', dtype=torch.float32)
            param.grad += grad_buffer

        if len(fetches) == 1:
            # TODO: most time it returns loss tensor for caller to report training progress.
            return torch_outputs[fetches[0]]
        else:
            return torch_outputs

    def eval(self, feeds, fetches=None):
        if fetches is None:
            fetches = [self.prediction_name]

        input_shape = {}
        for input_name in feeds:
            input = feeds[input_name]
            self.eval_io_binding.bind_input(self.input_name, input.device.type, input.device.index,
                                        dtype_torch_to_numpy(input.dtype),
                                        list(input.size()), input.data_ptr())
            input_shape[input_name] = list(input.size())

        dynamic_dims = self._resolve_dims(input_shape)

        torch_outputs = {}
        for fetch_name in fetches:
            dims = self.output_types[fetch_name].shape.dim
            shape = [dynamic_dims[_.dim_param] if _.dim_param else _.dim_value for _ in dims]
            torch_tensor = torch.zeros(shape, device='cuda', dtype=torch.float32)
            self.eval_io_binding.bind_output(fetch_name, torch_tensor.device.type, torch_tensor.device.index,
                                             np.float32, list(torch_tensor.size()), torch_tensor.data_ptr())
            torch_outputs[fetch_name] = torch_tensor

        run_options = ort.RunOptions()
        run_options.only_execute_path_to_fetches = True
        self.session.run_with_iobinding(self.eval_io_binding, run_options)
        if len(fetches) == 1:
            # TODO: most time it returns loss tensor for caller to report training progress.
            return torch_outputs[fetches[0]]
        else:
            return torch_outputs
