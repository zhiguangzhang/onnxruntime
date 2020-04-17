import torch
from optimization import BertLAMB
import numpy as np

def CreatePytorchLAMB(parameter, name, decay, lr):
    optimizer_grouped_parameters=[{'params': [parameter], 'weight_decay': decay, 'name': name}]

    return BertLAMB(optimizer_grouped_parameters,
                            lr=lr,
                            t_total=-1)

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
def CreateOnnxLAMB(name, shape, decay, lr, alpha, beta, epsilon, threshold):
    input_grad = helper.make_tensor_value_info('grad', TensorProto.FLOAT, shape)
    grad_norm = helper.make_tensor_value_info('grad_norm', TensorProto.FLOAT, [])
    reduce_node = helper.make_node('ReduceAllL2', ['grad'], ['grad_norm'])

    update_signal = helper.make_tensor_value_info('all_finite', TensorProto.BOOL, [1])
    lr = helper.make_tensor_value_info('lr', TensorProto.FLOAT, [1])
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
    m1 = helper.make_tensor_value_info('m1', TensorProto.FLOAT, shape)
    m2 = helper.make_tensor_value_info('m2', TensorProto.FLOAT, shape)

    new_weight = helper.make_tensor_value_info('new_weight', TensorProto.FLOAT, shape)
    new_m1 = helper.make_tensor_value_info('new_m1', TensorProto.FLOAT, shape)
    new_m2 = helper.make_tensor_value_info('new_m2', TensorProto.FLOAT, shape)
    lamb_node = helper.make_node('LambOptimizer', ['all_finite', '', 'grad_norm', 'lr', 'input', 'grad', 'm1', 'm2', ''],
            ['new_weight', '', 'new_m1', 'new_m2', ''],
            **{'alpha':[alpha], 'beta':[beta], 'lambda':[decay], 'epsilon':[epsilon], 'threshold':[threshold]}
            )
    #add a sqrt node, to fetch
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
    fake_node = helper.make_node('Sqrt', ['new_weight'], ['output'])
    graph_def = helper.make_graph([reduce_node, lamb_node, fake_node], 'test-model', [input, input_grad, update_signal, lr, m1, m2],[output], value_info=[grad_norm, new_m1, new_m2, new_weight])
    model_def = helper.make_model(graph_def, producer_name='test')
    #onnx.checker.check_model(model_def)
    return model_def, ['input', 'grad', 'all_finite', 'lr', 'm1', 'm2'], ['output']

import onnxruntime as ort

class OrtLAMB():
    def __init__(self, model, parameter, lr):
        self.sess_ = ort.InferenceSession(model.SerializeToString())
        self.io_binding_ = self.sess_.io_binding()
        #other tensors
        self.all_finite_ = torch.ones([1], dtype=torch.uint8)
        self.lr_ = torch.tensor([lr])
        self.m1_ = torch.zeros_like(parameter.grad).cuda()
        self.m2_ = torch.zeros_like(parameter.grad).cuda()

        self.io_binding_.bind_input('all_finite', self.all_finite_.device.type, 0, np.bool_, list(self.all_finite_.size()), self.all_finite_.data_ptr())
        self.io_binding_.bind_input('lr', self.lr_.device.type, 0, np.float32, list(self.lr_.size()), self.lr_.data_ptr())
        self.io_binding_.bind_input('m1', self.m1_.device.type, 0, np.float32, list(self.m1_.size()), self.m1_.data_ptr())
        self.io_binding_.bind_input('m2', self.m2_.device.type, 0, np.float32, list(self.m2_.size()), self.m2_.data_ptr())
        #output
        self.output_ = torch.zeros(parameter.size()).cuda()
        self.new_weight_ = torch.zeros(parameter.size()).cuda()
        self.io_binding_.bind_output('output', self.output_.device.type, 0, np.float32, list(self.output_.size()), self.output_.data_ptr())
        self.parameter_ = parameter

    def step(self, parameter):
        self.io_binding_.bind_input('input', parameter.device.type, 0, np.float32, list(parameter.size()), parameter.data_ptr())
        self.io_binding_.bind_input('grad', parameter.grad.device.type, 0, np.float32, list(parameter.grad.size()), parameter.grad.data_ptr())
        #
        self.sess_.run_with_iobinding(self.io_binding_)

def lamb_test(epoch):
    device = torch.device('cuda')
    p = torch.nn.Parameter(torch.from_numpy(np.random.random((4096, 4096)).astype(np.float32)).to(device), requires_grad=True)
    optimizer = CreatePytorchLAMB(p, 'test_param', 0.01, 3e-3)
    p.grad = torch.rand(p.size(), device=device)

    p2 = torch.nn.Parameter(torch.from_numpy(np.random.random((4096, 4096)).astype(np.float32)).to(device), requires_grad=True)
    p2.grad = torch.rand(p2.size(), device=device)
    model, inputs, outputs = CreateOnnxLAMB('test_param', [4096, 4096], 0.01, 3e-3, 0.9, 0.999, 1e-6, 1.0)
    ort_lamb = OrtLAMB(model, p2, 3e-3)
    for _ in range(epoch):
        weight = np.random.random((4096, 4096)).astype(np.float32)
        grad = np.random.random((4096, 4096)).astype(np.float32)
        p.data = torch.from_numpy(weight).cuda() 
        p.grad.data = torch.from_numpy(grad).cuda()
        optimizer.step()
        torch_w = p.data.cpu().numpy()

        p2.data = torch.from_numpy(weight).cuda()
        p2.grad.data = torch.from_numpy(grad).cuda()
        ort_lamb.step(p2)
        ort_w = p2.data.cpu().numpy()
        if np.allclose(torch_w, ort_w) is False:
            print('Failed')
    
lamb_test(10)
print('Finished')
