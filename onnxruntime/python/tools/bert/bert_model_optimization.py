################
# v5 - move gamma, beta to input
#
import onnx
import sys
import argparse

from onnx import ModelProto
from onnx import TensorProto
from google.protobuf import text_format

from onnx import numpy_helper
from collections import deque
import numpy as np

VERBOSE = True

class OnnxModel:
    def __init__(self, model):
        self.model = model

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_nodes(self):
        output_name_to_nodes = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                    output_name_to_nodes[output_name] = node
        return output_name_to_nodes

    def nodes(self):
        return self.model.graph.node

    def graph(self):
        return self.model.graph

    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self.model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        self.model.graph.initializer.extend([tensor])
 
    def add_input(self, input):
        self.model.graph.input.extend([input])

    def get_initializer(self,name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None
        
    def nodes_by_type(self, op_type):
        return [n for n in self.model.graph.node if n.op_type == op_type]

    # This only handle the first output
    def find_first_child_by_type(self, node, child_type, input_name_to_node=None):
        if (input_name_to_node is None):
            input_name_to_node = self.input_name_to_nodes()

        children = input_name_to_node[node.output[0]]
        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == child_type:
                return current_node
            children = input_name_to_node[current_node.output[0]]
            for child in children:
                dq.appendleft(child)

        return None

    '''
    def get_children(self, node, input_name_to_node=None):
        if (input_name_to_node is None):
            input_name_to_node = self.input_name_to_nodes()
            
        children = []
        for output in node.output:
            if output in input_name_to_node:
                children.append(input_name_to_node[output])
        return children
    
    def find_first_child_by_type_v2(sel, node, child_type, input_name_to_node=None, recursive = True):
        children = self.get_children(node, input_name_to_node)
        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == child_type:
                return current_node

            if recursive:
                children = self.get_children(current_node, input_name_to_node)
                for child in children:
                    dq.appendleft(child)

        return None
    '''
    def remove_unused_constant(self):
        input_name_to_node = self.input_name_to_nodes()
        
        #remove unused constant
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if node.op_type == "Constant" and node.output[0] not in input_name_to_node:
                unused_nodes.append(node)

        self.remove_nodes(unused_nodes)

        if len(unused_nodes) > 0:
            print("Removed unused constant nodes:", len(unused_nodes))

    def get_subgraph_nodes(self, root_node, stop_nodes, input_name_to_node = None):
        if input_name_to_node is None:
            input_name_to_node = self.input_name_to_nodes()

        children = input_name_to_node[root_node.output[0]]

        unique_nodes = []

        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)

            for output in current_node.output:
                if output in input_name_to_node:
                    children = input_name_to_node[output]
                    for child in children:
                        dq.appendleft(child)

        return unique_nodes

    def convert_model_to_half(self):
        graph = self.model.graph
        initializers = graph.initializer

        for input_value_info in graph.input:
            if input_value_info.type.tensor_type.elem_type == 1:
                input_value_info.type.tensor_type.elem_type = 10

        for output_value_info in graph.output:
            if output_value_info.type.tensor_type.elem_type == 1:
                output_value_info.type.tensor_type.elem_type = 10

        for initializer in initializers:
            if initializer.data_type == 1:
                initializer.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16), initializer.name))

        for node in graph.node:
            if node.op_type == 'Constant':
                for att in node.attribute:
                    if att.name == 'value' and att.t.data_type==1:
                        att.CopyFrom(onnx.helper.make_attribute("value", numpy_helper.from_array(numpy_helper.to_array(att.t).astype(np.float16))))
            if node.op_type == 'Cast':
                for att in node.attribute:
                    if att.name == 'to' and att.i == 1:
                        print(att.name)
                        att.CopyFrom(onnx.helper.make_attribute("to", 10))
   
    # create a new name for node 
    def create_node_name(self, op_type, name_prefix = None):
        nodes = self.nodes_by_type(op_type)

        #TODO: check duplication
        if name_prefix is not None:
            return name_prefix + str(len(nodes))

        return op_type + "_" + str(len(nodes))

    def find_graph_input(self, input_name):
        for input in self.model.graph.input:
            if input.name == input_name:
                return input
        return None

    def get_parent_nodes_and_inputs(self, node, stop_nodes, output_name_to_node = None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_nodes()

        unique_nodes = []
        unique_inputs = []
        
        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])

        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)
                
            for input in current_node.input:
                if input in output_name_to_node:
                    dq.appendleft(output_name_to_node[input])
                elif input not in unique_inputs:
                    print("input", input)
                    unique_inputs.append(input)

        return unique_nodes, unique_inputs
        
    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_nodes()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def find_first_parent_by_type(self, node, parent_type, output_name_to_node=None, recursive = True):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_nodes()
            
        parents = self.get_parents(node, output_name_to_node)
        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == parent_type:
                return current_node

            if recursive:
                parents = self.get_parents(current_node, output_name_to_node)
                for parent in parents:
                    dq.appendleft(parent)

        return None
class BertOnnxModel(OnnxModel):
    def __init__(self, model, normalize_name, num_heads, head_size, batch_size, sequence_length):
        assert(normalize_name == "LayerNormalization" or normalize_name == "SkipLayerNormalization" or normalize_name == "Normalize")
        assert(batch_size >= 0)
        assert(num_heads > 0)
        assert(head_size > 0)
        assert(sequence_length > 0)
        
        super(BertOnnxModel, self).__init__(model)
        self.normalize_name = normalize_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.hidden_size = num_heads * head_size
        self.mask_input = None
        self.embed_node = None

    def get_normalize_nodes(self):
        return self.nodes_by_type(self.normalize_name)

    def normalize_children_types(self):
        if self.normalize_name == "Normalize" or self.normalize_name == "LayerNormalization":
            return ['Add', 'MatMul', 'MatMul', 'MatMul']

        assert(self.normalize_name == "SkipLayerNormalization")
        return ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization']

    def find_transpose_nodes(self, normalize_node, input_name_to_node, output_name_to_node):
        assert(normalize_node.op_type == self.normalize_name)
        assert(len(normalize_node.output) == 1)

        children = input_name_to_node[normalize_node.output[0]]

        transpose_nodes = []
        add_nodes = []

        for child in children:
            if child.op_type == 'MatMul':
                transpose_node = self.find_first_child_by_type(child, 'Transpose', input_name_to_node)
                if transpose_node is None:
                    break
                transpose_nodes.append(transpose_node)

                add_node = self.find_first_child_by_type(child, 'Add', input_name_to_node)
                if add_node is None:
                    break
                add_nodes.append(add_node)

        if len(transpose_nodes) != 3:
            print("Failed to find 3 transpose nodes. Got ", len(transpose_nodes))
            raise Exception("Failed to find transpose nodes.")

        qxk = None
        q_transpose = None
        k_transpose = None
        v_transpose = None
        for transpose_node in transpose_nodes:
            transpose_children = input_name_to_node[transpose_node.output[0]]
            if len(transpose_children) == 1 and transpose_children[0].op_type == 'MatMul' \
               and input_index(transpose_node.output[0], transpose_children[0]) == 0:
                qxk = transpose_children[0]
                q_transpose = output_name_to_node[qxk.input[0]]
                k_transpose = output_name_to_node[qxk.input[1]]

        for transpose_node in transpose_nodes:
            if transpose_node != q_transpose and q_transpose != k_transpose:
                v_transpose = transpose_node

        if qxk is None:
            print("Failed to find qxk")
            raise Exception("Failed to find transpose nodes.")
            
        qkv = self.find_first_child_by_type(qxk, 'MatMul', input_name_to_node)
        qkv_transpose = input_name_to_node[qkv.output[0]][0]
        if qkv_transpose.op_type == "Transpose":
            return qkv_transpose, q_transpose, k_transpose, v_transpose, add_nodes

        raise Exception("Failed to find qkv node")
        return None, None, None, None, None

    def set_mask_input(self, input):
        if self.mask_input is not None and input != self.mask_input:
            raise Exception("Different mask inputs", self.mask_input, input)

        self.mask_input = input

    def transform_attention(self, normalize_node, nodes_to_remove):
        input_name_to_node = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_nodes()
        children = input_name_to_node[normalize_node.output[0]]
        children_types = sorted([child.op_type for child in children])

        if children_types != self.normalize_children_types():
            return

        print("processing attention for normalize node output:", normalize_node.output[0])

        qkv_transpose, q_transpose, k_transpose, v_transpose, add_nodes = self.find_transpose_nodes(normalize_node, input_name_to_node, output_name_to_node)
        if qkv_transpose is None:
            print("failed to find transpose nodes after " + normalize_node.output[0])
            return

        print("qkv_transpose:", qkv_transpose.output)
        reshape_node_after_att = self.find_first_child_by_type(qkv_transpose, 'Reshape')
        if reshape_node_after_att is None:
            print("failed to find reshape node after the qkv_transpose")
            return

        input_name_to_node = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_nodes()

        print("heads", self.num_heads, "head size", self.head_size, "hidden size", self.hidden_size)

        q_matmul = self.find_first_parent_by_type(q_transpose, "MatMul", output_name_to_node)
        k_matmul = self.find_first_parent_by_type(k_transpose, "MatMul", output_name_to_node)
        v_matmul = self.find_first_parent_by_type(v_transpose, "MatMul", output_name_to_node)
        q_weight = self.get_initializer(q_matmul.input[1])
        k_weight = self.get_initializer(k_matmul.input[1])
        v_weight = self.get_initializer(v_matmul.input[1])

        q_add = input_name_to_node[q_matmul.output[0]]
        k_add = input_name_to_node[k_matmul.output[0]]
        v_add = input_name_to_node[v_matmul.output[0]]
        q_bias = self.get_initializer(q_add[0].input[1])
        k_bias = self.get_initializer(k_add[0].input[1])
        v_bias = self.get_initializer(v_add[0].input[1])

        qb = numpy_helper.to_array(q_bias)

        if len(qb) != self.hidden_size:
            raise Exception("cannot resize weights. Use the debug_attention version instead.")

        #TODO: check shape
        qw = numpy_helper.to_array(q_weight)
        kw = numpy_helper.to_array(k_weight)
        vw = numpy_helper.to_array(v_weight)
        qkv_weight = np.stack((qw, kw, vw), axis=-2)
        
        qb = numpy_helper.to_array(q_bias)
        kb = numpy_helper.to_array(k_bias)
        vb = numpy_helper.to_array(v_bias)
        qkv_bias = np.stack((qb, kb, vb), axis=-2)

        subgraph_nodes, input_nodes = self.get_parent_nodes_and_inputs(reshape_node_after_att, [normalize_node], output_name_to_node)

        #do not remove now since mask processing node will be removed.
        #self.remove_node(reshape_node_after_att)
        #self.remove_nodes(subgraph_nodes)
        nodes_to_remove.extend(subgraph_nodes)
        nodes_to_remove.extend([reshape_node_after_att])

        input_nodes = [n for n in input_nodes if self.get_initializer(n) is None]
        if len(input_nodes) != 1:
            print("Failed. Current normalize node output", normalize_node.output[0])
            raise Exception("There should be one input node linked to attention. Got:", input_nodes)
        # Here we assume that attention will get only one graph input: the mask
        
        self.set_mask_input(input_nodes[0])

        attention_node_name = self.create_node_name('Attention')

        weight = onnx.helper.make_tensor(
            name=attention_node_name + '_qkv_weight',
            data_type=TensorProto.FLOAT,
            dims=[self.hidden_size, 3 * self.hidden_size],
            vals=qkv_weight.flatten().tolist())
        self.add_initializer(weight)

        weight_input = onnx.helper.make_tensor_value_info(weight.name, TensorProto.FLOAT, [self.hidden_size, 3 * self.hidden_size])
        self.add_input(weight_input)

        bias = onnx.helper.make_tensor(
            name=attention_node_name + '_qkv_bias',
            data_type=TensorProto.FLOAT,
            dims=[3 * self.hidden_size],
            vals=qkv_bias.flatten().tolist())
        self.add_initializer(bias)

        bias_input = onnx.helper.make_tensor_value_info(bias.name, TensorProto.FLOAT, [3 * self.hidden_size])
        self.add_input(bias_input)

        
        attention_node = onnx.helper.make_node(
            'Attention',
            inputs=[normalize_node.output[0], attention_node_name + '_qkv_weight', attention_node_name + '_qkv_bias', self.mask_input],
            outputs=[reshape_node_after_att.output[0]],
            name=attention_node_name
            )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([
            onnx.helper.make_attribute("num_heads", self.num_heads)
            ])

        self.add_node(attention_node)

    def update_graph(self, verbose=VERBOSE):
        graph = self.model.graph

        remaining_input_names = []
        for node in graph.node:
            if node.op_type != "Constant":
                for input_name in node.input:
                    if input_name not in remaining_input_names:
                        remaining_input_names.append(input_name)
        if verbose:
            print("remaining input names", remaining_input_names)

        # remove graph input that is not used
        inputs_to_remove = []
        for input in graph.input:
            if input.name not in remaining_input_names:
                inputs_to_remove.append(input)
        for input in inputs_to_remove:
            graph.input.remove(input)
        if verbose:
            print("remove unused input ", len(inputs_to_remove), [input.name for input in inputs_to_remove])
        
        # remove weights that are not used
        weights_to_remove = []
        for initializer in graph.initializer:
            if initializer.name not in remaining_input_names:
                weights_to_remove.append(initializer)
        for initializer in weights_to_remove:
            graph.initializer.remove(initializer)
        if verbose:
            print("remove unused initializers:", len(weights_to_remove), [initializer.name for initializer in weights_to_remove])

        if verbose:
            print("--remaining---")
            for node in graph.node:
                print("  node", node.op_type, "input", node.input, "output", node.output)
            print("  initializer", [initializer.name for initializer in graph.initializer])
            print("  input", [input.name for input in graph.input])
            print(" output", [output.name for output in graph.output])

    def fuse_attention(self):
        normalize_nodes = self.nodes_by_type(self.normalize_name)
        nodes_to_remove = []
        for normalize_node in normalize_nodes:
            self.transform_attention(normalize_node, nodes_to_remove)
        self.remove_nodes(nodes_to_remove)
        self.update_graph()

    def fuse_gelu(self):
        nodes = self.nodes()
        input_name_to_node = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_nodes()

        nodes_to_remove=[]
        nodes_to_add=[]

        for node in self.get_normalize_nodes():

            children = input_name_to_node[node.output[0]]
            if len(children) != 2:
                continue

            children_types = sorted([child.op_type for child in children])
            if children_types != ['MatMul', 'SkipLayerNormalization' if self.normalize_name == 'SkipLayerNormalization' else 'Add']:
                continue

            matmul_node = self.find_first_child_by_type(node, 'MatMul', input_name_to_node)
            matmul_child = input_name_to_node[matmul_node.output[0]]
            if len(matmul_child) != 1 or matmul_child[0].op_type != 'Add':
                continue
            add_node = matmul_child[0]
            
            children = input_name_to_node[add_node.output[0]]

            children_types = sorted([child.op_type for child in children])
            if children_types != ['Div', 'Mul']:
                continue

            matmul_2 = self.find_first_child_by_type(add_node, 'MatMul', input_name_to_node)
            if matmul_2 is None:
                continue

            subgraph_nodes = self.get_subgraph_nodes(add_node, [matmul_2], input_name_to_node)
            if len(subgraph_nodes) != 5:
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node(
                'FastGelu',
                inputs=[add_node.output[0]],
                outputs=[matmul_2.input[0]]
                )
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        self.remove_unused_constant()

    def fuse_embed_layer(self):
        nodes = self.nodes()
        input_name_to_node = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_nodes()
        mask_input_name = self.mask_input

        nodes_to_remove=[]
        nodes_to_add=[]

        normalize_node = None
        for node in self.get_normalize_nodes():
            if self.find_first_parent_by_type(node, self.normalize_name, output_name_to_node, recursive=False) is None:
                normalize_node = node
                break

        if normalize_node is None:
            raise Exception("did not find node with op_type", self.normalize_name)

        #This is the first normalize normalize_node
        add_node = self.find_first_parent_by_type(normalize_node, 'Add', output_name_to_node, recursive = False)
        segment_embedding_gather = self.find_first_parent_by_type(normalize_node, 'Gather', output_name_to_node, recursive = False)

        parents = self.get_parents(add_node, output_name_to_node)
        if len(parents) == 2 and parents[0].op_type == 'Gather' and parents[1].op_type == 'Gather':
            word_embedding_gather = parents[0]
            position_embedding_gather = parents[1]

        input_ids = word_embedding_gather.input[1]
        segment_ids = segment_embedding_gather.input[1]

        nodes_to_remove, inputs = self.get_parent_nodes_and_inputs(normalize_node, [], output_name_to_node)

        embed_node = onnx.helper.make_node(
                        'EmbedLayerNormalization',
                        inputs=[input_ids, segment_ids, mask_input_name, 
                                word_embedding_gather.input[0], position_embedding_gather.input[0], segment_embedding_gather.input[0],
                                normalize_node.input[2], normalize_node.input[3]], # gamma and beta
                        outputs=["embed_output", "mask_idx"],
                        name="EmbedLayer"
                    )
        embed_node.domain = "com.microsoft"
        #embed_node.attribute.extend(normalize_node.attribute) #copy gamma and beta attributes from normalize node
        # store embed node for other processing
        self.embed_node = embed_node

        nodes_to_add.extend([embed_node])
        nodes_to_remove.append(normalize_node)

        for n in nodes:
            replace_input(n, normalize_node.output[0], 'embed_output')
            replace_input(n, mask_input_name, 'mask_idx')

        # Remove Cast and ReduceSum for mask, and link mask_idx to attention node directly.
        for n in nodes:
            if 'mask_idx' in n.input and n.op_type != 'Attention':
                delete_and_skip_node(nodes, n, nodes_to_remove, nodes_to_add)
        
        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        self.update_graph()

    def change_input_to_int32(self):
        graph = self.graph()
        inputs = []
        input_map = {}
        for input in self.embed_node.input:
            input_map[input] = onnx.helper.make_tensor_value_info(input, TensorProto.INT32, [self.batch_size if self.batch_size > 0 else 1, self.sequence_length])

        new_graph_inputs = []
        for input in graph.input:
            if input.name in self.embed_node.input:
                print("input", input.name)
                new_graph_inputs.append(input_map[input.name])

        graph_def = onnx.helper.make_graph(
                        graph.node,
                        'int32 inputs',
                        new_graph_inputs,
                        graph.output,
                        initializer=graph.initializer,
                        value_info=graph.value_info
                    )

        # replace model
        self.model = onnx.helper.make_model(graph_def, producer_name='bert model optimizer')

    def update_input_output_as_dynamic(self):
        assert(self.batch_size == 0)

        from onnx.tools import update_model_dims

        dynamic_batch_inputs = {}
        for input in self.model.graph.input:
            if input.name in self.embed_node.input:
                dynamic_batch_inputs[input.name] = ['batch', self.sequence_length]

        print("old output", self.model.graph.output[0])
        variable_length_model = update_model_dims.update_inputs_outputs_dims(self.model, 
            dynamic_batch_inputs,
            {self.model.graph.output[0].name: ['batch', 1]} #TODO: keep the original dims, and only change dimension 0
        )
        print("new output", variable_length_model.graph.output[0])
        self.model = variable_length_model

    def cast_input_to_int32(self):
        model = self.model

        nodes_to_add = []
        for input in self.embed_node.input:
            graph_input = self.find_graph_input(input)
            if graph_input is None:
                continue

            need_cast = graph_input.type.tensor_type.elem_type == TensorProto.INT64
            if need_cast:
                cast_output = input + '_cast32'
                cast_node = onnx.helper.make_node(
                    'Cast',
                    inputs=[input],
                    outputs=[cast_output]
                    )
                cast_node.attribute.extend([
                    onnx.helper.make_attribute("to", int(TensorProto.INT32))
                    ])
                nodes_to_add.extend([cast_node])
                for n in model.graph.node:
                    if input in n.input:
                        replace_input(n, input, input + '_cast32')

        model.graph.node.extend(nodes_to_add)

    def fuse_layerNorm(self):
        graph = self.graph()
        nodes = self.nodes()

        input_name_to_node = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_nodes()

        nodes_to_remove=[]
        nodes_to_add=[]

        for node in nodes:
            # do gather on word directly
            if node.op_type =='Add':
                output_name = node.output[0]
                children = input_name_to_node[output_name]
                sub_count = 0
                reduceMean = 0
                for child in children:
                    if child.op_type == 'Sub':
                        sub_count = sub_count +1
                    if child.op_type == 'ReduceMean':
                        reduceMean = reduceMean + 1
                if sub_count == 2 and reduceMean == 1: #find a normalize pattern
                    nodes_to_remove.extend(children)
                    for child in children:
                        if child.op_type == 'Sub':
                            if input_name_to_node[child.output[0]][0].op_type == 'Div':
                                div_node = input_name_to_node[child.output[0]][0]
                            else:
                                pow_node = input_name_to_node[child.output[0]][0]
                    reduceMean_node = input_name_to_node[pow_node.output[0]][0]
                    first_add_node = input_name_to_node[reduceMean_node.output[0]][0]
                    sqrt_node = input_name_to_node[first_add_node.output[0]][0]
                    mul_node = input_name_to_node[div_node.output[0]][0]
                    second_add_node = input_name_to_node[mul_node.output[0]][0]
                    nodes_to_remove.extend([pow_node, reduceMean_node, first_add_node, sqrt_node, div_node, mul_node, second_add_node])
                    if self.normalize_name == "LayerNormalization":
                        normalize_node = onnx.helper.make_node(
                            'LayerNormalization',
                            inputs=[output_name, mul_node.input[0], second_add_node.input[1]],
                            outputs=[second_add_node.output[0]]
                            )
                        nodes_to_add.extend([normalize_node])
                    else:
                        assert(self.normalize_name == "SkipLayerNormalization")
                        sln_inputs = [i for i in node.input]
                        sln_inputs.extend([mul_node.input[0], second_add_node.input[1]])
                        normalize_node = onnx.helper.make_node(
                            'SkipLayerNormalization',
                            inputs=sln_inputs,
                            outputs=[second_add_node.output[0]],
                            name="SkipLayerNorm_" + second_add_node.output[0]
                        )
                        normalize_node.domain = "com.microsoft"
                        #gamma_attribute = onnx.helper.make_attribute("gamma", self.get_initializer(mul_node.input[0]))
                        #beta_attribute = onnx.helper.make_attribute("beta", self.get_initializer(second_add_node.input[1]))
                        #normalize_node.attribute.extend([gamma_attribute, beta_attribute])
                        nodes_to_add.extend([normalize_node])
                        nodes_to_remove.append(node)

        for to_remove in nodes_to_remove:
            nodes.remove(to_remove)
        nodes.extend(nodes_to_add)

#-----
def input_index(node_output, child_node):
    index = 0
    for input in child_node.input:
        if input == node_output:
            return index
        index += 1
    return -1

def add_or_replace_weight(weights_to_add, graph, weight_tensor):
    # weight existed in graph
    initializer = findTensor(weight_tensor.name, graph)
    if initializer is not None:
        print("replace weight in initializer", weight_tensor.name)
        graph.initializer.remove(initializer)

    # weight existed to be added
    for weight in weights_to_add:
        if weight.name == weight_tensor.name:
            weights_to_add.remove(weight)
            break

    weights_to_add.append(weight_tensor)



def replace_graph_input(graph_inputs, old_input_name, new_input):
    inputs = []
    for input in graph_inputs:
        if input.name != old_input_name:
            inputs.append(input)

    # always add new input
    inputs.append(new_input)
    return inputs

def replace_input(node, old_input, new_input):
    assert(isinstance(new_input, str))
    
    for j in range(len(node.input)):
        if node.input[j] == old_input:
            node.input[j] = new_input

def create_new_node_for_replace_input(node, old_input, new_inputs):
    inputs = []
    for input in node.input:
        if input == old_input:
            for i in new_inputs:
                inputs.append(i)
        else:
            inputs.append(input)

    #TODO: copy node name and attributes
    new_node = onnx.helper.make_node(
        node.op_type,
        inputs=inputs,
        outputs=node.output
        )
    new_node.name = node.name
    new_node.attribute = node.attribute
    return new_node

def delete_and_skip_node(nodes, node, nodes_to_remove, nodes_to_add):
    nodes_to_remove.append(node)

    all_nodes = nodes_to_add + [n for n in nodes if n not in nodes_to_remove]

    # for all children, change their input to parent of current node.
    for node_output in node.output:
        for n in all_nodes:
            if node_output in n.input:
                if len(node.input) == 1:
                    replace_input(n, node_output, node.input[0])
                else:
                    new_node = create_new_node_for_replace_input(n, node_output, node.input)
                    nodes_to_add.append(new_node)
                    if n in nodes:
                        nodes_to_remove.append(n)
                    else:
                        nodes_to_add.remove(n)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)

    # TODO: deduce these from graph
    parser.add_argument('--batch_size', required=False, type=int, default=0)
    parser.add_argument('--num_heads', required=False, type=int, default=12)
    parser.add_argument('--head_size', required=False, type=int, default=64)
    parser.add_argument('--sequence_length', required=False, type=int, default=128)

    parser.add_argument('--fuse_attention', required=False, action='store_true')
    parser.set_defaults(fuse_attention=False)
    parser.add_argument('--fuse_skipLayerNorm', required=False, action='store_true')
    parser.set_defaults(fuse_skipLayerNorm=False)
    parser.add_argument('--fuse_gelu', required=False, action='store_true')
    parser.set_defaults(fuse_gelu=False)
    parser.add_argument('--fuse_embed_layer', required=False, action='store_true')
    parser.set_defaults(fuse_embed_layer=False)
    parser.add_argument('--fuse_all', required=False, action='store_true')
    parser.set_defaults(fuse_all=True)
    parser.add_argument('--input_int32', required=False, action='store_true')
    parser.set_defaults(input_int32=False)
    parser.add_argument('--output_float16', required=False, action='store_true')
    parser.set_defaults(output_float16=False)
    parser.add_argument('--dynamic_batch', required=False, action='store_true')
    parser.set_defaults(dynamic_batch=False)

    args = parser.parse_args()

    if args.fuse_all:
        args.fuse_attention = True
        args.fuse_skipLayerNorm = True
        args.fuse_gelu = True
        args.fuse_embed_layer = True

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    normalize_name =  'SkipLayerNormalization' if args.fuse_skipLayerNorm else 'LayerNormalization'
    
    bert_model = BertOnnxModel(model, normalize_name, args.num_heads, args.head_size, args.batch_size, args.sequence_length)

    bert_model.fuse_layerNorm()

    if args.fuse_gelu:
        bert_model.fuse_gelu()

    if args.fuse_attention:
        bert_model.fuse_attention()

        if args.fuse_embed_layer:
            bert_model.fuse_embed_layer()

            if args.input_int32:
                bert_model.change_input_to_int32()
            else:
                bert_model.cast_input_to_int32()

            #if args.batch_size == 0:
            #    bert_model.update_input_output_as_dynamic()

    bert_model.remove_unused_constant()

    if args.output_float16:
        bert_model.convert_model_to_half()

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

main()