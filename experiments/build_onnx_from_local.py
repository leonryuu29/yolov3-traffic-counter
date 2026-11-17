
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import os

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each block describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0 and x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    blocks = []
    block = {}

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, inp_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.inp_dim = inp_dim
        self.grid = torch.zeros(1)

    def forward(self, x, CUDA=False):
        nB = x.size(0)
        nG = x.size(2)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

        if self.grid.shape[2:4] != x.shape[2:4]:
            self.grid = self._make_grid(nG, CUDA)

        prediction = x.view(nB, self.num_anchors, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        stride = self.inp_dim / nG

        # Add the center offsets
        grid_x = self.grid[:, :, 0].unsqueeze(1).repeat(1, self.num_anchors, 1, 1)
        grid_y = self.grid[:, :, 1].unsqueeze(1).repeat(1, self.num_anchors, 1, 1)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y

        # Log space transform height and width
        anchors = FloatTensor([(a[0] / stride, a[1] / stride) for a in self.anchors])
        anchor_w = anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Final output
        _scale = torch.cat((pred_boxes.view(nB, -1, 4) * stride,
                            conf.view(nB, -1, 1),
                            pred_cls.view(nB, -1, self.num_classes)), -1)
        return _scale

    def _make_grid(self, nG, CUDA):
        grid_x = torch.arange(nG).repeat(nG, 1).view(1, 1, nG, nG).float()
        grid_y = torch.arange(nG).repeat(nG, 1).t().view(1, 1, nG, nG).float()
        if CUDA:
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
        return torch.cat((grid_x, grid_y), 2)

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=False)

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x["type"] == "convolutional":
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", activn)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = Upsample(stride)
            module.add_module(f"upsample_{index}", upsample)

        elif x["type"] == "route":
            layers_str_list = x["layers"].split(',')
            start = int(layers_str_list[0])
            try:
                end = int(layers_str_list[1])
            except:
                end = 0

            if start > 0: start = start - index
            if end > 0: end = end - index

            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        elif x["type"] == "yolo":
            mask = [int(m) for m in x["mask"].split(',')]
            anchors = [int(a) for a in x["anchors"].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            yolo_layer = YOLOLayer(anchors, int(x["classes"]), int(net_info["height"]))
            module.add_module(f"yolo_{index}", yolo_layer)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type in ["convolutional", "upsample"]:
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = [int(l) for l in module["layers"].split(',')]
                if layers[0] > 0: layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0: layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                x = self.module_list[i][0](x, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block['type'] == 'convolutional':
                conv_layer = module[0]
                try:
                    batch_normalize = int(block['batch_normalize'])
                except:
                    batch_normalize = 0

                if batch_normalize:
                    bn_layer = module[1]
                    num_bn_biases = bn_layer.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_biases = bn_biases.view_as(bn_layer.bias.data)
                    bn_weights = bn_weights.view_as(bn_layer.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)

                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.copy_(bn_running_mean)
                    bn_layer.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv_layer.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_biases)

                num_weights = conv_layer.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_weights)

if __name__ == '__main__':
    print("开始进行本地模型转换...")
    cfg_file = "yolov3.cfg"
    weights_file = "yolov3.weights"
    onnx_file = "yolov3.onnx"

    # 加载模型
    print(f"1. 正在从 {cfg_file} 加载模型架构...")
    model = Darknet(cfg_file)

    # 加载权重
    print(f"2. 正在从 {weights_file} 加载模型权重...")
    model.load_weights(weights_file)

    # 设置为评估模式
    model.eval()

    # 获取网络输入尺寸
    inp_dim = int(model.net_info["height"])

    # 创建一个虚拟输入
    print(f"3. 创建一个 {inp_dim}x{inp_dim} 的虚拟输入用于追踪模型...")
    dummy_input = torch.randn(1, 3, inp_dim, inp_dim)

    # 导出到 ONNX
    print(f"4. 正在将模型导出为 {onnx_file}...")
    try:
        torch.onnx.export(model,
                          dummy_input,
                          onnx_file,
                          opset_version=11,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'])
        print("\n转换成功！")
        print(f"模型文件 {onnx_file} 已在当前目录下生成。")
        print("现在您可以运行 traffic_detector_gpu.py 了。")

    except Exception as e:
        print(f"\n转换失败: {e}")
        print("请检查 PyTorch, onnx, onnxruntime 的版本，或者 cfg/weights 文件是否正确。")
