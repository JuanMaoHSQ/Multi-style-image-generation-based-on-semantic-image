### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

if __name__ == '__main__':

    opt = TestOptions().parse(save=False)      #命令  操作
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)         #根据操作创建数据集
    dataset = data_loader.load_data()          #数据集为接口
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))    #创建网页
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)     #创建模型
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:   #详细信息，输出模型
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    for i, data in enumerate(dataset):  #列出数据下标和数据
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:  #扩展ONNX模型，数据在不同框架间转移
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                              opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:
             with torch.no_grad():
                generated = model.inference(data['label'], data['inst'], data['image'])

        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),  #张量转为标签
                               ('synthesized_image', util.tensor2im(generated.data[0]))])  #张量转为图片   根据放入的元素的先后顺序排序
        img_path = data['path']
        print('process image... %s' % img_path)  #输入的图像路径
        visualizer.save_images(webpage, visuals, img_path)  #将文件路径与结果放入网页

    webpage.save()
