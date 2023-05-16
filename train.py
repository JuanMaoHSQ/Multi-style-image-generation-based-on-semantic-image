### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# coding=UTF-8
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchsummary
import data.golbal as gol
#from  tensorboardX  import SummaryWriter

if __name__ == '__main__':

    #writer=SummaryWriter()
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)  #开始时期，第几次迭代
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:    
        start_epoch, epoch_iter = 1, 0

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10
    #test
    #opt.print_freq = 1
    #opt.display_freq = 1

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    #print("cuda ::::::",torch.cuda.device_count())

    gol.init()
    gol.set_value('nums',0)
    gol.set_value('add',0)




    for epoch in range(start_epoch, 1 + opt.niter_decay):  #开始时期，与总迭代次数之间
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
            
        a=gol.get_value('nums')
        b=gol.get_value('add')
        if epoch>1:
            a=(a+dataset_size//opt.batchSize)%opt.styleNums
            b=(b+dataset_size//opt.batchSize//opt.styleNums)%opt.styleSize
            gol.set_value('nums',a)
            gol.set_value('add',b)

        
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta   #是否需要显示生成图像

            ############## Forward Pass ######################
            #print(Variable(data['label']).device,Variable(data['inst']).device,Variable(data['image']).device, Variable(data['feat']).device)
            #model=model.cuda()
            #print('# $$$$$$$$$$$$$$$$$###########################$$$$$4 sum parameters:', sum(param.numel() for param in model.parameters()))
            #try:
            losses, generated = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), infer=save_fake)   #数据经模型返回损失值与生成图像
                #print("*************************************",data['label'].shape,data['feat'].shape)
                # sum per device losses
                
            #except:
            #    continue
            # calculate final loss scalar 计算最终损失
            #d_r=nn.Sigmoid(loss_dict['D_real']-loss_dict['D_fake'])
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ] #对于损失中的x，若x为int，则保留x，否则使用x的平均值
            loss_dict = dict(zip(model.module.loss_names, losses))  #将其打包成元组并创建成字典
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_dict['G_GAN']=loss_dict['G_GAN'].cuda(opt.liby[1])
            loss_dict['G_GAN_Feat']= loss_dict['G_GAN_Feat'].cuda(opt.liby[1])
            if opt.style_loss==1:
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)+loss_dict.get('G_style',0)#*epoch*1.0/100
            else:
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
            #若存在，取键值，否则默认值0
            #writer.add_scalars('scalar/test',{'loss_D':loss_D,'G_GAN':loss_dict['G_GAN'],'G_GAN_Feat':loss_dict['G_GAN_Feat'],'loss_G':loss_G,'epoch':epoch},epoch*1000+epoch_iter/dataset_size*1000)
                ############### Backward Pass ####################
                # d generator weights
            model.module.optimizer_G.zero_grad()  #清除所有优化的梯度
            loss_G.backward()
            model.module.optimizer_G.step()  #执行单个优化步骤
                # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()    

            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
            #if epoch%100==0:
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
               
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
            #if epoch%50==0:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                    ("style_image",util.tensor2im(data['feat'][0])),
                                    ('synthesized_image', util.tensor2im(generated.data[0])),
                                    ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
            #if 1!=1:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest%d_%d'%(epoch/2,total_steps))            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 时期结尾再次记录时间
        iter_end_time = time.time()
        if epoch%100==-1:
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:  #根据频率判断该时期是否保存模型
        #if epoch%1500==0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save('latest')
            model.module.save(epoch/2)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations  微调
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations  开始学习率迭代次数达到后，开始线性衰减学习
        if epoch > opt.niter:
            model.module.update_learning_rate()
