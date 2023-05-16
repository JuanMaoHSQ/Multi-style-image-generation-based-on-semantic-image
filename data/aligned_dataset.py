### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np
import torch
import random
from data.golbal import get_value
from data.golbal import set_value

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_style:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 

      
    def __getitem__(self, index):        
        ### input A (label maps)
        #print("index:::::::::",index)
        A_path = self.A_paths[index]              
        #print(A_path)
        A = Image.open(A_path)        #打开图片
        params = get_params(self.opt, A.size)   #获取参数
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)   #操作，参数图片编辑  转换方式
            A_tensor = transform_A(A.convert('RGB'))      #对RGB格式图片处理后转为张量
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')          #RGB格式打开图片
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)                 #系列操作后转为张量

        ### if using instance maps        
        if not self.opt.no_instance:                       #使用实例图
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_style:                  #特征   ？？？
                #feat_path = self.feat_paths[index % 3]    
                a=get_value('nums')
                b=get_value('add')
                a=a+1
                if   a>=self.opt.styleNums:
                    a=0
                    b=(b+1)%self.opt.styleSize
                    #print(a,b,self.feat_paths[b])
                set_value('nums',a)
                set_value('add',b)
                #print(a,b)
                feat_path = self.feat_paths[b]            
                feat = Image.open(feat_path).convert('RGB')
                feat2=feat.resize((512, 256),Image.ANTIALIAS)
                #feat1=np.array(feat)
                #feat1=torch.from_numpy(cv2.resize(feat1, (1024,512)))
                #print(feat.shape)
                norm = normalize()
                feat_tensor= norm(transform_A(feat2))
                #print(feat_tensor.shape)
                #feat_tensor =cv2.resize(feat_tensor1, (1024,512))  
                #print("feat_tensor::::::",feat_tensor.shape)              

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'