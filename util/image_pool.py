import random
import torch
from torch.autograd import Variable
class ImagePool():
    def __init__(self, pool_size):  #创建图像缓冲池（空）
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):  #若缓冲池大小为0，直接返回图片列表；否则，返回可能包含原缓冲池中的图像的列表
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)  #在第零维上增加一个维度，即在原列表上的所有元素外再加一个括号
            if self.num_imgs < self.pool_size:  #缓冲还未满，则放入缓冲池，并加入到返回图像列表中
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:                   #若缓冲池已满，则50%几率随机替换其中一个，并将被替换图像加入返回列表 ；或者 直接将图像放入返回列表
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))   #按第0维度拼接，即不同张量（列表）的元素放到同一列表中 （去掉加的括号？？？）  并转化为variable
        return return_images
