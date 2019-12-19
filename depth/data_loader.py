import cv2
import csv
import torch
import numpy as np
from random import random, shuffle, randint
from torch.utils.data import DataLoader, Dataset
from django.db.models import Sum, Count
from django.db.models import Q
from data_labelling.models import DepthImage, FRONT_NORMAL_CAMERA_TYPE_V2, FRONT_ZOOM_CAMERA_TYPE_V2
torch.set_num_threads(5)


class DepthDataset(Dataset) :
    """
    Implementation of Pytorch way of loading data in parallel instead of a single threaded training loop
    """
    
    def __init__(self, H, W) :
        
        self.qs                 = DepthImage.objects.filter(Q(camera_type=FRONT_NORMAL_CAMERA_TYPE_V2) | Q(camera_type=FRONT_ZOOM_CAMERA_TYPE_V2))
        self.length             = self.qs.count()
        self.image_labels       = list(self.qs)
        self.H                  = H
        self.W                  = W
        self.DISTANCE_UNIT      = 10 # 10 m = 1 unit
    
    
    @property
    def writer(self):
        return csv.writer(open('offending_image_labels.csv'), delimiter=',')
    
    def __len__(self) :
        return self.length
    
    
    def __getitem__(self, index) :
        try :
            
            image_label = self.image_labels[index]
            speed = image_label.obd_speed * 1000/3600
            time = (image_label.image3.get_timestamp() - image_label.image2.get_timestamp())
            distance = speed * time / self.DISTANCE_UNIT
            image1 = image_label.image1.get_image_data()
            
            image2 = image_label.image2.get_image_data()
            
            image3 = image_label.image3.get_image_data()
            
            image4 = image_label.image4.get_image_data()
            
            object_mask1 = np.zeros_like(image1)
            object_mask1 = image_label.draw_mask(object_mask1, image_label.object_mask1)[:, :, 0]
            object_mask2 = np.zeros_like(image2)
            object_mask2 = image_label.draw_mask(object_mask2, image_label.object_mask2)[:, :, 0]
            object_mask3 = np.zeros_like(image3)
            object_mask3 = image_label.draw_mask(object_mask3, image_label.object_mask3)[:, :, 0]
            object_mask4 = np.zeros_like(image4)
            object_mask4 = image_label.draw_mask(object_mask4, image_label.object_mask4)[:, :, 0]
            
#             image1 = cv2.GaussianBlur(image1,(5,5),cv2.BORDER_DEFAULT)
#             image2 = cv2.GaussianBlur(image2,(5,5),cv2.BORDER_DEFAULT)
#             image3 = cv2.GaussianBlur(image3,(5,5),cv2.BORDER_DEFAULT)
#             image4 = cv2.GaussianBlur(image4,(5,5),cv2.BORDER_DEFAULT)
            
            kernel = np.ones((21, 21), np.uint8)
            object_mask1 = cv2.dilate(object_mask1, kernel, iterations=1)
            object_mask2 = cv2.dilate(object_mask2, kernel, iterations=1)
            object_mask3 = cv2.dilate(object_mask3, kernel, iterations=1)
            object_mask4 = cv2.dilate(object_mask4, kernel, iterations=1)
            
            object_mask1 = np.array(object_mask1 > 0.1, dtype='uint8')
            object_mask2 = np.array(object_mask2 > 0.1, dtype='uint8')
            object_mask3 = np.array(object_mask3 > 0.1, dtype='uint8')
            object_mask4 = np.array(object_mask4 > 0.1, dtype='uint8')
            
            image1, object_mask1 = self.reshape(image1, object_mask1)
            image2, object_mask2 = self.reshape(image2, object_mask2)
            image3, object_mask3 = self.reshape(image3, object_mask3)
            image4, object_mask4 = self.reshape(image4, object_mask4)
            

            image1 = torch.ByteTensor(image1.transpose((2, 0, 1)).copy())
            mask1 = torch.ByteTensor(object_mask1.copy())

            image2, mask2 = torch.ByteTensor(image2.transpose((2, 0, 1)).copy()), torch.ByteTensor(object_mask2.copy())
            image3, mask3 = torch.ByteTensor(image3.transpose((2, 0, 1)).copy()), torch.ByteTensor(object_mask3.copy())
            image4, mask4 = torch.ByteTensor(image4.transpose((2, 0, 1)).copy()), torch.ByteTensor(object_mask4.copy())

            return image1, image2, image3, image4, mask1, mask2, mask3, mask4, torch.FloatTensor([distance])
        except Exception as e:
            print(e)
            print(image_label.pk)
            
    
    def reshape(self, image, mask) :
        image = cv2.resize(image, (self.W, self.H))
        mask = cv2.resize(mask, (self.W, self.H))
        return image, mask