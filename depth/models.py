import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.set_num_threads(1)

class ConvBatch(nn.Module) :
    def __init__(self, H, W, in_channels, out_channels, stride=1, dropout_rate=0.2, upsample=1) :
        super(ConvBatch, self).__init__()
        conv_ker = 30
        self.H = H
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv8 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv9 = nn.Conv2d(in_channels=conv_ker, out_channels=conv_ker, kernel_size=(5,5), stride=1, padding=2)
        self.conv10 = nn.Conv2d(in_channels=conv_ker, out_channels=out_channels, kernel_size=(5,5), stride=stride, padding=2)
        if upsample > 1 :
            self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=False)
        else :
            self.upsample = None
    
    def forward(self, x) :
        activation = F.elu
        x = self.norm(x)
        x = activation(self.conv1(x))
        x = activation(self.conv2(x))
        x = self.dropout1(x)
        x = activation(self.conv3(x))
        x = activation(self.conv4(x))
        x = activation(self.conv5(x))
        x = self.dropout2(x)
        x = activation(self.conv6(x))
        x = activation(self.conv7(x))
        x = self.dropout3(x)
        x = activation(self.conv8(x))
        x = activation(self.conv9(x))
        if self.upsample is not None:
            x = self.upsample(x)
        
        x = activation(self.conv10(x))
        return x

class DepthModel(nn.Module) :
    def __init__(self, H, W, dropout_rate=0.2) :
        super(DepthModel, self).__init__()
        self.depth_scale = 1.
        ## Encoder
        self.H = H
        self.W = W
        h = H
        w = W
        
        b1 = 20
        s1 = 5
        self.batch1 = ConvBatch(H=h, W=w, in_channels=6, out_channels=b1, stride=s1) # (b1, H/s1, W/s1)
        h = h/s1
        w = w/s1
        
        b2 = 40
        s2 = 2
        self.batch2 = ConvBatch(H=h, W=w, in_channels=b1, out_channels=b2, stride=s2) # (b2, H/(s1 * s2), W/(s1*s2))
        h = h/s2
        w = w/s2
        
        b3 = 80
        s3 = 2
        self.batch3 = ConvBatch(H=h, W=w, in_channels=b2, out_channels=b3, stride=s3) # (b3, H/(s1 * s2 * s3), W/(s1*s2*s3))
        h = h/s3
        w = w/s3
        
        self.conv1 = nn.Conv2d(in_channels=b3, out_channels=4, kernel_size=(5,5), padding=2, stride=1)
        self.fc1 = nn.Linear(4 * h * w, 4 * h * w)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(4 * h * w, 4 * h * w)
        self.fc3 = nn.Linear(4 * h * w, 4 * h * w)
        
        ## Decoder
        
        b4 = 20
        self.batch4 = ConvBatch(H=h, W=w, in_channels=b3 + 4, out_channels=b4, upsample=s3) # (b4, H/(s1 * s2), W/(s1*s2))
        h = h * s3
        w = w * s3
        
        b5 = 10
        self.batch5 = ConvBatch(H=h, W=w, in_channels=b4 + b2, out_channels=b5, upsample=s2) # (b5, H/(s1), W/(s1))
        h = h * s2
        w = w * s2
        
        b6 = 8
        self.batch6 = ConvBatch(H=h, W=w, in_channels=b5 + b1, out_channels=b6, upsample=s1) # (b6, H, W)
        h = h * s1
        w = w * s1
        
        b7 = 8
        self.batch7 = ConvBatch(H=h, W=w, in_channels=b6 + 6, out_channels=b7) # (b7, H, W)
        self.depth = nn.Conv2d(in_channels=b7, out_channels=1, kernel_size=(5,5), padding=2, stride=1)
        
    
    def forward(self, img1, img2) :
        """
        Depth will be estimated for img2 frame and not img1
        """
        img = torch.cat((img1, img2), dim=1)
        b1 = self.batch1(img)
        b2 = self.batch2(b1)
        b3 = self.batch3(b2)
        x = F.elu(self.conv1(b3))
        x = x.view(-1, self.fc1.in_features)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        channels = self.batch4.in_channels - self.batch3.out_channels
        h, w = self.batch4.H, self.batch4.W
        embedding = x.view(-1, channels, h, w)
        x = torch.cat((b3, embedding), dim=1)
        b4 = self.batch4(x)
        x = torch.cat((b2, b4), dim=1)
        b5 = self.batch5(x)
        x = torch.cat((b1, b5), dim=1)
        b6 = self.batch6(x)
        x = torch.cat((img, b6), dim=1)
        b7 = self.batch7(x)
        depth = self.depth(b7)
        out = F.softplus(depth) * self.depth_scale
        return out

class MotionModel(nn.Module) :
    def __init__(self, n) :
        super(MotionModel, self).__init__()
        self.rotation_scale = 1e-2
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n)
        self.rotation = nn.Linear(n, 3)
        self.translation = nn.Linear(n, 3)
    
    def forward(self, x) :
        x = F.elu(self.fc1(x))
        translation = F.tanh(self.translation(x))
        x = F.elu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        rotation = self.rotation(x) * self.rotation_scale
        return rotation, translation

class IntrinsicsModel(nn.Module) :
    def __init__(self, n, H, W) :
        super(IntrinsicsModel, self).__init__()
        self.skew_scale = 1e-3
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 5)
        self.H = H
        self.W = W
    
    def forward(self, x) :
        x = F.elu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        intrinsics = torch.cat((F.softplus(x[:, :1]) * self.W, 
                                F.softplus(x[:, 1:2]) * self.H,
                                F.sigmoid(x[:, 2:3]) * self.W,
                                F.sigmoid(x[:, 3:4]) * self.H,
                                x[:, 4:] * self.skew_scale
                               ) , dim=1)
        return intrinsics
    
class PoseModel(nn.Module) :
    def __init__(self, H, W, dropout_rate=0.2) :
        super(PoseModel, self).__init__()
        ## Encoder
        self.H = H
        self.W = W
        h = H
        w = W
        
        b1 = 20
        s1 = 5
        self.batch1 = ConvBatch(H=h, W=w, in_channels=6, out_channels=b1, stride=s1) # (b1, H/s1, W/s1)
        h = h/s1
        w = w/s1
        
        b2 = 40
        s2 = 2
        self.batch2 = ConvBatch(H=h, W=w, in_channels=b1, out_channels=b2, stride=s2) # (b2, H/(s1 * s2), W/(s1*s2))
        h = h/s2
        w = w/s2
        
        b3 = 80
        s3 = 2
        self.batch3 = ConvBatch(H=h, W=w, in_channels=b2, out_channels=b3, stride=s3) # (b3, H/(s1 * s2 * s3), W/(s1*s2*s3))
        h = h/s3
        w = w/s3
        
        self.conv1 = nn.Conv2d(in_channels=b3, out_channels=4, kernel_size=(5,5), padding=2, stride=1)
        self.fc1 = nn.Linear(4 * h * w, 4 * h * w)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(4 * h * w, 4 * h * w)
        self.fc3 = nn.Linear(4 * h * w, 4 * h * w)
        self.bottleneck_size = 4 * h * w
        self.motion = MotionModel(self.bottleneck_size)
        self.intrinsics = IntrinsicsModel(n=self.bottleneck_size, H=H, W=W)
        
        ## Decoder
        b4 = 40
        self.batch4 = ConvBatch(H=h, W=w, in_channels=b3 + 4 + 3, out_channels=b4, upsample=s3) # (b4, H/(s1 * s2), W/(s1*s2))
        h = h * s3
        w = w * s3
        
        b5 = 20
        self.batch5 = ConvBatch(H=h, W=w, in_channels=b4 + b2, out_channels=b5, upsample=s2) # (b5, H/(s1), W/(s1))
        h = h * s2
        w = w * s2
        
        b6 = 10
        self.batch6 = ConvBatch(H=h, W=w, in_channels=b5 + b1, out_channels=b6, upsample=s1) # (b6, H, W)
        h = h * s1
        w = w * s1
        
        b7 = 8
        self.batch7 = ConvBatch(H=h, W=w, in_channels=b6 + 6, out_channels=b7) # (b7, H, W)
        
        self.foreground_translation = nn.Conv2d(in_channels=b7, out_channels=3, kernel_size=(5,5), padding=2, stride=1)
    
    def forward(self, img1, img2) :
        """
        Depth will be estimated for img2 frame and not img1
        """
        N, _, H, W = img1.shape
        img = torch.cat((img1, img2), dim=1)
        b1 = self.batch1(img)
        b2 = self.batch2(b1)
        b3 = self.batch3(b2)
        x = F.elu(self.conv1(b3))
        x = x.view(-1, self.fc1.in_features)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        fc3 = F.elu(self.fc3(x))
        
        rotation, background_translation = self.motion(fc3)
        intrinsics = self.intrinsics(fc3)
        
        channels = self.batch4.in_channels - self.batch3.out_channels - 3
        h, w = self.batch4.H, self.batch4.W
        motion_refinement = background_translation.reshape(N, 3, 1, 1).expand(N, 3, h, w)
        embedding = fc3.view(-1, channels, h, w)
        x = torch.cat((b3, embedding, motion_refinement), dim=1)
        b4 = self.batch4(x)
        x = torch.cat((b2, b4), dim=1)
        b5 = self.batch5(x)
        x = torch.cat((b1, b5), dim=1)
        b6 = self.batch6(x)
        x = torch.cat((img, b6), dim=1)
        b7 = self.batch7(x)
        foreground_translation = self.foreground_translation(b7)
        background_translation = background_translation.reshape(N, 3, 1, 1).expand(N, 3, H, W)
        return intrinsics, rotation, background_translation, foreground_translation