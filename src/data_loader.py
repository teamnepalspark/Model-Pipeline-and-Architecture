import os 
import torch 
import torch .nn as nn 
from monai .networks .nets import SwinUNETR 
from monai .data import DataLoader ,Dataset 
from monai .transforms import (
Compose ,LoadImaged ,EnsureChannelFirstd ,Spacingd ,Orientationd ,
ScaleIntensityRanged ,EnsureTyped ,RandRotated ,RandFlipd ,RandGaussianNoised 
)
import nibabel as nib 
import numpy as np 
from sklearn .model_selection import KFold 

class DWIDataset (Dataset ):
    def __init__ (self ,data_list ,transform =None ):
        self .data_list =data_list 
        self .transform =transform 

    def __len__ (self ):
        return len (self .data_list )

    def __getitem__ (self ,idx ):
        data =self .data_list [idx ]
        if self .transform :
            data =self .transform (data )
        return data 

def get_data_loader (data_dir ,batch_size =4 ,train =True ):
    data_list =[]
    for subject in os .listdir (data_dir ):
        subject_dir =os .path .join (data_dir ,subject )
        if os .path .isdir (subject_dir ):
            t1_path =os .path .join (subject_dir ,'T1.nii.gz')
            distorted_path =os .path .join (subject_dir ,'distorted_b0.nii.gz')
            undistorted_path =os .path .join (subject_dir ,'undistorted_b0.nii.gz')
            if all (os .path .exists (p )for p in [t1_path ,distorted_path ,undistorted_path ]):
                data_list .append ({
                'T1':t1_path ,
                'distorted':distorted_path ,
                'undistorted':undistorted_path 
                })

    transforms =Compose ([
    LoadImaged (keys =['T1','distorted','undistorted']),
    EnsureChannelFirstd (keys =['T1','distorted','undistorted']),
    Spacingd (keys =['T1','distorted','undistorted'],pixdim =(1.0 ,1.0 ,1.0 ),mode ='bilinear'),
    Orientationd (keys =['T1','distorted','undistorted'],axcodes ='RAS'),
    ScaleIntensityRanged (keys ='T1',a_min =0 ,a_max =2000 ,b_min =0 ,b_max =1 ),
    ScaleIntensityRanged (keys =['distorted','undistorted'],a_min =0 ,a_max =1000 ,b_min =0 ,b_max =1 ),
    EnsureTyped (keys =['T1','distorted','undistorted']),
    ])

    if train :
        transforms =Compose ([
        transforms ,
        RandRotated (keys =['T1','distorted','undistorted'],range_x =0.1 ,range_y =0.1 ,range_z =0.1 ,prob =0.5 ),
        RandFlipd (keys =['T1','distorted','undistorted'],spatial_axis =0 ,prob =0.5 ),
        RandGaussianNoised (keys =['T1','distorted','undistorted'],mean =0.0 ,std =0.1 ,prob =0.2 ),
        ])

    dataset =DWIDataset (data_list ,transform =transforms )
    loader =DataLoader (dataset ,batch_size =batch_size ,shuffle =train ,num_workers =4 )
    return loader 