import torch 
import torch .nn as nn 
from monai .networks .nets import SwinUNETR 
from monai .losses import DiceLoss 
from monai .metrics import DiceMetric 
from torch .optim import Adam 
from torch .utils .data import DataLoader 
import numpy as np 
from tqdm import tqdm 
import os 

class SwinUNETRModel (nn .Module ):
    def __init__ (self ,img_size =(96 ,96 ,96 ),in_channels =2 ,out_channels =1 ,feature_size =48 ):
        super ().__init__ ()
        self .model =SwinUNETR (
        img_size =img_size ,
        in_channels =in_channels ,
        out_channels =out_channels ,
        feature_size =feature_size ,
        use_checkpoint =True ,
        )

    def forward (self ,x ):
        return self .model (x )

def train_model (model ,train_loader ,val_loader ,num_epochs =100 ,lr =1e-4 ,device ='cuda'):
    model .to (device )
    optimizer =Adam (model .parameters (),lr =lr ,betas =(0.9 ,0.999 ),weight_decay =1e-5 )
    criterion =nn .MSELoss ()

    best_loss =float ('inf')
    for epoch in range (num_epochs ):
        model .train ()
        train_loss =0 
        for batch in tqdm (train_loader ,desc =f'Epoch {epoch +1 }/{num_epochs }'):
            t1 =batch ['T1'].to (device )
            distorted =batch ['distorted'].to (device )
            undistorted =batch ['undistorted'].to (device )


            inputs =torch .cat ([t1 ,distorted ],dim =1 )
            targets =undistorted 

            optimizer .zero_grad ()
            outputs =model (inputs )
            loss =criterion (outputs ,targets )
            loss .backward ()
            optimizer .step ()

            train_loss +=loss .item ()

        train_loss /=len (train_loader )


        model .eval ()
        val_loss =0 
        with torch .no_grad ():
            for batch in val_loader :
                t1 =batch ['T1'].to (device )
                distorted =batch ['distorted'].to (device )
                undistorted =batch ['undistorted'].to (device )

                inputs =torch .cat ([t1 ,distorted ],dim =1 )
                targets =undistorted 

                outputs =model (inputs )
                loss =criterion (outputs ,targets )
                val_loss +=loss .item ()

        val_loss /=len (val_loader )

        print (f'Epoch {epoch +1 }, Train Loss: {train_loss :.4f}, Val Loss: {val_loss :.4f}')

        if val_loss <best_loss :
            best_loss =val_loss 
            torch .save (model .state_dict (),'checkpoints/best_model.pth')

def cross_validate (data_dir ,k =5 ,**kwargs ):

    subjects =[s for s in os .listdir (data_dir )if os .path .isdir (os .path .join (data_dir ,s ))]
    kf =KFold (n_splits =k ,shuffle =True ,random_state =42 )

    fold_results =[]
    for fold ,(train_idx ,val_idx )in enumerate (kf .split (subjects )):
        print (f'Fold {fold +1 }/{k }')


        train_subjects =[subjects [i ]for i in train_idx ]
        val_subjects =[subjects [i ]for i in val_idx ]



        train_loader =get_data_loader (data_dir ,train =True )
        val_loader =get_data_loader (data_dir ,train =False )

        model =SwinUNETRModel ()
        train_model (model ,train_loader ,val_loader ,**kwargs )

        fold_results .append (...)

    return fold_results 