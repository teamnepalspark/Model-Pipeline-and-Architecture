import torch 
from src .models .swin_unetr import SwinUNETRModel 
from src .data_loader import get_data_loader 
import os 

def main ():
    data_dir ='data/HCP'
    device ='cuda'if torch .cuda .is_available ()else 'cpu'



    train_loader =get_data_loader (data_dir ,batch_size =2 ,train =True )
    val_loader =get_data_loader (data_dir ,batch_size =2 ,train =False )

    model =SwinUNETRModel ()
    train_model (model ,train_loader ,val_loader ,num_epochs =100 ,lr =1e-4 ,device =device )

if __name__ =='__main__':
    main ()