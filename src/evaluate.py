import torch 
import numpy as np 
from src .models .swin_unetr import SwinUNETRModel 
from src .data_loader import get_data_loader 
from sklearn .metrics import mean_squared_error 
import os 

def evaluate_model (model ,data_loader ,device ='cuda'):
    model .to (device )
    model .eval ()

    mse_scores =[]
    with torch .no_grad ():
        for batch in data_loader :
            t1 =batch ['T1'].to (device )
            distorted =batch ['distorted'].to (device )
            undistorted =batch ['undistorted'].to (device )

            inputs =torch .cat ([t1 ,distorted ],dim =1 )
            targets =undistorted 

            outputs =model (inputs )


            mse =mean_squared_error (targets .cpu ().numpy ().flatten (),outputs .cpu ().numpy ().flatten ())
            mse_scores .append (mse )

    avg_mse =np .mean (mse_scores )
    print (f'Average MSE: {avg_mse :.4f}')
    return avg_mse 

def main ():
    model_path ='checkpoints/best_model.pth'
    data_dir ='data/HCP'

    device ='cuda'if torch .cuda .is_available ()else 'cpu'
    model =SwinUNETRModel ()
    model .load_state_dict (torch .load (model_path ,map_location =device ))


    test_loader =get_data_loader (data_dir ,batch_size =2 ,train =False )

    evaluate_model (model ,test_loader ,device )

if __name__ =='__main__':
    main ()