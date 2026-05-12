import torch 
import nibabel as nib 
import numpy as np 
from src .models .swin_unetr import SwinUNETRModel 
import os 

def load_nifti (path ):
    img =nib .load (path )
    return img .get_fdata (),img .affine 

def save_nifti (data ,affine ,path ):
    img =nib .Nifti1Image (data ,affine )
    nib .save (img ,path )

def infer (model ,t1_path ,distorted_path ,output_path ,device ='cuda'):
    model .to (device )
    model .eval ()

    t1_data ,t1_affine =load_nifti (t1_path )
    distorted_data ,distorted_affine =load_nifti (distorted_path )


    t1_data =(t1_data -t1_data .min ())/(t1_data .max ()-t1_data .min ())
    distorted_data =(distorted_data -distorted_data .min ())/(distorted_data .max ()-distorted_data .min ())


    inputs =np .stack ([t1_data ,distorted_data ],axis =0 )
    inputs =torch .tensor (inputs ,dtype =torch .float32 ).unsqueeze (0 ).to (device )

    with torch .no_grad ():
        output =model (inputs ).squeeze (0 ).squeeze (0 ).cpu ().numpy ()

    save_nifti (output ,t1_affine ,output_path )

def main ():
    model_path ='checkpoints/best_model.pth'
    local_data_dir ='data/local'
    output_dir ='results'

    os .makedirs (output_dir ,exist_ok =True )

    device ='cuda'if torch .cuda .is_available ()else 'cpu'
    model =SwinUNETRModel ()
    model .load_state_dict (torch .load (model_path ,map_location =device ))

    for subject in os .listdir (local_data_dir ):
        subject_dir =os .path .join (local_data_dir ,subject )
        if os .path .isdir (subject_dir ):
            t1_path =os .path .join (subject_dir ,'T1.nii.gz')
            distorted_path =os .path .join (subject_dir ,'distorted_b0.nii.gz')
            output_path =os .path .join (output_dir ,f'{subject }_undistorted_b0.nii.gz')

            if os .path .exists (t1_path )and os .path .exists (distorted_path ):
                infer (model ,t1_path ,distorted_path ,output_path ,device )
                print (f'Processed {subject }')

if __name__ =='__main__':
    main ()