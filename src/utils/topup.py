import os 
import subprocess 

def run_topup (synthesized_b0_path ,distorted_b0_path ,output_dir ):
    """
    Run FSL TOPUP for final distortion correction.
    Assumes FSL is installed and available in PATH.
    """

    acq_params ='0 -1 0 0.05\n0 1 0 0.05'

    with open (os .path .join (output_dir ,'acqparams.txt'),'w')as f :
        f .write (acq_params )


    merged_path =os .path .join (output_dir ,'merged_b0.nii.gz')
    subprocess .run (['fslmerge','-t',merged_path ,synthesized_b0_path ,distorted_b0_path ])


    topup_base =os .path .join (output_dir ,'topup')
    subprocess .run (['topup','--imain='+merged_path ,'--datain='+os .path .join (output_dir ,'acqparams.txt'),
    '--config=b02b0.cnf','--out='+topup_base ,'--iout='+topup_base +'_corrected'])


    corrected_path =os .path .join (output_dir ,'final_corrected.nii.gz')
    subprocess .run (['applytopup','--imain='+distorted_b0_path ,'--inindex=2','--datain='+os .path .join (output_dir ,'acqparams.txt'),
    '--topup='+topup_base ,'--out='+corrected_path ,'--method=jac'])

    return corrected_path 


