# Generative AI & LLM - Assignment 2: Diffusion Models

Name: Farwa | Roll number: MSDS25028

## Files
- diffusion.py: DDPM noise schedule, forward process, loss, reverse sampling (implemented)
- unet.py: U-Net forward pass (implemented)
- trainer.py, utils.py, main.py: training, FID and visualization (starter code; main.py has one fix)
- my_checks.py: sanity tests for the implementation
- plots/: results for questions 6.1 to 6.5
- Report.pdf: answers and analysis

## How to run
pip install -r requirements.txt
python main.py   (check the flags in main.py: train_steps, save_and_sample_every, fid, save_folder, data_path)
The experiments were run on Google Colab with a T4 GPU.

## Experiments
- 6.1: train_steps=1000, save_and_sample_every=100, fid=False
- 6.2: train_steps=1000, save_and_sample_every=100, fid=True
- 6.3: train_steps=10000, save_and_sample_every=1000, fid=False
- 6.4 and 6.5: visualize_diffusion with the 10,000-step model

## Results
![loss](plots/6_1_loss_1000_steps.png)
![samples 1000](plots/6_1_samples_1000_steps.png)
![fid](plots/6_2_fid_1000_steps.png)
![samples 10000](plots/6_3_samples_10000_steps.png)
![forward](plots/6_4_forward_diffusion.png)
![backward](plots/6_5_backward_diffusion.png)

## Model checkpoint (10,000 steps)
Google Drive: https://drive.google.com/file/d/1HkQru_ERkKOPZ1imAraigLeP844EgDmh/view?usp=sharing

