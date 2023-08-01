# Global Momentum Initialization

This repository offers TensorFlow code to reproduce results from the paper:

[Boosting the Transferability of Adversarial Attacks with Global Momentum
Initialization](https://arxiv.org/abs/2211.11236)


## Requirements

- Python >= 3.6.5
- Tensorflow >= 1.12.0
- Numpy >= 1.15.4
- opencv >= 3.4.2
- scipy > 1.1.0
- pandas >= 1.0.1
- imageio >= 2.6.1


## Quick Start

- **Prepare models**

  Download pretrained TensorFlow models(https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw). Then put these models and the data into `./models/` and `./dev_data`, respectively.

- **Attack with GI-FGSM**

  Using `gi_mi_fgsm.py` or `mi_fgsm.py` to implement GI_MI-FGSM and MI-FGSM,  you can run this attack as following
  
  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python gi_mi_fgsm.py.py --output_dir outputs
  ```
  where `gpuid` can be set to any free GPU ID in your machine. And adversarial examples will be generated in directory `./outputs`.
  
- **Evaluations on models**

  Taking `gi_mi_fgsm.py` for example, you can comment out the attack phase code and implement the inference using the inference code, the corresponding code is already marked in the comments.

  ```bash
  CUDA_VISIBLE_DEVICES=gpuid python gi_mi_fgsm.py.py --output_dir outputs
  ```

- **Evaluations on defenses**

    You can refer to the relevant code [here](https://github.com/JHL-HUST/VT), where we use an aligned experimental setup.
