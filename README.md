# TM-NAS
Training-free Multi-Scale Neural Architecture Search for High-incidence Cancer Prediction

## Citation
```
@INPROCEEDINGS{FPSO,
  title={A Flexible Variable-length Particle Swarm Optimization Approach to Convolutional Neural Network Architecture Design},
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie},
  booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
  year={2021},
  pages={934-941},
  doi={10.1109/CEC45853.2021.9504716}
}

@ARTICLE{EPCNAS,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  journal={IEEE Transactions on Evolutionary Computation (Early Access)},
  title={Particle Swarm Optimization for Compact Neural Architecture Search for Image Classification},
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TEVC.2022.3217290}
}

@ARTICLE{10132401,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Split-Level Evolutionary Neural Architecture Search With Elite Weight Inheritance},
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2023.3269816}
}

@inproceedings{lee2024az,
  title={AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search},
  author={Lee, Junghyup and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5893--5903},
  year={2024}
}

@article{wang2024mednas,
  title={MedNAS: Multi-Scale Training-Free Neural Architecture Search for Medical Image Analysis},
  author={Wang, Yan and Zhen, Liangli and Zhang, Jianwei and Li, Miqing and Zhang, Lei and Wang, Zizhou and Feng, Yangqin and Xue, Yu and Wang, Xiao and Chen, Zheng and others},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2024},
  publisher={IEEE}
}

@article{li2023zico,
  title={Zico: Zero-shot nas via inverse coefficient of variation on gradients},
  author={Li, Guihong and Yang, Yuedong and Bhardwaj, Kartikeya and Marculescu, Radu},
  journal={arXiv preprint arXiv:2301.11300},
  year={2023}
}
```

## Requirements

- `python 3.9`
- `Pytorch >= 1.8`
- `torchvison`
- `opencv-python`

## Data

Download MedmnistV2, NAS-Bench-201, LC25000, BreakHis and Colorectal(CRC-5000) datasets, and place them in `global.ini` file.

- MedmnistV2 from [here](https://medmnist.com/)
- NAS-Bench-201 from [here](https://github.com/D-X-Y/NAS-Bench-201)
- LC25000 dataset from [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- BreakHis dataset from [here](https://www.kaggle.com/datasets/ambarish/breakhis)
- Colorectal dataset from [here](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)
    
    
## Folder Structure
- datasets : store dataset
- load_dataset: load data for train eval and test
- log : store train eval and test log
- populations : store population err flops gbest params pbest population information
- scripts : According to the data set corresponding to the template folder, the corresponding network script is generated for training evaluation
- template: Network files constructed from different datasets
- trained_models : stored model
- evaluate : Architecture evaluation
- evolve : Weights and particles correspond to the evolution of architecture parameters
- global : Configuration files
- main ：Program main function
- population : Population structure generation

# Using ETM-NAS for Custom Dataset Search and Training

If you want to use this repository to perform search and training on your own dataset, please follow these steps:

## Steps

1. **Store Your Data**  
   Place your data in the `datasets` directory.

2. **Modify Data Loading**  
   Modify the data loading method in the `load_dataset` function to ensure that your data is correctly loaded.

3. **Set Up Configuration File**  
   Configure the settings in the `global.ini` file before running the search and training:

   ```ini
   [PSO]
   pop_size = 20
   num_iteration = 20
   particle_length = 24
   weight = 0.7298
   c1 = 1.49618
   c2 = 1.49618
   start_init_gen = 1

   [NETWORK]
   min_epoch_eval = 6
   dataset = Medmnist
   name = BreastMNIST
   max_strided = 2
   image_channel = 1
   max_output_channel = 100
   dense = 1

   [SEARCH]
   agent_name = MY
   repeat = 32
   epoch_test = 450
   batch_size = 16
   weight_decay = 5e-4
   sigma = 0.03
   b1 = 1
   b2 = 2
   w_min = 0.4
   w_max = 0.9
   c_max = 2.01
   c_min = 0.8
   Tp = 2e6
   Tf = 200e6
   wp = -0.01, -1
   wf = -0.01, -1
   wa = -0.01, -1

   [DATASETS]
   root_dir = E:\\Zero_NAS_RES\\Code\\TM-NAS\\datasets
   ```

4. **Run the Program.**  
   Execute the main.py file.

# Enjoy the efficient and enjoyable training experience with TM-NAS!

# Notice
If you would like to receive full source results of TM-NAS, please contact the author Jie Zheng at zjpdd0905@stu.cwnu.edu.cn. Please indicate your purpose of use, I will reply the first moment I see the email, thank you!
