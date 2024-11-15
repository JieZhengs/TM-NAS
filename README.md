# ETM-NAS
An Efficient Train-free NAS for Medical Image Classification

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
   root_dir = E:\\Zero_NAS_RES\\Code\\ETM-NAS\\datasets
   ```

4. **Run the Program.**  
   Execute the main.py file.

# Enjoy the efficient and enjoyable training experience with ETM-NAS!

# If you would like to receive full source results of ETM-NAS, please contact first author Jie Zheng at zjpdd0905@stu.cwnu.edu.cn. Please indicate your purpose of use, I will reply the first moment I see the email, thank you
