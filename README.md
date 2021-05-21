# PDA
This is an implemention for our SIGIR 2021 paper "Causal Intervention for Leveraging Popularity Bias in Recommendation" based on tensorflow.


## Requirements
+ tensorflow == 1.14
+ Cython (for neurec evaluator)
+ Numpy
+ prefetch-generator
+ python3

## Datasets

+ Kwai: we provide the URL of original data and the pre-processing codes (not checked) for filtering and splitting it. We do not provide the processed Kwai dataset here because we don't make sure whether we have the right to open it. If you have difficulties getting it or process it, you can connect us.  
  
  [original data](https://www.kuaishou.com/activity/uimc);  [processing-code](/data/kwai) 
  
+ Douban(Movie): we provide the URL of original data. Please note that only the movies dataset is utilized. We also provide the preprocessing codes (not checked, please check it with the processed data) and the processed dataset. If you use this dataset,  you may need to cite the following paper as the dataset owners request:`` Song, Weiping, et al. "Session-based social recommendation via dynamic graph attention networks." WSDM 2019.``  
  [original data](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data); [processing-code](/data/douban); [processed data](/data/douban)

+ Tencent: this is a private dataset.

## Paramters
Key parameters in train_new_api.py:
+ --pop_exp: gamma or $\tilde{gamma}$ in paper.
+ --train: model selection (normal:BPRMF/BPRMF-A | s_condition:PD/PDA | temp_pop:BPR(t)-pop).
+ --test: similar to train.
+ -- saveID: saved name flag.
+ others: others: read help, or "python xxx.py --help"

## Commands 
We provide following commands for our models and baselines.
### PD & PDA
We provide two methods.
#### 1. Simply Reproduce the Results:
+ First, we have provided the model that we trained for simply reproducing the results in our paper. Then you can run the following commands to reproduce the results in our paper.
  ```
  python -u MF/simple_reproduce.py --dataset douban --epoch 2000 --save_flag 0 --log_interval 5 --start 0 --end 10 --step 1 --batch_size 2048 --lr 1e-2 --train s_condition --test s_condtion --saveID xxx --cuda 0 --regs 1e-2 --valid_set valid --pop_exp 0.22 --save_dir /home/PDA/save_model/ --Ks [20,50]
  ```
  And you need to change the 'pop_exp' for different datasets. (kwai:0.16, douban:0.22)
+ The trained model can be download at this [URL](http:).
 
#### 2. Start from Scratch:
If you want to run PD and PDA on new datasets, you need:

##### **1). Split data**: 

+ Split the dataset into T stages by yourself and save each stage data as a file with the name like "t_0.txt". The files should have the following format in each line:
```` user interacted_item1 interacted_item2 ... ````.
+ And you also need to save all training/testing/valid data in one file with a name such as "train_with_time.txt", and it should have the following format:
```` uid iid time stars ````, where time in [0,T-1]. 
  
  Note that we have provided the processed data for Douban, you can refer to it.
  

##### **2). Compute Popularity**: 
+ Compute the item popularity of each stage and normalize the computed popularity: 
  ```
  python pop_pre.py --path your_data_path --slot_count T
  ```
  the computed popularity will be saved in a file, so you only need to run this code once.

##### **3). Run PD/PDA**: 
+ Run the main command: 
  ````
  nohup python -u MF/train_new_api.py --dataset kwai --epoch 2000 --save_flag 1 --log_interval 5 --start 0 --end 10 --step 1 --batch_size 2048 --lr 1e-2 --train s_condition --test s_condition --saveID s_condition --cuda 1 --regs 1e-2 --valid_set valid --pop_exp gamma > output.out &
  ````
  and tune the parameters.

##### **4). Optimal Parameters for Douban and Kwai**:
+ the prameters that we found:

  | datasets\para     | PD reg    | PD pop_exp(gamma)     | PDA reg | PDA pop_exp(gama) |
  | ---------- | :-----------:  | :-----------: | :-----------: | :-----------: |
  | kwai     | 1e-3   | 0.02  | 1e-3 | 0.16 |
  | Douban | 1e-3 | 0.02 | 1e-3 | 0.22 |
  
  Other parameters: default. 
  
  Note: For PD and PDA, they can get good performance compared with baselines with the same gamma, such as for Kwai with gamma=0.1, both PD and PDA can get a good performance compared with baselines. Due to the influence of random seeds and different machines, the results may have a few differences. To reproduce the same results easily, we provide the trained models. 
  
 
 ### Baselines

We provide the codes for BPRMF/BPR-PC/BPR(t)-pop implemented by ourselves. 
 
 ##### 1. BPRMF/BPRMF-A:
 + run BPRMF/BPRMF-A:
   ````
   nohup python -u MF/train_new_api.py --dataset kwai --epoch 2000 --save_flag 0 --log_interval 5 --start 0 --end 10 --step 1 --batch_size 2048 --lr lr --train normal --test normal --saveID normal --cuda 1 --regs reg --valid_set valid > output.out &
   ````
   When run BPRMF, it will run BPRMF-A synchronously.
 ##### 2. BPR-PC:
+ you need two steps:
  ````
  step1: get the trained model of BPRMF.
  step2: run BPR_PC.py and find hyper-parameters(alpha and beta): 
       python3 -W ignore  -u MF/BPR_PC.py --dataset dataset --epoch 2000 --save_flag 0 --log_interval 5 --start 0 --end 10 --step 1 --batch_size 2048 --lr 1e-3 --train normal --test normal  --saveID normal --pretrain 0 --cuda 1 --regs 1e-2  --valid_set valid --Ks [20,50] --pc_alpha $alpha --pc_beta $beta
  ````
  The model path is determined by parameters such as --train/test/saveID. For more detail, please read the code. 

 ##### 3.  BPR(t)-Pop:

+ Similar to BPR-MF, change the parameters "train/test/saveID" to "temp_pop".

 ##### 4. xQuad and DICE
+ The code for xQuad and DICE is provided by the original authors. You can also ask the origin authors for the code.Please note that for DICE, regarding a regularization term  named $L_{discrepancy}$, we replace $dCor$ with another option --- $L2$. Because our datasets are far bigger than the datasets taken by the original paper, computing $dCor$ is very slow and will be out of memory for the 2080Ti GPU that we used. As suggested in the paper, for the large-scale dataset, $L_2$ is taken as a substitute to be its $L_{discrepancy}$.

 ##### 5. hyper-parameters:
 + Please read our paper about tuning the hyper-parameters.

## Acknowledgments
+ The c++ evaluator implemented by [Neurec](https://github.com/wubinzzu/NeuRec) is used in some case.

+ Some codes are provided by the co-author [Tianxin Wei](https://github.com/weitianxin).

## Citation
If you use the codes in your research, please cite our paper.
