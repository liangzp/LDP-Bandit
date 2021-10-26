# Generalized Linear Bandits with Local Differential Privacy

This repository is the official implementation of [Generalized Linear Bandits with Local Differential Privacy](https://arxiv.org/abs/2106.03365) by Yuxuan Han, Zhipeng Liang, Yang Wang, and Jiheng Zhang.

## Requirements:

Requires python 3, numpy, matplotlib, etc.
Please use the following command to install the dependencies:
```setup
pip install -r requirements.txt
```

## Datasets: 

We evaluate our algorithms in a real data [CRPM-12-001: On-Line Auto Lending dataset](https://www8.gsb.columbia.edu/cprm/research/datasets), provided by the Center for Pricing and Revenue Management at Columbia University.
To obtain a data set, please go to the website and follow their instructions

## Citation:
If you wish to use our repository in your work, please cite our paper:

BibTex:
```
@article{han2021generalized,
  title={Generalized Linear Bandits with Local Differential Privacy},
  author={Han, Yuxuan and Liang, Zhipeng and Wang, Yang and Zhang, Jiheng},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```

Any question about the scripts can be directed to the authors <a href = "mailto: zliangao@connect.ust.hk"> via email</a>.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Generating the figures in the paper:

'./Scheme1/' folder is for step size $1/t$ version of LDP-SGD, which is originally designed for low dimensional setting 

'./Scheme2/' folder is for step size $1/\sqrt{t}$ version of LDP-SGD, which is use for producing figures in Appendix F.

For generating the figures in the paper please execute the following codes:
+ Figure 2
    + Single-param Experiments: 
    ```
    cd Scheme1
    nohup python3 instances/singleparam.py --eps 1  --dest "/results/single-param-eps=1/" > singleparam.out 2>&1 &
    nohup python3 instances/singleparam.py --eps 0.5  --dest "/results/single-param-eps=0.5/" > singleparam.out 2>&1 &
    ```
    + Multi-param Experiments in Figure 2: 
    ```
    cd Scheme1
    nohup python3 instances/multiparam.py --eps 1  --dest "/results/multi-param-eps=1/" > multiparam.out 2>&1 &
    nohup python3 instances/multiparam.py --eps 0.5  --dest "/results/multi-param-eps=0.5/" > multiparam.out 2>&1 &
    ```
+ Figure 3
    + Single-param Experiments: 
    ```
    cd Scheme2
    nohup python3 instances/singleparam.py --dimension 20 --n_actions 20 --eps 0.5 --dest '/results/single-param-d=20-k=20-eps=0.5/' > singleparam.out 2>&1 &
    nohup python3 instances/singleparam.py --dimension 20 --n_actions 20 --eps 1 --dest '/results/single-param-d=20-k=20-eps=1/' > singleparam.out 2>&1 &
    ```
    + Multi-param Experiments:
    ```
    cd Scheme2
    nohup python3 instances/multiparam.py --dimension 10 --n_actions 10 --eps 0.5 --dest '/results/multiparam-d=10-k=10-eps=0.5/' > multiparam.out 2>&1 &
    nohup python3 instances/multiparam.py --dimension 10 --n_actions 10 --eps 1 --dest '/results/multiparam-d=10-k=10-eps=1/' > multiparam.out 2>&1 &
    ```
+ Real-data Experiments Figure 4: 
```
cd Scheme1
nohup python3 instances/crpm.py --eps 1  --dest "/results/crpm-eps=1/" > crpm.out 2>&1 &
nohup python3 instances/crpm.py --eps 0.5  --dest "/results/crpm-eps=0.5/" > crpm.out 2>&1 &
```
