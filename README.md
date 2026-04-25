# Generalized Linear Bandits with Local Differential Privacy

This repository is the official implementation of [Generalized Linear Bandits with Local Differential Privacy](https://arxiv.org/abs/2106.03365) by Yuxuan Han, Zhipeng Liang, Yang Wang, and Jiheng Zhang.

## Requirements:

Requires python 3, numpy, matplotlib, etc.
Please use the following command to install the dependencies:
```setup
pip install -r requirements.txt
```

If you use `uv`, install/sync the project dependencies with:
```setup
uv sync
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

### Figures 2--5 (Discussion and Ablation Experiments)

Self-contained Python scripts in this folder. Each can be run in two modes:

```bash
uv run <script>.py          # display figures interactively
uv run <script>.py --save   # save as PNG files
```

| Script | Figure(s) | Description |
| ------ | --------- | ----------- |
| `reproduce_frequent_update.py` | Figure 2 | Algorithm 1 (sparse update) vs Algorithm 5 (frequent update) under different privacy levels (epsilon = 0.1, 0.5, 1, +inf) |
| `reproduce_adv_context.py` | Figure 3 | Elliptical estimation error of the LDP OLS estimator under adversarial vs stochastic context sequences |
| `reproduce_diversity.py` | Figures 4 & 5 | Epoch greedy vs greedy vs greedy first, without strong diversity (Figure 4) and with strong diversity (Figure 5) |

---

### Figures 6--10 (Main Synthetic and Real-Data Experiments)


'./Scheme1/' folder is for step size $1/t$ version of LDP-SGD, which is originally designed for low dimensional setting 

'./Scheme2/' folder is for step size $1/\sqrt{t}$ version of LDP-SGD, which is use for producing figures in Appendix F.

#### Figures 6 & 7 — Single-param Synthetic Experiments
Estimation error and cumulative regret comparisons of LDP-SGD, LDP-OLS, LDP-UCB and LDP-GLOC in the single-parameter setting.

For generating the figures in the paper please execute the following codes:

```
cd Scheme1
uv run instances/singleparam.py --eps 1  --dest "/results/single-param-eps=1/" > singleparam.out 2>&1 &
uv run instances/singleparam.py --eps 0.5  --dest "/results/single-param-eps=0.5/" > singleparam.out 2>&1 &
```

#### Figures 8 — Single-param Synthetic Experiments (Generalized Linear Reward Case)
Cumulative regret comparisons of LDP-SGD, and LDP-GLOC in the single-parameter generalized linear reward setting.

```
cd Scheme1
uv run instances/logistic.py --eps 1  --dest "/results/logistic-eps=1/" > logistic.out 2>&1 &
uv run instances/logistic.py --eps 0.5  --dest "/results/logistic-eps=0.5/" > logistic.out 2>&1 &

uv run instances/poisson.py --eps 1  --dest "/results/poisson-eps=1/" > poisson.out 2>&1 & 
uv run instances/poisson.py --eps 0.5  --dest "/results/poisson-eps=0.5/" > poisson.out 2>&1 &
```

#### Figure 9 — Multi-param Synthetic Experiments
```
cd Scheme1
uv run instances/multiparam.py --eps 1  --dest "/results/multi-param-eps=1/" > multiparam.out 2>&1 &
uv run instances/multiparam.py --eps 0.5  --dest "/results/multi-param-eps=0.5/" > multiparam.out 2>&1 &
```
#### Figure 10 — Real-Data Experiments (Online Auto Lending)

Cumulative regret comparisons of LDP-SGD and LDP-GLOC on the CRPM-12-001 dataset.
```
cd Scheme1
uv run instances/crpm.py --eps 1  --dest "/results/crpm-eps=1/" > crpm.out 2>&1 &
uv run instances/crpm.py --eps 0.5  --dest "/results/crpm-eps=0.5/" > crpm.out 2>&1 &
```

To launch all Scheme1 runs listed above, execute:
```run
cd Scheme1
sh run.sh
```
