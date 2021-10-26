nohup python3 instances/singleparam.py --dimension 20 --n_actions 20 --eps 0.5 --dest '/results/single-param-d=20-k=20-eps=0.5/' > singleparam.out 2>&1 &
nohup python3 instances/singleparam.py --dimension 20 --n_actions 20 --eps 1 --dest '/results/single-param-d=20-k=20-eps=1/' > singleparam.out 2>&1 &

nohup python3 instances/multiparam.py --dimension 10 --n_actions 10 --eps 0.5 --dest '/results/multiparam-d=10-k=10-eps=0.5/' > multiparam.out 2>&1 &
nohup python3 instances/multiparam.py --dimension 10 --n_actions 10 --eps 1 --dest '/results/multiparam-d=10-k=10-eps=1/' > multiparam.out 2>&1 &
    