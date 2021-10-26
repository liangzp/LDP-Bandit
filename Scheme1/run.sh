nohup python3 instances/singleparam.py --eps 1  --dest "/results/multi-param-eps=1/" > singleparam.out 2>&1 &
nohup python3 instances/singleparam.py --eps 0.5  --dest "/results/multi-param-eps=0.5/" > singleparam.out 2>&1 &

nohup python3 instances/multiparam.py --eps 1  --dest "/results/multi-param-eps=1/" > multiparam.out 2>&1 &
nohup python3 instances/multiparam.py --eps 0.5  --dest "/results/multi-param-eps=0.5/" > multiparam.out 2>&1 &

nohup python3 instances/crpm.py --eps 1  --dest "/results/crpm-eps=1/" > crpm.out 2>&1 &
nohup python3 instances/crpm.py --eps 0.5  --dest "/results/crpm-eps=0.5/" > crpm.out 2>&1 &
