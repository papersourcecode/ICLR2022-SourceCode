nohup python main.py --uid teacher \
--gpu-wait 0.8 \
--task svhn \
--beta 1e-3 \
--seed 16 \
--s-arch 1 \
> teacher.out 2>&1 &


