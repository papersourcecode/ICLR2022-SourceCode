uid=teacher

nohup python main.py --uid $uid \
--gpu-wait 0.8 \
--task mnist \
--mode VIB \
--beta 1e-3 \
--seed 2 \
--s-arch 1 \
--train-MCsamples 1 \
> ${uid}.out 2>&1 &


