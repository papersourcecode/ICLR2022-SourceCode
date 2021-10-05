seed=1
task=svhn
search=6
sdarch=6
beta=1e-1
uid=teacher

nohup python main.py --uid ${uid} \
--gpu-wait 0.5 \
--calculate-fid-with inceptionv3 \
--task $task \
--beta $beta \
--warmup-epoch 100 \
--epochs 200 \
--batch-size 256 \
--seed $seed \
--s-e-arch $search \
--s-d-arch $sdarch \
> ${uid}.out 2>&1 &
