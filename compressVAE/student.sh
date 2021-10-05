seed=1
task=svhn
beta=1e-1
search=6
sdarch=4
tseed=1
mode=our
dzlambda=1e-3
uid=student

nohup python main.py --uid $uid \
--gpu-wait 0.5 \
--calculate-fid-with inceptionv3 \
--task $task \
--mode $mode \
--beta $beta \
--warmup-epoch 100 \
--epochs 200 \
--batch-size 256 \
--distill-z-kl-lambda $dzlambda \
--seed $seed \
--s-e-arch $search \
--s-d-arch $sdarch \
--t-e-arch $search \
--t-d-arch $search \
--teacher-model ./CKPT/teacher-model.pth.tar \
> ${uid}.out 2>&1 &

