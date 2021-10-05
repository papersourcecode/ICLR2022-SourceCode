
task=mnist
tarch=1
sarch=2
mode=our
beta=1e-3
MCs=1
tseed=2 
tuid=teacher
uid=student-${seed}

nohup python main.py --uid $uid \
--gpu-wait 0.8 \
--task $task \
--mode $mode \
--beta $beta \
--distill-z-kl-lambda 1e-3 \
--distill-out-kl-lambda 0.8 \
--temperature 2 \
--seed $seed \
--s-arch $sarch \
--t-arch $tarch \
--teacher-model ./CKPT/${tuid}-model.pth.tar \
--train-MCsamples $MCs \
> ${uid}.out 2>&1 &
