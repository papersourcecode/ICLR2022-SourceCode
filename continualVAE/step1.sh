###########################################
# change tclass to change attribute
tclass=15
###########################################


task=celeba
mode=our
dzlambda=1e-1
alpha=1e0
lr=1e-4
uid=continual

nohup python main.py --uid ${uid} \
--alpha $alpha \
--lr $lr \
--gpu-wait 0 \
--epochs 50 \
--target-class $tclass \
--continual-step 0 \
--distill-z-kl-lambda $dzlambda \
--calculate-fid-with inceptionv3 \
--task $task \
--mode $mode \
--seed 1 \
>> ${uid}.out 2>&1 &


