

seed=1
mode=our
resume=0
epoch=5
std=1
lr=1e-4
arch=1
uid=student

nohup python main.py --uid $uid \
--resume $resume \
--epochs $epoch \
--lr $lr \
--STD $std \
--mode $mode \
--seed $seed \
>> ${uid}.out 2>&1 &


