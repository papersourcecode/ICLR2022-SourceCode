



seed=1
mode=scratch
resume=0
epoch=6
std=1
lr=1e-4
uid=teacher

nohup python main.py --uid $uid \
--resume $resume \
--epochs $epoch \
--lr $lr \
--STD $std \
--mode $mode \
--seed $seed \
>> ${uid}.out 2>&1 &
