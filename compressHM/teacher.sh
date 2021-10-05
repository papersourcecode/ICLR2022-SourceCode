
epoch=1000
uid=teacher

nohup python main.py --uid $uid \
--epochs $epoch \
--lr 1e-2 \
--mode scratch \
--seed $seed \
>> ${uid}.out 2>&1 &



