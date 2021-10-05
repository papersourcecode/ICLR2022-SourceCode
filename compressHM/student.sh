

epoch=1000
uid=student

nohup python main.py --uid $uid \
--epochs $epoch \
--lr 1e-2 \
--mode our \
--seed 1 \
>> ${uid}.out 2>&1 &
