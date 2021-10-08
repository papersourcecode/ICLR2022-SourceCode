nohup python main.py --uid student \
--teacher-model teacher-model.pth.tar \
--epochs 1000 \
--lr 1e-2 \
--seed 1 \
>> student.out 2>&1 &
