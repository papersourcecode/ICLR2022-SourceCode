nohup python main.py --uid student \
--teacher-model teacher-model.pth.tar \
--epochs 200 \
--lr 1e-4 \
--seed 1 \
>> student.out 2>&1 &


