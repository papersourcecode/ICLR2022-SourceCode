nohup python main.py --uid continual \
--lr 1e-4 \
--gpu-wait 0 \
--epochs 50 \
--warmup-epoch 25 \
--target-class 15 \
--continual-step 0 \
--distill-z-kl-lambda 1e-1 \
--calculate-fid-with inceptionv3 \
--seed 1 \
>> continual.out 2>&1 &


