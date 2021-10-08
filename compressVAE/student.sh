nohup python main.py --uid student \
--gpu-wait 0.5 \
--calculate-fid-with inceptionv3 \
--task celeba \
--beta 1e-1 \
--warmup-epoch 25 \
--epochs 50 \
--batch-size 128 \
--distill-z-kl-lambda 1e-3 \
--seed 1 \
--s-arch 4 \
--t-arch 6 \
--teacher-model ./CKPT/teacher-model.pth.tar \
> student.out 2>&1 &

