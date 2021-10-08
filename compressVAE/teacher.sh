nohup python main.py --uid teacher \
--gpu-wait 0.5 \
--calculate-fid-with inceptionv3 \
--task celeba \
--beta 1e-1 \
--warmup-epoch 25 \
--epochs 50 \
--batch-size 128 \
--seed 1 \
--s-arch 6 \
> teacher.out 2>&1 &
