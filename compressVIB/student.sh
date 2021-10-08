for seed in 1 2 3 4 5
do
    nohup python main.py --uid student-${seed} \
    --gpu-wait 0.8 \
    --task svhn \
    --beta 1e-3 \
    --distill-z-kl-lambda 1e-5 \
    --distill-out-kl-lambda 1.0 \
    --temperature 2 \
    --seed $seed \
    --s-arch 2 \
    --t-arch 1 \
    --teacher-model ./CKPT/teacher-model.pth.tar \
    > student-${seed}.out 2>&1 &
done


