EXP=$1

function compress () {
    zip -rq $(basename $1).zip $1
}

if [[ $EXP == "1" ]]; then
    echo "LinBP single."
    res=linbp_single
    python attack.py --attack linbp \
        --iters 30 \
        --model cifar100_resnet56 \
        --linbp-layer 2_8 \
        --sgm-lambda 0.5 \
        --savedir $res
elif [[ $EXP == "2" ]]; then
    echo "LinBP ensemble."
    res=linbp_ensemble
    python attack.py --attack linbp \
        --iters 30 \
        --model cifar100_resnet56,cifar100_resnet44,cifar100_resnet32,cifar100_resnet20 \
        --linbp-layer 2_8,2_5,2_1,1_3 \
        --sgm-lambda 0.5 \
        --savedir $res
elif [[ $EXP == "3" ]]; then
    echo "IFGSM ensemble (10)."
    res=ifgsm_ensemble10
    python attack.py --attack ifgsm \
        --iters 30 \
        --model nin_cifar100`
                `,xdensenet40_2_k24_bc_cifar100`
                `,ror3_56_cifar100`
                `,diaresnet56_cifar100`
                `,diapreresnet56_cifar100`
                `,sepreresnet56_cifar100`
                `,preresnet56_cifar100`
                `,seresnet56_cifar100`
                `,shakeshakeresnet26_2x32d_cifar100`
                `,cifar100_shufflenetv2_x0_5 \
        --savedir $res
elif [[ $EXP == "4" ]]; then
    echo "TIFGSM ensemble (10)."
    res=tifgsm_ensemble10
    python attack.py --attack tifgsm \
        --iters 30 \
        --diversity-prob 0 \
        --model nin_cifar100`
                `,xdensenet40_2_k24_bc_cifar100`
                `,ror3_56_cifar100`
                `,diaresnet56_cifar100`
                `,diapreresnet56_cifar100`
                `,sepreresnet56_cifar100`
                `,preresnet56_cifar100`
                `,seresnet56_cifar100`
                `,shakeshakeresnet26_2x32d_cifar100`
                `,cifar100_shufflenetv2_x0_5 \
        --savedir $res
elif [[ $EXP == "5" ]]; then
    echo "TIFGSM ensemble (15)."
    res=tifgsm_ensemble15
    python attack.py --attack tifgsm \
        --iters 30 \
        --diversity-prob 0 \
        --model nin_cifar100`
                `,shakeshakeresnet20_2x16d_cifar100`
                `,xdensenet40_2_k24_bc_cifar100`
                `,resnet110_cifar100`
                `,diaresnet56_cifar100`
                `,ror3_56_cifar100`
                `,diapreresnet56_cifar100`
                `,wrn40_8_cifar100`
                `,preresnet56_cifar100`
                `,xdensenet40_2_k36_bc_cifar100`
                `,seresnet56_cifar100`
                `,wrn28_10_cifar100`
                `,diapreresnet110_cifar100`
                `,ror3_110_cifar100`
                `,preresnet110_cifar100 \
        --savedir $res
elif [[ $EXP == "6" ]]; then
    echo "TIFGSM ensemble (20)."
    res=tifgsm_ensemble20
    python attack.py --attack tifgsm \
        --iters 30 \
        --diversity-prob 0 \
        --model nin_cifar100`
                `,shakeshakeresnet20_2x16d_cifar100`
                `,xdensenet40_2_k24_bc_cifar100`
                `,resnet110_cifar100`
                `,diaresnet56_cifar100`
                `,ror3_56_cifar100`
                `,diapreresnet56_cifar100`
                `,wrn40_8_cifar100`
                `,preresnet56_cifar100`
                `,xdensenet40_2_k36_bc_cifar100`
                `,seresnet56_cifar100`
                `,wrn28_10_cifar100`
                `,diapreresnet110_cifar100`
                `,ror3_110_cifar100`
                `,preresnet110_cifar100`
                `,sepreresnet56_cifar100`
                `,seresnet110_cifar100`
                `,sepreresnet110_cifar100`
                `,sepreresnet164bn_cifar100`
                `,shakeshakeresnet26_2x32d_cifar100 \
        --savedir $res
fi
compress $res