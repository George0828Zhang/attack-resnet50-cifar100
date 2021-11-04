# python attack.py --attack linbp \
#     --iters 30 \
#     --model cifar100_resnet56,cifar100_resnet44,cifar100_resnet32,cifar100_resnet20 \
#     --linbp-layer 2_8,2_5,2_1,1_3 \
#     --sgm-lambda 0.5 \
#     --savedir adv_images/

python attack.py --attack tifgsm \
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
    --iters 30 \
    --diversity-prob 0 \
    --savedir ensemble10_tifgsm


python divergence.py \
    --csv-path ensemble10_v2_result.csv \
    --model nin_cifar100

    

# python divergence.py \
#     --model cifar100_vgg11_bn,`
#     `cifar100_vgg13_bn,`
#     `cifar100_vgg16_bn,`
#     `cifar100_vgg19_bn`
#     `,cifar100_resnet20`
#     `,cifar100_resnet32`
#     `,cifar100_resnet44`
#     `,cifar100_resnet56`
#     `,cifar100_vgg11_bn`
#     `,cifar100_vgg13_bn`
#     `,cifar100_vgg16_bn`
#     `,cifar100_vgg19_bn`
#     `,cifar100_mobilenetv2_x0_5`
#     `,cifar100_mobilenetv2_x0_75`
#     `,cifar100_mobilenetv2_x1_0`
#     `,cifar100_mobilenetv2_x1_4`
#     `,cifar100_shufflenetv2_x0_5`
#     `,cifar100_shufflenetv2_x1_0`
#     `,cifar100_shufflenetv2_x1_5`
#     `,cifar100_shufflenetv2_x2_0`
#     `,cifar100_repvgg_a0`
#     `,cifar100_repvgg_a1`
#     `,cifar100_repvgg_a2`
#     `,nin_cifar100`
#     `,diapreresnet56_cifar100`
#     `,diaresnet56_cifar100`
#     `,shakeshakeresnet26_2x32d_cifar100`
#     `,rir_cifar100`
#     `,ror3_56_cifar100`
#     `,wrn20_10_32bit_cifar100`
#     `,wrn28_10_cifar100`
#     `,xdensenet40_2_k24_bc_cifar100`
#     `,densenet100_k24_cifar100`
#     `,pyramidnet164_a270_bn_cifar100`
#     `,sepreresnet56_cifar100`
#     `,seresnet56_cifar100`
#     `,resnext29_32x4d_cifar100`
#     `,preresnet56_cifar100`
#     `,resnet164bn_cifar100


