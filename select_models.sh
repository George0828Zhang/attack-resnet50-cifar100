#!/usr/bin/env bash
if [ ! -f "kld_result.csv" ]; then
    python divergence.py \
        --csv-path benign_result.csv \
        --model cifar100_vgg11_bn,`
        `cifar100_vgg13_bn,`
        `cifar100_vgg16_bn,`
        `cifar100_vgg19_bn`
        `,cifar100_resnet20`
        `,cifar100_resnet32`
        `,cifar100_resnet44`
        `,cifar100_resnet56`
        `,cifar100_vgg11_bn`
        `,cifar100_vgg13_bn`
        `,cifar100_vgg16_bn`
        `,cifar100_vgg19_bn`
        `,cifar100_mobilenetv2_x0_5`
        `,cifar100_mobilenetv2_x0_75`
        `,cifar100_mobilenetv2_x1_0`
        `,cifar100_mobilenetv2_x1_4`
        `,cifar100_shufflenetv2_x0_5`
        `,cifar100_shufflenetv2_x1_0`
        `,cifar100_shufflenetv2_x1_5`
        `,cifar100_shufflenetv2_x2_0`
        `,cifar100_repvgg_a0`
        `,cifar100_repvgg_a1`
        `,cifar100_repvgg_a2`
        `,nin_cifar100`
        `,resnet110_cifar100`
        `,resnet164bn_cifar100`
        `,resnet272bn_cifar100`
        `,xdensenet40_2_k24_bc_cifar100`
        `,xdensenet40_2_k36_bc_cifar100`
        `,ror3_56_cifar100`
        `,ror3_110_cifar100`
        `,ror3_164_cifar100`
        `,diapreresnet56_cifar100`
        `,diapreresnet110_cifar100`
        `,diapreresnet164bn_cifar100`
        `,preresnet56_cifar100`
        `,preresnet110_cifar100`
        `,preresnet164bn_cifar100`
        `,preresnet272bn_cifar100`
        `,seresnet56_cifar100`
        `,seresnet110_cifar100`
        `,seresnet164bn_cifar100`
        `,seresnet272bn_cifar100`
        `,wrn16_10_cifar100`
        `,wrn28_10_cifar100`
        `,wrn40_8_cifar100`
        `,sepreresnet56_cifar100`
        `,sepreresnet110_cifar100`
        `,sepreresnet164bn_cifar100`
        `,sepreresnet272bn_cifar100`
        `,shakeshakeresnet20_2x16d_cifar100`
        `,shakeshakeresnet26_2x32d_cifar100`
        `,resnext29_32x4d_cifar100`
        `,resnext29_16x64d_cifar100`
        `,resnext272_1x64d_cifar100`
        `,diaresnet56_cifar100`
        `,densenet100_k24_cifar100`
        `,pyramidnet164_a270_bn_cifar100
fi
sort -k2 -n -t"," kld_result.csv | cut -d',' -f1 | head -21