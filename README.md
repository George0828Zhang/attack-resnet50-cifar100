# Security and Privacy of Machine Learning - Homeworks 
## 1-1 Black-box attack
Implements black-box attack for homework 1-1.
### How to run
1. install dependencies
```bash
pip install -r requirements.txt
```
2. download data
```bash
pip install gdown
gdown --id 1WpkI4K0OXbgVUJwqNcxTW2vlF2UHA7Ob
unzip cifar-100_eval.zip
```
3. usage
```
usage: attack.py [-h] [--datadir DATADIR] [--num-workers NUM_WORKERS]
                 [--batch-size BATCH_SIZE] [--cpu] [--model MODEL]
                 [--victim VICTIM] [--attack {none,fgsm,ifgsm,linbp,tisgbp}]
                 [--atk-cfg ATK_CFG] [--save-adv SAVE_ADV]

optional arguments:
  -h, --help            show this help message and exit
  --datadir DATADIR
  --num-workers NUM_WORKERS
  --batch-size BATCH_SIZE
  --cpu
  --model MODEL
  --victim VICTIM
  --attack {none,fgsm,ifgsm,linbp,tisgbp}
  --atk-cfg ATK_CFG
  --save-adv SAVE_ADV
```
4. use resnet56 as source and pyramidnet as victim, with fgsm attack
```bash
python attack.py --attack fgsm \
    --victim pyramidnet110_a48_cifar100
```
5. use resnet56 as source and densenet (default) as victim, then use SGM+LinBP attack with
- 30-iteration
- linear bp after 8th block in 2nd group
- then save the images in adv_images/
```bash
python attack.py --attack linbp \
    --iters 30 \
    --linbp-layer 2_8 \
    --sgm-lambda 0.5 \
    --savedir adv_images/
```

### Use other models
Choose any of the Cifar100 models in these directories:
#### pytorchcv
Choose any in [link](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py) with cifar100 in its name.
#### pytorch_cifar_models
1. Choose the models in [link](https://github.com/chenyaofo/pytorch-cifar-models#cifar-100) under CIFAR-100. 
2. Add `cifar100_` to the beginning of the name, e.g. `cifar100_mobilenetv2_x1_4`

## 1-2 Grey-box attack
Implements grey-box attack for homework 1-2.
### Procedure
1. Submit benign images to the server and obtain the prediction and logits (probabilities) in `benign_result.csv`.
2. Compute benign accuracy.
```bash
python score.py benign_result.csv
# 2021-11-06 01:39:45 | INFO | root | top-1: 0.834
```
3. Search the available pretrained models for models $Q$ who minimizes its Kullback-Leibler divergence from the target model $P$, i.e. $\min D(P\|Q)$.
```bash
bash select_models.sh
# see attack.sh for selection.
```
4. Use Translation Invariant attack (TIFGSM) with ensemble of the 20 models selected above:
```bash
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
```
5. Compute accuracy.
```bash
python score.py tifgsm_ensemble15.csv
# 2021-11-08 04:55:30 | INFO | root | top-1: 0.134
```
6. See `attack.sh` for other options presented in report.

## References
Code references: 
- [https://github.com/qizhangli/linbp-attack](https://github.com/qizhangli/linbp-attack)
- [https://github.com/Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [https://github.com/chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
- [https://colab.research.google.com/drive/1Hb7HasWTKKmIGBVGEdfSEfJKC3O-a2FS?usp=sharing](https://colab.research.google.com/drive/1Hb7HasWTKKmIGBVGEdfSEfJKC3O-a2FS?usp=sharing)
- [https://colab.research.google.com/github/George0828Zhang/seq2seq-nmt/blob/main/HW05.ipynb](https://colab.research.google.com/github/George0828Zhang/seq2seq-nmt/blob/main/HW05.ipynb)