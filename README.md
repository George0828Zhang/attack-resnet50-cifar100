# Homework 1-1 Black-box attack
Implements black-box attack for homework 1.1.
## How to run
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
5. use resnet56,44,32,20 as source and densenet (default) as victim, then use SGM+LinBP attack with
- ensemble attack
- 30-iteration
- linear bp after each specified layer/block
- then save the images in adv_images/
```bash
python attack.py --attack linbp \
    --iters 30 \
    --model cifar100_resnet56,cifar100_resnet44,cifar100_resnet32,cifar100_resnet20 \
    --linbp-layer 2_8,2_5,2_1,1_3 \
    --sgm-lambda 0.5 \
    --savedir adv_images/
```

## Use other models
Choose any of the Cifar100 models in these directories:
### pytorchcv
Choose any in [link](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py) with cifar100 in its name.
### pytorch_cifar_models
1. Choose the models in [link](https://github.com/chenyaofo/pytorch-cifar-models#cifar-100) under CIFAR-100. 
2. Add `cifar100_` to the beginning of the name, e.g. `cifar100_mobilenetv2_x1_4`


## References
Code references: 
- [https://github.com/qizhangli/linbp-attack](https://github.com/qizhangli/linbp-attack)
- [https://github.com/Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [https://github.com/chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
- [https://colab.research.google.com/drive/1Hb7HasWTKKmIGBVGEdfSEfJKC3O-a2FS?usp=sharing](https://colab.research.google.com/drive/1Hb7HasWTKKmIGBVGEdfSEfJKC3O-a2FS?usp=sharing)
- [https://colab.research.google.com/github/George0828Zhang/seq2seq-nmt/blob/main/HW05.ipynb](https://colab.research.google.com/github/George0828Zhang/seq2seq-nmt/blob/main/HW05.ipynb)