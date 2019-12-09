# robustness-self-supervision
An official implementation of [Improving Model Robustness via Automatically Incorporating Self-supervision Tasks](http://metalearning.ml/2019/papers/metalearn2019-kim.pdf)

# Prerequisites
* Python 3.7+
* Pytorch 1.3.0+

# Dataset
This code requires CIFAR-10-C dataset. You can download the dataset [here](https://zenodo.org/record/2535967).

# Usage

Training:
```
python train.py -s {save_point} --rot --lambda_fix_epoch 5 --batch_size 64
```

Test with CIFAR-10-C:
```
python test_cifarc.py -l {save_point}
```

Use `-h` option to check more flags.

# Acknowledgement
This code is based on [Dan Hendrycks' code](https://github.com/hendrycks/pre-training/tree/master/robustness/adversarial).

# Author
Donghwa Kim ([@storykim](https://github.com/storykim))
