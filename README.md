# Optimized Gradient Clipping for Noisy Label Learning

Code for paper *Optimized Gradient Clipping for Noisy Label Learning*

## Requirements

```bash
python >= 3.10
pytorch >= 2.3
scikit-learn >= 1.4.0
tensorboard
```

## How to use

### Modifiy the following path

* CIFAR-10 and CIFAR-10PDN

```bash
--train_dataset_params \
   root /path/to/dataset/cifar10 \
```

```bash
--eval_dataset_params \
    root /path/to/dataset/cifar10 \
```

* CIFAR-10PDN

```bash
--train_dataset_params \
    root /path/to/dataset/cifar10 \
    noise_file_path /path/to/cifar10n/CIFAR-10_human.pt \
```

```bash
--eval_dataset_params \
    root /path/to/dataset/cifar10 \
```

### Run the script

```bash
bash ./script/cifar10/agc_ce.sh
```
