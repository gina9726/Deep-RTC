#!/bin/bash
batch_size=128

if [[ "$1" =~ ^(awa2|cifar100|imagenet|inaturalist)$ ]]; then
    if [[ "$1" == "cifar100" ]]; then
        python train_cifar.py --config configs/${1}.yml
        python test_cifar.py --checkpoint runs/${1}/deep-rtc --b ${batch_size}
    else
        python train.py --config configs/${1}.yml
        python test.py --checkpoint runs/${1}/deep-rtc --b ${batch_size}
    fi
else
    echo "usage: \$ ./run.sh {dataset}"
    echo "    dataset: awa2/cifar100/imagenet/inaturalist"
fi


