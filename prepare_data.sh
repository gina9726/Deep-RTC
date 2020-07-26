#!/bin/bash

if [[ "$1" =~ ^(awa2|cifar100|imagenet|inaturalist)$ ]]; then
    echo $1
    for split in train valid test
    do
            subsample="prepro/splits/$1/${split}.json"
            data="prepro/raw/$1"

            python prepro/prepro.py --data ${data} --out $1 --subsample ${subsample}

    done
else
    echo "usage: \$ ./prepare_data.sh {dataset}"
    echo "    dataset: awa2/cifar100/imagenet/inaturalist"
fi


