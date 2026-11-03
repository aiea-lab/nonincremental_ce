#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Downloading resnet18, alexnet, and densenet161 trained on Places365"
mkdir -p data/model/zoo
mkdir -p data/model/other
pushd data/model/zoo
wget --progress=bar \
   http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
wget --progress=bar \
    http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar
wget --progress=bar \
    http://places2.csail.mit.edu/models_places365/whole_densenet161_places365_python36.pth.tar
popd

echo "Downloading the BiLSTM, BERT, and LLaMA models trained on SNLI"
pushd data/model/other
if [ ! -f bowman.pth ]
then
wget --progress=bar \
    http://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth \
    -O bowman.pth
fi
for model in bert llama
do
if [ ! -f ${model}_dense.pth ]
then
wget --progress=bar \
    https://huggingface.co/ccenli/dense_weights/resolve/main/${model}_dense.pth
fi
done
popd

# The BERT and LLaMA checkpoints need the vocabulary stored in the BiLSTM one
python3 scripts/convert_nli_checkpoints.py
echo "done"
