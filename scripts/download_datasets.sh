#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Download broden1_224, used by ResNet18 and DenseNet161
if [ ! -f data/datasets/broden/broden1_224/index.csv ]
then

echo "Downloading broden1_224"
mkdir -p data/datasets/broden
pushd data/datasets/broden
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_224.zip \
   -O broden1_224.zip
unzip broden1_224.zip
rm broden1_224.zip
popd

fi

# Download broden1_227, used by AlexNet
if [ ! -f data/datasets/broden/broden1_227/index.csv ]
then

echo "Downloading broden1_227"
mkdir -p data/datasets/broden
pushd data/datasets/broden
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_227.zip \
   -O broden1_227.zip
unzip broden1_227.zip
rm broden1_227.zip
popd

fi

# Download the SNLI dev split, used by the NLI models
if [ ! -f data/datasets/snli_1.0/snli_1.0_dev.txt ]
then

echo "Downloading snli_1.0"
mkdir -p data/datasets
pushd data/datasets
wget --progress=bar \
   https://archive.nyu.edu/bitstream/2451/41728/2/snli_1.0.zip \
   -O snli_1.0.zip
unzip snli_1.0.zip "snli_1.0/snli_1.0_dev.txt"
rm snli_1.0.zip
popd

fi

# Download the preprocessed SNLI dev features and word vectors of Mu and Andreas (2020)
for extension in feats vec
do
if [ ! -f data/datasets/snli_1.0/snli_1.0_dev.${extension} ]
then
wget --progress=bar \
   https://raw.githubusercontent.com/jayelm/compexp/master/nli/data/analysis/snli_1.0_dev.${extension} \
   -O data/datasets/snli_1.0/snli_1.0_dev.${extension}
fi
done
