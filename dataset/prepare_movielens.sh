#!/bin/sh
echo "Making Directory"
mkdir -p movie_lens

if [ ! -f /movie_lens/ml-20m.zip ]; then
    echo "Downloading Movielens"
    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip -P movie_lens
fi

cd movie_lens

unzip ml-20m.zip
rm -f ml-20m.zip
