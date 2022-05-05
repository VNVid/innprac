#!/usr/bin/env bash

echo Unit Tests
pytest unittests.py

echo
echo Testing models
python3 models_testing.py
echo End of testing
echo

git clone https://github.com/cardiffnlp/tweeteval
git clone https://github.com/eusip/SILICONE-benchmark.git
python3 conf_matr.py