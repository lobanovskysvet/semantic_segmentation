#!/bin/bash
env_name=lobanova_test
echo y|conda update conda
echo y|conda create -n $env_name python=3.7
source activate $env_name
pip install -r requirements.txt
echo completed install
