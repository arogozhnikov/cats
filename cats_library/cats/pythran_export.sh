#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $DIR
pythran -march=native -D USE_BOOST_SIMD -fopenmp fm_python_utils.py
popd