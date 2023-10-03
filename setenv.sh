#!/bin/sh
#
# Set library paths

ln -f -s $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc ~/.local/bin/gcc
ln -f -s $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ ~/.local/bin/g++

export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
