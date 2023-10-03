# *hypertrack*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/mieskolainen/hypertrack/actions/workflows/hypertrack-install-test.yml/badge.svg)](https://github.com/mieskolainen/hypertrack/actions)

https://arxiv.org/abs/2309.14113 <br>

## *HyperTrack*: Neural Combinatorics for High Energy Physics
Presented in CHEP 2023<br>
https://indico.jlab.org/event/459/contributions/11748
<br>
<br>
Mikael Mieskolainen<br>
m.mieskolainen@imperial.ac.uk <br>
<br>

## Overview

_HyperTrack_ is a new hybrid algorithm for deep learned clustering based on a learned graph constructor called Voxel-Dynamics, Graph Neural Networks and Transformers. For more details, see the paper and the conference talk.
</br>

This repository together with pre-trained torch models downloaded from Hugging Face can be used to reproduce the paper results on the charged particle track reconstruction problem.
</br>

The technical API and instructions at:

https://mieskolainen.github.io/hypertrack

</br>


## Hugging Face Quick Start

Install the framework, process TrackML dataset files, download the pre-trained models from Hugging Face https://huggingface.co/mieskolainen and follow the documentation for inference.

</br>


## Citation

If you use this in your work or find ideas interesting, please cite:

```
@Conference{citekey,
  author    = "Mikael Mieskolainen",
  title     = "HyperTrack: Neural Combinatorics for High Energy Physics",
  booktitle = "CHEP 2023, 26th International Conference on Computing in High Energy & Nuclear Physics",
  year      = "2023"
}
```