Introduction
=======================


Overview
-----------------

*HyperTrack* is a new hybrid algorithm for deep learned clustering based
on a learned graph constructor called Voxel-Dynamics, Graph Neural Networks
and Transformers. For more details, see the paper and the conference talk.

This repository together with pre-trained torch models downloaded from Hugging
Face can be used to reproduce the paper results on the charged particle
track reconstruction problem.


Hugging Face Quick Start
------------------------------

Install the framework, process TrackML dataset files, download the pre-trained models
from Hugging Face https://huggingface.co/mieskolainen and follow the documentation for inference.


TrackML dataset
------------------------------

Install first the Kaggle API as instructed in https://www.kaggle.com/docs/api and download TrackML challenge data:

.. code-block:: none

	cd .. && mkdir trackml && cd trackml
	kaggle competitions download -c trackml-particle-identification -f train_1.zip
	kaggle competitions download -c trackml-particle-identification -f train_2.zip
	kaggle competitions download -c trackml-particle-identification -f train_3.zip
	kaggle competitions download -c trackml-particle-identification -f train_4.zip
	kaggle competitions download -c trackml-particle-identification -f train_5.zip

	kaggle competitions download -c trackml-particle-identification -f test.zip

	unzip train_1.zip
	unzip train_2.zip
	unzip train_3.zip
	unzip train_4.zip
	unzip train_5.zip

	unzip test.zip

Then execute the following to convert events into pickle files:

.. code-block:: none

	source tests/process_trackml.sh


Folder structure
-----------------------

.. code-block:: none

	- docs              : Documentation
	- data              : Pickle files of input data
	- figs              : Training and prediction figures
	- models            : Trained torch models
	- hypertrack        : Main source code
	- hypertrack/models : Model definitions and hyperparameters
	- src               : Training and inference steering code
	- tests             : Launch and test scripts
