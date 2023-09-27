Installation
=======================

The installation goes as follows.

.. contents::
    :local:


Required: Conda installation
----------------------------------
.. code-block:: none

	wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh 

Then execute the installer with ``bash filename.sh``


Install environment
----------------------------------

Remark: To avoid ``No space left on device`` problem with conda or pip, set the temporary path first
to point to a path where there is enough space, for example:

.. code-block:: none
	
	mkdir $HOME/tmp
	export TMPDIR=$HOME/tmp

Then execute:

.. code-block:: none

	git clone git@github.com:mieskolainen/hypertrack.git && cd hypertrack

	# Create the environment
	conda env create -f environment.yml
	conda activate hypertrack

	# Install pip packages
	pip install -r requirements.txt && pip install -r requirements-aux.txt


Initialize environment
----------------------------------

Always start with

.. code-block:: none

	conda activate hypertrack
	source setenv.sh


GPU-support commands
---------------------

Show the graphics card status

.. code-block:: none
	
	nvidia-smi

Show CUDA-compiler tools status

.. code-block:: none
	
	nvcc --version

Show Pytorch GPU support in Python

.. code-block:: none
	
	import torch
	torch.cuda.is_available()
	print(torch.cuda.get_device_name(0))


Conda virtual environment commands
-----------------------------------
.. code-block:: none

	conda activate hypertrack

	...[install dependencies with pip, do your work]...
	
	conda deactivate

	conda info --envs
	conda list --name hypertrack
	
	# Remove environment completely
	conda env remove --name hypertrack

C-library versions
-----------------------------------

.. code-block:: none

	ldd --version
