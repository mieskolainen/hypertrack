Training
=======================


Train Voxel-Dynamics graph constructor
----------------------------------------

The voxelized space(-time) Voronoi division & connectivity ``C-matrix`` estimator takes
into account the dynamics of particle tracks (equations of motion) as described by the simulations (training data),
where do they appear in space-time and also the detector geometry adaptively.

For example:

.. code-block:: none

	python src/trackml_pretrain_voxdyn.py --node2node hyper --ncell 65536 131072 --device cpu

Once trained, the estimator's true (false) edge efficiency is pile-up invariant (!) by construction, but the purity is not.

The Voronoi division quality saturates quickly with the number of training sample size (number of tracks ~ hits),
but the connectivity C-matrix is estimated with much higher accuracy with larger samples.


Train neural GNN + Transformer model
----------------------------------------

The training first proceeds only with the GNN edge predictor part of the network and once
the AUC is over ``--auc_threshold``, then the neural clustering transformer training is activated end-to-end.

The training for the track reconstruction problem is executed with:

.. code-block:: none

	python src/trackml_train.py \
		--param tune-5 \
		--cluster "transformer" \
		--soft_reset 0 \
		--learning_rate 5e-4 \
		--scheduler_type "warm-cos" \
		--optimizer "AdamW" \
		--epoch -1 \
		--save_tag f-0p1-hyper-5 \
		--rfactor_start 0.1 --rfactor_end 0.1 --noise_ratio 0.05 \
		--node2node hyper --ncell 262144 \
		--validate 0 \
		--batch_size 1 \
		--fp_dtype "float32"

See more examples under ``/tests/train*`` for different pile-up scenarios.

Training will take days (weeks) on a single Nvidia V100 GPU, and VRAM memory requirements grow at
least linearly as a function of the track density if ``--ncell`` is scaled up accordingly.

A GPU with 32+ GB VRAM is recommended (a must), to be able to increase the pile-up scenario
with ``--rfactor_start`` and ``--rfactor_end``, use deeper models and possibly
higher ``--batch_size`` parameter in the neural training.


Training parameters
------------------------------

The training steering program parameters, such as the starting learning rate, can be seen with the command ``--help``
and the rest are descibed under ``/hypertrack/models/global_<TAG>.py``.


Model hyperparameters
-------------------------------

The model hyperparameters are encoded in the file ``/hypertrack/models/global_<TAG>.py``.
One can create several different tunes or custom models, by choosing a ``--save_tag <TAG>`` name and
by copying existing models files into a new pair:

.. code-block:: none

	/hypertrack/models/global_<TAG>.py
	/hypertrack/models/models_<TAG>.py
	
before starting the training.


Ground truth graph topology
-------------------------------------------------------------

The ground truth adjacency of graph nodes (hits) is controlled with the command line parameter ``--node2node``.
This needs to be consistent between ``trackml_pretrain_voxdyn``, ``trackml_train`` and ``trackml_inference``,
but one can do mixed diagnostics.

.. code-block:: none

	`hyper`   for a fully connected hyperedge like 'lasso' over all nodes
	`eom`     for a minimal spanning tree (equations of motions helix trajectory)
	`cricket` for double hops included eom trajectory

This impacts directly the Voxel-Dynamics estimator ``C-matrix`` construction and indirectly
the neural estimator training, especially the edge predictor part, because it defines its
label target goal. The ``eom`` will favor to train an edge predictor which
is *space-time local* and the ``hyper`` mode a fully *space-time non-local* one.

In general, only ``hyper`` mode is fully compatible with the overall neural
clustering goal, but one can use ``eom`` mode, e.g., to train only a highly performing
GNN edge predictor. Set also ``--cluster none``, which inactivates the clustering transformer from training.

The ground truth definition impacts also edge ROC/AUC (efficiency) between ``A_hat`` (estimate) and ``A`` (ground truth) adjacency.
However, the final track clustering metrics such as ``efficiency`` or ``purity`` definition are topology independent,
but naturally the chosen definition impacts the underlying model construction.


Progressive transfer learning
---------------------------------

To reduce overall GPU time and improve convergence, it is beneficial to train the model using a low pile-up
scenario and once converged, use that model as a starting point for the higher pile-up scenario.
This can be done progressively in multiple steps, e.g. by setting ``--rfactor_start`` and
``--rfactor_end`` first to ``0.01``, then both ``0.1`` and finally both to ``0.2`` (assuming you have enough VRAM).

Similarly, one should increase the voxel space dimension ``ncell`` as a function of the mean pile-up.

Starting the training using an existing model weights can be done simply by
using ``--transfer_tag`` and ``--save_tag`` command line parameters when starting the new training.

Remember to remove ``--transfer_tag`` afterwards. 


Learning rate
----------------------
Start with

.. code-block:: none

	--learning_rate 5e-4

The learning rate can be crucial, e.g. with 1e-5 the GNN does not really enter learning phase.

Then in the final stage, perhaps decrease to

.. code-block:: none

	--learning_rate 1e-4 (e.g. Transformer more stable)

In general, larger models with more GNN and transformer layers and higher latent dimensions may require smaller learning rates.

