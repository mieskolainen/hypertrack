Inference
=======================


The inference for the track reconstruction problem is executed with:

.. code-block:: none

    python src/trackml_inference.py \
        --param tune-5 \
        --cluster "transformer" \
        --epoch -1 \
        --event_start 8750 --event_end 9999 \
        --read_tag f-0p1-hyper-5 \
        --out_tag  f-0p1-hyper-5 \
        --rfactor 0.1 --noise_ratio 0.05 \
        --node2node hyper --ncell 262144 \
        --fp_dtype "float32"

See more examples under ``/tests/inference*`` for different pile-up scenarios.

Main parameters
---------------------

The inference steering program parameters can be seen with the command ``--help``. The main parameters are:

.. code-block:: none

    --edge_threshold 0.55
    --cluster "transformer" (cut, dbscan, hdbscan)

The ``edge_threshold`` parameter defines the clustering efficiency/purity/latency tradeoff.
Small values retain more graph edges for the clustering stage after the GNN and load the transformer, but increase latency.

Similarly with ``--cluster`` algorithm, the transformer should provide always the highest
efficiency and purity, but at the cost of latency. The clustering algorithm parameters
are under model hyperparameters ``/hypertrack/models/global_<TAG>.py``, such as the pivot search
for the transformer. More exhaustive search can improve results but at the cost of increased
computing time.
