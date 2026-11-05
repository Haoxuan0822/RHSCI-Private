# Experiments

Evaluation orchestration and exploratory tools call the model and simulator
interfaces maintained elsewhere in this repository.

- [SIMPLE evaluation worker](pi05_between/README.md): model serving, simulator launch and multi-round results.
- [Evaluation queue](simple_queue/README.md): task configuration, asset preparation and GPU scheduling.

Data, weights, environments and outputs are stored outside the checkout.
Configure filesystem paths for your machine before running these scripts.
See [Evaluation](../docs/EVALUATION.md) for the versioned protocol.
