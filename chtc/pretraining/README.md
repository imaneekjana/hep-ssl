# Pretraining deployment

The active workflow is [the augmentation sweep](../../chtc_phase1_steps/README_ZH.md).
Generate a deployment with `python3 chtc_phase1_steps/build_deployment.py` from the project root.
The generated `manage.py` checks preparation before submitting the batch.
The former single-run submit file and wrapper have been removed to avoid competing workflows.
