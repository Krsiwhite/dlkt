# Local original baseline (2026-10-13)

This snapshot preserves the local TextAKT3PL model and training source as received, including working-tree changes relative to upstream 26c43e0. API keys have been removed; preprocessing reads OPENAI_API_KEY from the environment. The original training script is historical: it uses the test set during checkpoint selection. Use the experimental runner on the experiment branch for independent validation. Existing semantic vectors are not regenerated or trained. The retained 80.6453% ensemble is a separate local artifact and is not this model.

Training data already tracked by the existing upstream repository is unchanged. New runs, model weights and local credentials are excluded from commits. The original local folders are not modified.
