- [x] Configuring hyperparameters through CLI (e.g. using a yaml/json)
- [x] A simple solution (can be any free library or service) for hyperparameter management and tracking
- [x] Storing and visualizing the training loss
- [x] A simple solution for profiling the training performance to identify bottlenecks in the model configuration
- [x] Include as many best practices into the training that you know of to ensure the fastest performance possible (i.e. half precision, ...)
- [ ] Extension of this training function in order to be scaleable to a multi-GPU or multi-node setting.


Run with `uv run main.py`

See metrics at `uv run mlflow ui`

Mixed precision out of the box with lightning (uncomment in main, easier than pass to args), batch size tuning. See more at https://lightning.ai/docs/pytorch/stable/levels/intermediate_level_11.html 