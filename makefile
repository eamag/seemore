.PHONY: experiment1 experiment2

experiment1:
	@uv run main.py --n_layer=10
experiment2:
	@uv run main.py
