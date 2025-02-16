# Instructions for Evaluating Accuracy of Depth Predictions

We assume you have followed instructions for generating depth predictions from [DAv2](../dav2/README.md) or [DINOv2](../dino/README.md), and you have activated the [`terra-torch3d`](../../README.md) Conda environment.

- To compute quantitative accuracy of depth predictions
    - Run [`python eval_depth_accuracy.py`](eval_depth_accuracy.py)
- To generate delayed reprojections of videos using depth predictions
    - Run [`python save_delay_proj_renders.py`](save_delay_proj_renders.py)
- To compute quantitative accuracy of delayed reprojections
    - Run [`python eval_delay_proj_psnr.py`](eval_delay_proj_psnr.py)
- To save colormapped visualizations of depth predictions
    - Run [`python visualize_depth.py`](visualize_depth.py)
