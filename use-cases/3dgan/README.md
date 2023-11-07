# 3DGAN use case

To visualize the logs with MLFLow run the following in the terminal:

```bash
micromamba run -p ../../.venv-pytorch mlflow ui --backend-store-uri ml_logs/mlflow_logs
```

And select the "3DGAN" experiment.

## Inference

The following is preliminary and not 100% ML/scientifically sound.

1. As inference dataset we can reuse training/validation dataset
2. As model, we can create a dummy version of it with:

    ```python
    import torch
    from model import ThreeDGAN
    # Same params as in the training config file!
    my_gan = ThreeDGAN()
    torch.save(my_gan, '3dgan-inference.pth')
    ```

3. Run inference with the following command:

    ```bash
    TODO
    ```
