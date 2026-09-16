from unittest.mock import patch

import torch

from itwinai.model_hub.feature import ModelHubModelLoader
from itwinai.torch.inference import TorchPredictor


def test_execute_loader_called_once(
    tmp_path,
    sanity_check_model_class,
    synthetic_inference_dataset_class,
):
    ckpt_path = tmp_path / "model.pt"
    torch.save(sanity_check_model_class().state_dict(), ckpt_path)

    with (
        patch("itwinai.model_hub.feature.has_internet_connection", return_value=True),
        patch(
            "itwinai.model_hub.feature.download_file", return_value=ckpt_path
        ) as mock_download,
    ):
        loader = ModelHubModelLoader(
            model_id="x", file_path="root/x/model.pt", model_class=sanity_check_model_class
        )
        predictor = TorchPredictor(config={}, model=loader, strategy="ddp")

        with patch.object(predictor.strategy, "clean_up"):
            predictor.execute(inference_dataset=synthetic_inference_dataset_class())

    assert isinstance(predictor.model, sanity_check_model_class)
    mock_download.assert_called_once()
