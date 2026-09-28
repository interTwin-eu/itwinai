from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from itwinai.model_hub.feature import ModelHubFeature, ModelHubModelLoader


def test_pull_strict_mismatch_raises(tmp_path, sanity_check_model_class):
    class WrongShapeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.other_layer = nn.Linear(10, 10)

    ckpt_path = tmp_path / "model.pt"
    torch.save(sanity_check_model_class().state_dict(), ckpt_path)

    with (
        patch("itwinai.model_hub.feature.has_internet_connection", return_value=True),
        patch("itwinai.model_hub.feature.download_file", return_value=ckpt_path),
    ):
        loader = ModelHubModelLoader(
            model_id="x", file_path="root/x/model.pt", model_class=WrongShapeModel
        )
        with pytest.raises(RuntimeError):
            loader()


@pytest.mark.parametrize(
    "mode, has_internet, expect_upload",
    [
        ("online", False, True),
        ("auto", True, True),
        ("auto", False, False),
        ("deferred", True, False),
    ],
)
def test_push_mode_dispatch(tmp_path, mode, has_internet, expect_upload):
    config = {
        "enabled": True,
        "backend": "ai-model-hub",
        "mode": mode,
        "manifest": {"id": "my-model", "name": "My Model"},
    }
    feature = ModelHubFeature(config)
    feature.backend = MagicMock()

    with patch("itwinai.model_hub.feature.has_internet_connection", return_value=has_internet):
        feature.on_training_end(trainer=MagicMock(), ckpt_dir=tmp_path)

    if expect_upload:
        feature.backend.upload.assert_called_once_with(tmp_path)
    else:
        feature.backend.upload.assert_not_called()
