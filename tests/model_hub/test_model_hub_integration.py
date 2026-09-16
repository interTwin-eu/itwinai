import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Dataset

from itwinai.model_hub.feature import ModelHubModelLoader
from itwinai.torch.inference import TorchPredictor
from itwinai.torch.trainer import TorchTrainer


class SyntheticTrainDataset(Dataset):
    def __init__(self, n=16, h=16, w=16):
        self.n, self.h, self.w = n, h, w

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(2, self.h, self.w), torch.randn(1, self.h, self.w)


def _fake_hub_response(hub_root, model_id, subpath=""):
    target = hub_root / model_id / subpath if subpath else hub_root / model_id
    entries = []
    for item in target.iterdir():
        entries.append(
            {
                "type": "directory" if item.is_dir() else "file",
                "name": item.name,
            }
        )
    return entries


def test_push_then_pull(tmp_path, sanity_check_model_class, synthetic_inference_dataset_class):
    # Step 1: train and push
    model = sanity_check_model_class()
    ckpt_root = tmp_path / "checkpoints"

    trainer = TorchTrainer(
        model=model,
        config={"optimizer": "sgd", "loss": "mse"},
        epochs=1,
        strategy=None,
        checkpoint_every=1,
        checkpoints_location=ckpt_root,
    )
    with patch.object(trainer.strategy, "clean_up", new=MagicMock()):
        trainer.execute(SyntheticTrainDataset(), SyntheticTrainDataset())

    best_ckpt_dir = ckpt_root / "best_model"
    assert (best_ckpt_dir / "model.pt").exists()

    original_state_dict = torch.load(best_ckpt_dir / "model.pt", weights_only=True)

    hub_root = tmp_path / "fake_hub"
    model_id = "test-model"
    hub_model_dir = hub_root / model_id
    (hub_model_dir / "root" / "best_model").mkdir(parents=True)
    (hub_model_dir / "root" / "best_model" / "model.pt").write_bytes(
        (best_ckpt_dir / "model.pt").read_bytes()
    )
    (hub_model_dir / "p").mkdir(parents=True)

    shutil.rmtree(Path("tmp") / "modelhub_downloads" / model_id, ignore_errors=True)

    def fake_list_files(base_url, m_id, subpath=""):
        return _fake_hub_response(hub_root, m_id, subpath)

    def fake_download_file(base_url, m_id, file_path, dst_dir):
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = hub_model_dir / file_path
        dst = dst_dir / src.name
        dst.write_bytes(src.read_bytes())
        return dst

    # Step 2: pull model back and run inference
    with (
        patch("itwinai.model_hub.feature.has_internet_connection", return_value=True),
        patch("itwinai.model_hub.download.list_files", side_effect=fake_list_files),
        patch("itwinai.model_hub.feature.download_file", side_effect=fake_download_file),
    ):
        loader = ModelHubModelLoader(model_id=model_id, model_class=sanity_check_model_class)
        predictor = TorchPredictor(config={}, model=loader, strategy="ddp")

        with patch.object(predictor.strategy, "clean_up", new=MagicMock()):
            predictions = predictor.execute(
                inference_dataset=synthetic_inference_dataset_class()
            )

    pulled_state_dict = predictor.model.state_dict()
    for key in original_state_dict:
        assert torch.allclose(original_state_dict[key], pulled_state_dict[key])

    assert len(predictions) == 4
