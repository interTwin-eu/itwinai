Accessing models from the RI-SCALE Model Hub
============================================

**Author(s)**: Rakesh Sarma (FZJ)

Once a ML model has been trained, it is often needed to be shared with collaborators, or to
publish it in open repositories. itwinai integrates with the `RI-SCALE Model Hub
<https://modelhub.riscale.eu>`_ to support users with this functionality which allows pushing
a trained checkpoint to the Model Hub, and pulling a checkpoint to run inference on it.

Pushing a model
---------------

Pushing is handled automatically by :class:`~itwinai.torch.trainer.TorchTrainer` whenever
Model Hub support is enabled in its configuration. On every checkpoint save
(:meth:`~itwinai.torch.trainer.TorchTrainer.save_checkpoint`), the checkpoint directory --
containing ``model.pt`` (the model's raw ``state_dict``), ``state.pt`` (optimizer/scheduler/
epoch state), and ``config.yaml`` -- is handed to
:class:`~itwinai.torch.model_hub.feature.ModelHubFeature`, which:

1. Writes a ``manifest.yaml`` into the checkpoint directory via
   :func:`~itwinai.torch.model_hub.manifest.write_manifest`, merging user-supplied fields
   with sensible defaults. At minimum, ``id`` and ``name`` must be provided.
2. Uploads the checkpoint directory using the configured backend at the end of all epochs.
   Backends implement :class:`~itwinai.torch.model_hub.backends.base.BaseBackend` and are
   selected by name via :func:`~itwinai.torch.model_hub.backends.get_backend`. Currently the
   only backend is :class:`~itwinai.torch.model_hub.backends.itwinai_hub.AIModelHubBackend`.
   The abstraction is to enable future backends (e.g. HuggingFace).

The timing of the upload is controlled by a ``mode`` setting:

- ``online``: upload immediately, regardless of connectivity.
- ``auto``: upload if internet is available; otherwise print the checkpoint's local
  location and skip the upload.
- ``deferred``: never upload automatically; the checkpoint is left ready to be pushed
  manually later.

.. admonition:: Example Model Hub push configuration

    .. code-block:: yaml

        model_hub:
          enabled: true
          backend: ai-model-hub
          mode: online
          manifest:
            id: checkpoint-example
            name: My Model
            published: true

    The final `published: true` ensures that the pushed model is readily visible to all users
    on the AI Model Hub.

Pulling a model
---------------

Pulling is handled by :class:`~itwinai.torch.inference.ModelHubModelLoader`, an
implementation of :class:`~itwinai.serialization.ModelLoader`. Like any other
``ModelLoader``, it can be used wherever a model loader is expected -- most commonly as the
``model`` argument of :class:`~itwinai.torch.inference.TorchPredictor`.

Unlike pushing, the Model Hub's file API has no endpoint to download a whole model folder
at once: files are retrieved one at a time, by exact path
(``GET /artifacts/{model_id}/files/{file_path}``). To spare users from needing to know that
exact path, ``ModelHubModelLoader`` supports two modes:

  - If ``file_path`` is provided explicitly, that file is downloaded directly.
  - If ``file_path`` is omitted, itwinai lists the model's files
    (:func:`~itwinai.torch.model_hub.download.list_files`) and locates
    ``root/<checkpoint_dir_name>/model.pt`` automatically
    (:func:`~itwinai.torch.model_hub.download.discover_weights_file`), matching the layout
    produced by :meth:`~itwinai.torch.trainer.TorchTrainer.save_checkpoint`.
    ``discover_weights_file`` only looks for a top-level ``root/`` entry; it does not
    inspect or otherwise handle any other top-level entries the Hub may contain.

.. admonition:: Example Model Hub pull configuration

    .. code-block:: yaml

        predictor:
          _target_: itwinai.torch.inference.TorchPredictor
          config: {}
          model:
            _target_: itwinai.torch.inference.ModelHubModelLoader
            model_id: checkpoint-example
            model_class: my_module.MyModel

.. important::
   Model Hub checkpoints store a raw ``state_dict`` -- just tensors, with no architecture
   information -- following the same convention used by
   :meth:`~itwinai.torch.trainer.TorchTrainer.save_checkpoint`. This means ``model_class``
   is **always required** when pulling: it must be the exact :class:`~torch.nn.Module`
   subclass used at training time. If the original training script is not available, the
   downloaded ``state_dict``'s keys and tensor shapes can be inspected directly to
   reconstruct a matching class by hand.

Connectivity
------------

Both pushing (in ``auto`` mode) and pulling rely on the same connectivity check,
:func:`~itwinai.torch.model_hub.utils.has_internet_connection`. Pulling always requires
internet access -- unlike pushing, there is no offline or deferred mode for pulling, since
there is no local fallback artifact to use in its place.
