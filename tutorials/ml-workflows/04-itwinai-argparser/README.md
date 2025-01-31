# itwinai ArgumentParser

**Author(s)**: Matteo Bunino

itwinai provides a wrapper of jsonarparse's ArgumentParser which supports
configuration files by default.

To run as usual:

```bash
python main.py -d 20 --train-prop 0.7 --val-prop 0.2 --lr 1e-5
```

To reuse the parameters saved in a configuration file and override some
parameter (e.g., learning rate):

```bash
python main.py --config advanced_tutorial_conf.yaml --lr 2e-3
```
