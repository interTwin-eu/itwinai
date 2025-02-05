import yappi
from itwinai.torch.trainer import TorchTrainer

def fib(n):
    val = 1
    conf = {}
    obj = TorchTrainer(config=conf, epochs=2)
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)


def main():
    yappi.start()
    fib(15)
    yappi.stop()
    stats = yappi.get_func_stats()

    pstats = yappi.convert2pstats(stats)
    pstats.dump_stats(filename="my_stats.pstat")


if __name__ == "__main__":
    main()
