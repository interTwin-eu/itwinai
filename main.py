# from itwinai.torch.analyzer import TrainingAnalyzer
# from scalene import scalene_profiler
# import scalene
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

    # for elem in stats:
    #     print(f"elem.name: {elem.name}")
    #     print(f"elem.ncall: {elem.ncall}")
    #     print(f"elem.ttot: {elem.ttot}")
    #     print(f"elem.module: {elem.module}")
    #     print(f"elem.lineno: {elem.lineno}")
    #     print(f"elem.builtin: {elem.builtin}")
    #     print(f"elem.ctx_name: {elem.ctx_name}")
    #     print()

    stats.save(path="my_stats.pstats", type="pstat")

    # profiler.stop()
    # profiler.print(show_all=True, flat=True)
    # profiler.open_in_browser()

if __name__ == "__main__":
    main()
