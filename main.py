# from itwinai.torch.analyzer import TrainingAnalyzer
# from scalene import scalene_profiler
# import scalene
from pyinstrument import Profiler

def fib(n):
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)


def main():
    profiler = Profiler()
    profiler.start()
    fib(35)
    profiler.stop()
    profiler.print(show_all=True)

if __name__ == "__main__":
    main()
