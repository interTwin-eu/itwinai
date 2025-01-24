from itwinai.torch.analyzer import TrainingAnalyzer

def fib(n):
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)


def main():
    output_file = "stats.txt"
    analyzer = TrainingAnalyzer(output_file=output_file)
    analyzer.start_analysis()
    fib(30)
    analyzer.stop_analysis()

if __name__ == "__main__":
    main()
