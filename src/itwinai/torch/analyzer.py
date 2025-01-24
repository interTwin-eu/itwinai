import io
import cProfile
import pstats
from pathlib import Path

class TrainingAnalyzer:

    def __init__(self, output_file: str | Path) -> None:
        self.profile = cProfile.Profile()

        if isinstance(output_file, str): 
            output_file = Path(output_file)
        self.output_file = output_file

    def start_analysis(self):
        self.profile.enable()

    def stop_analysis(self):
        self.profile.disable()
        self.profile.create_stats()

        stats_string = io.StringIO()
        stats = pstats.Stats(self.profile, stream=stats_string)
        stats.strip_dirs()
        stats.sort_stats("tottime", "ncalls")
        stats.print_stats()

        with open(self.output_file, "w") as f: 
            f.write(stats_string.getvalue())
        print(f"Wrote profiling statistics to {self.output_file.resolve()}")
