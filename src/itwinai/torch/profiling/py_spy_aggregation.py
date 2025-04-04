import re
from typing import Dict, List

function_pattern = r"([^\s]+) \(([^():]+):(\d+)\)"


def handle_trace_line(trace_line: str, num_samples: int) -> Dict[str, str | int]:
    pattern_match = re.match(function_pattern, trace_line)
    assert pattern_match is not None, f"trace_line was: {trace_line}"
    function_name, function_path, line = pattern_match.groups()

    if "frozen importlib" in function_path:
        function_path = "built-in"
        line = "-1"

    return {
        "name": function_name,
        "path": function_path,
        "line": line,
        "num_samples": num_samples,
    }


def handle_data_point(line: str) -> List[Dict]:
    """Handles a single line of the raw profiling output, which contains a stack trace
    and a number of samples
    """
    # The number of samples is the last number, separated from the rest with a space
    # Note: There can be multiple spaces, however, so need to do -1 for the index to get last
    num_samples_str = line.split(" ")[-1].strip()
    assert num_samples_str.isnumeric()
    num_samples = int(num_samples_str)

    # Everything except the 'num_samples'
    stack_trace = " ".join(line.split(" ")[:-1]).strip()
    if not stack_trace:
        return []

    # The various function calls are separed with semicolons
    stack_trace = stack_trace.split(";")
    stack_trace = [trace_line for trace_line in stack_trace if trace_line]

    return [handle_trace_line(trace_line, num_samples) for trace_line in stack_trace]


def create_bottom_function_table(function_data_list: List, num_rows: int | None = None) -> str:
    final_table = ""
    header = f"{'samples':>7} {'function':>35} {'path':>35} {'line':>5}"
    final_table += header
    final_table += "\n" + "=" * len(header)

    # If not passed the do the whole table
    if num_rows is None:
        num_rows = len(function_data_list)

    for function_data in function_data_list[:num_rows]:
        num_samples = function_data["num_samples"]
        function_name = function_data["name"]
        path = function_data["path"]
        line = function_data["line"]
        final_table += "\n" + f"{num_samples:<7} {function_name:>35} {path:>35} {line:>5}"
    return final_table
