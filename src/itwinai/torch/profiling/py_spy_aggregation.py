import re
from typing import Dict, List
from pathlib import Path


def add_lowest_library_function(
    stack_traces: List[List[Dict]], library_name: str = "itwinai"
) -> None:
    """Iterates through each stack traces in the given list and finds the lowest call with
    the given library name, if any, and adds it to each dictionary in the trace.

    This function changes the given list in-place.
    """
    trace: List[Dict]
    for trace in stack_traces:
        library_path = "Not Found"
        library_function = "Not Found"
        library_line = "Not Found"
        for function_dict in reversed(trace):
            if library_name not in function_dict["path"]:
                continue

            library_path = function_dict["path"]
            library_function = function_dict["name"]
            library_line = function_dict["line"]
            # Since we iterate backwards then we stop once we found a match
            break

        # Adding the last one that was found (lowest) to all the dicts
        for function_dict in trace:
            function_dict["library_function_name"] = library_function
            function_dict["library_function_path"] = library_path
            function_dict["library_function_line"] = library_line


def get_aggregated_paths(functions: List[Dict]) -> List[Dict]:
    """Return a new list where all entries with the same information have been combined
    into one. This means that if the name, path and line for the function itself and the
    specified library function are the same in two entries, then they are combined into one,
    with their number of samples summed.
    """
    aggregated_functions = {}
    for entry in functions:
        key = (
            entry["name"],
            entry["path"],
            entry["line"],
            entry["library_function_name"],
            entry["library_function_path"],
            entry["library_function_line"],
        )
        aggregated_functions[key] = aggregated_functions.get(key, 0) + entry["num_samples"]
    aggregated_functions = [
        {
            "name": key[0],
            "path": key[1],
            "line": key[2],
            "library_function_name": key[3],
            "library_function_path": key[4],
            "library_function_line": key[5],
            "num_samples": value,
        }
        for key, value in aggregated_functions.items()
    ]
    return aggregated_functions


def parse_trace_line_to_dict(trace_line: str, num_samples: int) -> Dict[str, str | int] | None:
    """Parses a single trace line, which contains a function name, file name and line number,
    to a dictionary with the name, path, line and number of samples.

    Raises:
        ValueError: If the given trace line does not conform to the expected structure of
        "function_name (path/to/function:line_number)"
    """

    # Skip all the process patterns
    process_pattern = r"process [0-9]+:\".*\""
    process_match = re.match(process_pattern, trace_line)
    if process_match is not None:
        return None

    function_pattern = r"([^\s]+) \(([^():]+):(\d+)\)"
    pattern_match = re.match(function_pattern, trace_line)
    if pattern_match is None:
        raise ValueError(
            f"Failed to extract function information from trace line: '{trace_line}'."
        )
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


def convert_stack_trace_to_list(line: str) -> List[Dict[str, str | int]]:
    """Converts a single line of the raw profiling output, which contains a stack trace
    and a number of samples, to a list of dictionaries with the data for each line of the
    stack trace.

    Note:
        The number of samples is the last number in the string, separated from the rest with
        a space. There can be multiple spaces, however, so we have to extract the last element.

    Raises:
        ValueError: If any of the samples are not numeric.
        ValueError: If any of the lines in the stack trace are not formatted as expected.
    """
    num_samples_str = line.split(" ")[-1].strip()

    if not num_samples_str.isnumeric():
        raise ValueError(
            f'Extracted number of samples, "{num_samples_str}", is not all numeric. Was part'
            f'of line: "{line}".'
        )
    num_samples = int(num_samples_str)

    # Everything except the 'num_samples'
    stack_trace = " ".join(line.split(" ")[:-1]).strip()
    if not stack_trace:
        return []

    # The various function calls are separated with semicolons
    stack_trace = stack_trace.split(";")
    stack_trace = [trace_line for trace_line in stack_trace if trace_line]

    result = []
    for trace_line in stack_trace:
        try:
            trace_line_dict = parse_trace_line_to_dict(trace_line, num_samples)
            if trace_line_dict is None:
                continue
            result.append(trace_line_dict)
        except ValueError as exception:
            raise ValueError(
                f"Failed to convert line from profiling output. Line was: \n\n{line}\nError "
                f"was: {str(exception)}"
            )

    return result


def parse_py_spy_lines(file_lines: List[str]) -> List[List[Dict]]:
    """

    Raises:
        ValueError: If converting the stack trace to a list fails
    """
    stack_traces: List[List[Dict]] = []
    for line in file_lines:
        try:
            structured_stack_trace = convert_stack_trace_to_list(line)
            if structured_stack_trace:
                stack_traces.append(structured_stack_trace)
        except ValueError as exception:
            raise ValueError(
                f"Failed to aggregate profiling data with following error:\n{exception}"
            )
    return stack_traces


def parse_num_rows(num_rows: str) -> int | None:
    """

    Raises:
        ValueError: If `num_rows` is not numeric or "all"
        ValueError: If 'num_rows` is parsed to a number smaller than one
    """
    if not num_rows.isnumeric() and num_rows != "all":
        raise ValueError(
            f"Number of rows must be either an integer or 'all'. Was '{num_rows}'.",
        )

    parsed_num_rows: int | None = int(num_rows) if num_rows.isnumeric() else None
    if isinstance(parsed_num_rows, int) and parsed_num_rows < 1:
        raise ValueError(
            f"Number of rows must be at least one! Was '{num_rows}'.",
        )
    return parsed_num_rows

def read_stack_traces(path: Path) -> List[List[Dict]]:
    """

    Raises:
        ValueError: If parsing the lines from one of the files fails
    """
    if not path.is_dir():
        with path.open("r") as f:
            profiling_data = f.readlines()
        return parse_py_spy_lines(file_lines=profiling_data)

    stack_traces = []
    # Try to read the data from all of the files in the directory
    for iter_file in path.iterdir():
        try:
            with iter_file.open("r") as f:
                data = f.readlines()
            file_stack_traces = parse_py_spy_lines(file_lines=data)
            stack_traces += file_stack_traces
        except ValueError as exception:
            raise ValueError(
                f"Failed to read one of the files in directory '{path.resolve()}'"
                f" with following error:\n{str(exception)}"
            )
    return stack_traces
