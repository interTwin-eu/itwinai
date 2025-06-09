import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel


class StackFrame(BaseModel):
    """Represents a single stack frame in a call stack."""

    name: str
    path: str
    line: str
    num_samples: int
    proportion: float | None = None

    # Allows the user to specify which library they want to tag
    library_function_name: str = "Not Found"
    library_function_path: str = "Not Found"
    library_function_line: str = "Not Found"

    def get_aggregation_key(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.name,
            self.path,
            self.line,
            self.library_function_name,
            self.library_function_path,
            self.library_function_line,
        )

    # For sorting purposes
    def __lt__(self, other: "StackFrame") -> bool:
        return self.num_samples < other.num_samples


def add_library_information(
    stack_traces: List[List[StackFrame]], library_name: str = "itwinai"
) -> None:
    """Iterates through each stack traces in the given list and finds the lowest call with
    the given library name, if any, and adds it to each dictionary in the trace.

    This function changes the given list in-place.
    """
    trace: List[StackFrame]
    for trace in stack_traces:
        library_path: str | None = None
        library_function: str | None = None
        library_line: str | None = None

        # Finding the lowest match
        for stack_frame in reversed(trace):
            if library_name not in stack_frame.path:
                continue

            library_path = stack_frame.path
            library_function = stack_frame.name
            library_line = stack_frame.line
            break

        # Only add something if we actually found the library (redundancy for linting)
        if library_path is None or library_function is None or library_line is None:
            continue

        # Adding the last one that was found (lowest) to all the dicts
        for stack_frame in trace:
            stack_frame.library_function_name = library_function
            stack_frame.library_function_path = library_path
            stack_frame.library_function_line = library_line


def get_aggregated_paths(stack_frames: List[StackFrame]) -> List[StackFrame]:
    """Aggregate stack frames with identical keys by summing their sample counts.

    Two stack frames are considered identical if their `get_aggregation_key()` values match.
    Returns a new list of stack frames, each representing a unique key with the
    corresponding total number of samples.
    """
    aggregates: Dict[Tuple, StackFrame] = {}
    for frame in stack_frames:
        key = frame.get_aggregation_key()
        if key not in aggregates:
            aggregates[key] = frame.model_copy()
        else:
            aggregates[key].num_samples += frame.num_samples

    return list(aggregates.values())


def parse_trace_line_to_stack_frame(trace_line: str, num_samples: int) -> StackFrame | None:
    """Parses a single trace line, which contains a function name, file name and line number,
    to a StackFrame object.

    Raises:
        ValueError: If the given trace line does not conform to the expected structure of
        "function_name (path/to/function:line_number)".
    """

    # We skip the patterns that specify which process is running
    process_pattern = r"process [0-9]+:\".*\""
    process_match = re.match(process_pattern, trace_line)
    if process_match:
        return None

    # We also skip lines that include `<module>`, since these don't give any useful information
    if "<module>" in trace_line:
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

    return StackFrame(
        name=function_name, path=function_path, line=line, num_samples=num_samples
    )


def convert_stack_trace_to_list(line: str) -> List[StackFrame]:
    """Converts a single line of the raw profiling output, which contains a stack trace
    and a number of samples, to a list of StackFrame objects with the data for each line of the
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
            stack_frame = parse_trace_line_to_stack_frame(trace_line, num_samples)
            if stack_frame is None:
                continue
            result.append(stack_frame)
        except ValueError as exception:
            raise ValueError(
                f"Failed to convert line from profiling output. Line was: \n\n{line}\nError "
                f"was: {str(exception)}"
            )

    return result


def parse_py_spy_lines(file_lines: List[str]) -> List[List[StackFrame]]:
    """Parse the lines of a py-spy-profiling output and turn it into a list of call stacks,
    each of which is a list of

    Raises:
        ValueError: If converting the stack trace to a list fails.
    """
    stack_traces: List[List[StackFrame]] = []
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
    """Parses the number of rows from a string. Makes sure it is either "all" or greater
    than zero. If it is "all" then it returns None.

    Raises:
        ValueError: If `num_rows` is not numeric or "all".
        ValueError: If 'num_rows` is parsed to a number smaller than one.
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


def read_stack_traces(path: Path) -> List[List[StackFrame]]:
    """Reads stack traces from a path. The path can either point to a file or a directory.
    If it points to a directory, it will read all files in this directory and combine them.

    Raises:
        ValueError: If parsing the lines from one of the files fails.
    """
    if not path.is_dir():
        with path.open("r") as f:
            profiling_data = f.readlines()
        return parse_py_spy_lines(file_lines=profiling_data)

    # Try to read the data from all of the files in the directory
    stack_traces = []
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
