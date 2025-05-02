import re
from typing import Dict, List


def add_lowest_itwinai_function(stack_traces: List[List[Dict]]) -> None:
    """Iterates through each stack traces in the given list and finds the lowest ``itwinai``
    call, if any, and adds it to each dictionary in the trace.

    This function changes the given list in-place.
    """
    trace: List[Dict]
    for trace in stack_traces:
        itwinai_path = "Not Found"
        itwinai_function = "Not Found"
        itwinai_line = "Not Found"
        for function_dict in reversed(trace):
            if "itwinai" not in function_dict["path"]:
                continue

            itwinai_path = function_dict["path"]
            itwinai_function = function_dict["name"]
            itwinai_line = function_dict["line"]
            # Since we iterate backwards then we stop once we found a match
            break

        # Adding the last one that was found (lowest) to all the dicts
        for function_dict in trace:
            function_dict["itwinai_function_name"] = itwinai_function
            function_dict["itwinai_function_path"] = itwinai_path
            function_dict["itwinai_function_line"] = itwinai_line


def get_aggregated_paths(functions: List[Dict]) -> List[Dict]:
    """Return a new list where all entries with the same information have been combined
    into one. This means that if the name, path and line for the function itself and the
    itwinai function are the same in two entries, then they are combined into one, with their
    number of samples summed.
    """
    aggregated_functions = {}
    for entry in functions:
        key = (
            entry["name"],
            entry["path"],
            entry["line"],
            entry["itwinai_function_name"],
            entry["itwinai_function_path"],
            entry["itwinai_function_line"],
        )
        aggregated_functions[key] = aggregated_functions.get(key, 0) + entry["num_samples"]
    aggregated_functions = [
        {
            "name": key[0],
            "path": key[1],
            "line": key[2],
            "itwinai_function_name": key[3],
            "itwinai_function_path": key[4],
            "itwinai_function_line": key[5],
            "num_samples": value,
        }
        for key, value in aggregated_functions.items()
    ]
    return aggregated_functions


def parse_trace_line_to_dict(trace_line: str, num_samples: int) -> Dict[str, str | int]:
    """Parses a single trace line, which contains a function name, file name and line number,
    to a dictionary with the name, path, line and number of samples.

    Raises:
        ValueError: If the given trace line does not conform to the expected structure of
        "function_name (path/to/function:line_number)"
    """
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
            result.append(trace_line_dict)
        except ValueError as exception:
            raise ValueError(
                f"Failed to convert line from profiling output. Line was: \n\n{line}\nError "
                f"was: {str(exception)}"
            )

    return result
