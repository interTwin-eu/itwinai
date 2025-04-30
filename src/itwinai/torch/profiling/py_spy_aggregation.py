import re
from typing import Dict, List

function_pattern = r"([^\s]+) \(([^():]+):(\d+)\)"


def convert_trace_line_to_dict(trace_line: str, num_samples: int) -> Dict[str, str | int]:
    """Converts a single trace line, which contains a function name, file name and line number,
    to a dictionary with the name, path, line and number of samples.

    Raises:
        ValueError: If the given trace line does not conform to the expected structure of
        "function_name (path/to/function:line_number)"
    """
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
            trace_line_dict = convert_trace_line_to_dict(trace_line, num_samples)
            result.append(trace_line_dict)
        except ValueError as exception:
            raise ValueError(
                f"Failed to convert line from profiling output. Line was: \n\n{line}\nError "
                f"was: {str(exception)}"
            )

    return result


def create_leaf_function_table(
    function_data_list: List,
    num_rows: int | None = None,
    aggregated_leaf_functions: bool = False,
) -> str:
    """Uses the data from `function_data_list` to create a table of the most time consuming
    functions from the bottom of their respective stack traces.
    """
    final_table = ""
    header = f"{'function':<30} {'path':<30} {'line':^5} "
    if not aggregated_leaf_functions:
        header += f"{'itwinai_function':>30} {'itwinai_path':>30} {'itwinai_line':^12} "
    header += f"{'samples':>7}"
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

        final_table += "\n" + f"{function_name[-30:]:<30} {path[-30:]:<30} {line:^5} "

        if not aggregated_leaf_functions:
            itwinai_function_name = function_data["itwinai_function_name"]
            itwinai_path = function_data["itwinai_function_path"]
            itwinai_line = function_data["itwinai_function_line"]
            final_table += f"{itwinai_function_name[-30:]:>30} {itwinai_path[-30:]:>30} "
            final_table += f"{itwinai_line:^12} "

        final_table += f"{num_samples:>7}"

    return final_table
