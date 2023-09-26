from typing import Hashable, Dict


def clear_key(
        my_dict: Dict,
        dict_name: str,
        key: Hashable,
        complain: bool = True
) -> Dict:
    """Remove key from dictionary if present and complain.

    Args:
        my_dict (Dict): Dictionary.
        dict_name (str): name of the dictionary.
        key (Hashable): Key to remove.
    """
    if key in my_dict:
        if complain:
            print(
                f"Field '{key}' should not be present "
                f"in dictionary '{dict_name}'"
            )
        del my_dict[key]
    return my_dict
