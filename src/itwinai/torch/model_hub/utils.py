def has_internet_connection(timeout: float = 3.0) -> bool:
    """Checks for internet connectivity."""
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=timeout)
        return True
    except OSError:
        return False
