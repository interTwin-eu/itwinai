# """
# Command line interface for out Python application.
# You can call commands from the command line.
# Example

# >>> $ itwinai --help

# """

# from typing import Optional
# import typer

# app = typer.Typer()


# @app.command()
# def hello(name: str):
#     print(f"Hello {name}")


# @app.command()
# def goodbye(name: str, formal: bool = False):
#     if formal:
#         print(f"Goodbye Ms. {name}. Have a good day.")
#     else:
#         print(f"Bye {name}!")


# @app.command()
# def setup_env(
#     deps: Optional[str] = typer.Option(
#         None,
#         help=("Comma separated requirements, each in pip format "
#               "(e.g., 'SomePackage==1.0.4,SomeOtherPackage>=2.3.0'). "
#               "If given, other arguments are ignored.")
#     ),
#     file: Optional[str] = typer.Option(
#         None,
#         help=("Path to requirements text file, in pip format "
#               "(e.g., 'SomePackage==1.0.4)")
#     )
# ):
#     """Install custom dependencies in this Python env using pip.
#     """
#     # TODO: improve using conda/mamba, for better compatibility
#     # with base env. Or give both options...


# if __name__ == "__main__":
#     app()
