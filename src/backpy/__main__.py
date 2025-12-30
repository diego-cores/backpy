"""
CLI module

Allows you to execute:
    python -m backpy
"""

from time import time
import contextlib
import argparse
import io

from . import __version__

def run_test() -> None:
    """
    Run test

    Execute backpy smoke test.
    """
    print("Running backpy smoke test...")

    t = time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):

        from pandas import DataFrame
        import logging

        from . import main
        from . import strategy
        
        logging.basicConfig(level=logging.ERROR)

        main.load_data(DataFrame({
            'open':[1,3,5],
            'high':[3,5,7],
            'low':[1,2,4],
            'close':[2,4,6],
            'volume':[1,1,1]
        }))

        class MinTest(strategy.StrategyClass):
            def next(self):
                if len(self.prev_positions('index')) > 0:
                    self.act_close(0)

                self.act_taker(amount=100)

        main.run(MinTest)

    print(f"Smoke test passed in: {time()-t:.4f}s.")

def main() -> None:
    """
    CLI main function.

    Handles command-line arguments and executes the corresponding actions.

    Args:
        --version: Displays the package version.
        --test: Runs a smoke test of the package.
    """

    parser = argparse.ArgumentParser(prog="backpyf")

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Perform a smoke test."
    )

    args = parser.parse_args()

    if args.test:
        run_test()

if __name__ == "__main__":
    main()
