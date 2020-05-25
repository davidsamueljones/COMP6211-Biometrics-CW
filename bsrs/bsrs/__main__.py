"""Main method for interacting with denoiser through CLI.
"""

import sys
import ssdn.cli

from typing import List


def start_cli(args: List[str] = None):
    if args is not None:
        sys.argv[1:] = args
    ssdn.cli.start()


if __name__ == "__main__":
    # Use arguments provided from command line
    start_cli()
