import sys

from policy_gradients.runner import algorithms, run
from policy_gradients.parser import create_parser

parser = create_parser(algorithms.keys())
options = vars(parser.parse_args())
show_trace = options.pop("trace", False)

try:
    run(options)
# pylint: disable=broad-except
except Exception as exception:
    if show_trace:
        raise exception
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
