import sys

from policy_gradients.runner import algorithms, run
from policy_gradients.parser import create_parser

parser = create_parser(algorithms.keys())
args = parser.parse_args()

try:
    run(vars(args))
# pylint: disable=broad-except
except Exception as exception:
    print(f"{type(exception).__name__}: {exception}")
    sys.exit(1)
