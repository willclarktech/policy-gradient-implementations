import sys

from policy_gradients.runner import algorithms, run
from policy_gradients.parser import create_parser

parser = create_parser(algorithms.keys())
args = parser.parse_args()

try:
    run(vars(args))
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)
