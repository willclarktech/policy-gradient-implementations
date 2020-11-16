from policy_gradients.runner import algorithms, run
from policy_gradients.parser import create_parser

parser = create_parser(algorithms.keys())
args = parser.parse_args()
run(vars(args))
