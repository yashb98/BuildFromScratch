"""A small, deterministic HumanEval-style coding benchmark for the coding-agent
harness-search target. Each task = a problem prompt + a hidden assert-based test
suite + a reference solution (used ONLY to verify the runner; never shown to a
searched harness). Split into SEARCH and HELDOUT task sets — a harness is tuned on
SEARCH and judged on HELDOUT, exactly like the paper's TerminalBench setup.

This file is pure data/CPU. The REWARD machinery (runner.py) is testable now; the
agent-in-the-loop part (a harness actually driving Claude to produce code) is the
staged piece that needs the model — see SKILL.md / TARGETS.md.
"""

# id, entry_point (the function the solution must define), prompt (what the agent
# sees), reference (correct solution — for runner self-test only), tests (asserts).
TASKS = [
    {"id": "add", "entry_point": "add",
     "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"",
     "reference": "def add(a, b):\n    return a + b",
     "tests": "assert add(2,3)==5\nassert add(-1,1)==0\nassert add(0,0)==0"},
    {"id": "is_even", "entry_point": "is_even",
     "prompt": "def is_even(n):\n    \"\"\"Return True iff n is even.\"\"\"",
     "reference": "def is_even(n):\n    return n % 2 == 0",
     "tests": "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True"},
    {"id": "reverse_string", "entry_point": "reverse_string",
     "prompt": "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"",
     "reference": "def reverse_string(s):\n    return s[::-1]",
     "tests": "assert reverse_string('abc')=='cba'\nassert reverse_string('')==''\nassert reverse_string('x')=='x'"},
    {"id": "factorial", "entry_point": "factorial",
     "prompt": "def factorial(n):\n    \"\"\"Return n! for n>=0.\"\"\"",
     "reference": "def factorial(n):\n    r=1\n    for i in range(2,n+1):\n        r*=i\n    return r",
     "tests": "assert factorial(0)==1\nassert factorial(5)==120\nassert factorial(1)==1"},
    {"id": "fib", "entry_point": "fib",
     "prompt": "def fib(n):\n    \"\"\"Return the nth Fibonacci number (fib(0)=0, fib(1)=1).\"\"\"",
     "reference": "def fib(n):\n    a,b=0,1\n    for _ in range(n):\n        a,b=b,a+b\n    return a",
     "tests": "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55"},
    {"id": "is_palindrome", "entry_point": "is_palindrome",
     "prompt": "def is_palindrome(s):\n    \"\"\"Return True iff s reads the same forwards and backwards.\"\"\"",
     "reference": "def is_palindrome(s):\n    return s==s[::-1]",
     "tests": "assert is_palindrome('racecar') is True\nassert is_palindrome('abc') is False\nassert is_palindrome('') is True"},
    {"id": "gcd", "entry_point": "gcd",
     "prompt": "def gcd(a, b):\n    \"\"\"Return the greatest common divisor of a and b.\"\"\"",
     "reference": "def gcd(a, b):\n    while b:\n        a,b=b,a%b\n    return a",
     "tests": "assert gcd(12,8)==4\nassert gcd(17,5)==1\nassert gcd(0,9)==9"},
    {"id": "count_vowels", "entry_point": "count_vowels",
     "prompt": "def count_vowels(s):\n    \"\"\"Return the number of vowels (aeiou, lowercase) in s.\"\"\"",
     "reference": "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s)",
     "tests": "assert count_vowels('hello')==2\nassert count_vowels('xyz')==0\nassert count_vowels('aeiou')==5"},
    {"id": "sum_list", "entry_point": "sum_list",
     "prompt": "def sum_list(xs):\n    \"\"\"Return the sum of the list xs (0 for empty).\"\"\"",
     "reference": "def sum_list(xs):\n    t=0\n    for x in xs:\n        t+=x\n    return t",
     "tests": "assert sum_list([1,2,3])==6\nassert sum_list([])==0\nassert sum_list([-1,1])==0"},
    {"id": "flatten", "entry_point": "flatten",
     "prompt": "def flatten(xss):\n    \"\"\"Flatten one level: [[1,2],[3]] -> [1,2,3].\"\"\"",
     "reference": "def flatten(xss):\n    out=[]\n    for xs in xss:\n        out.extend(xs)\n    return out",
     "tests": "assert flatten([[1,2],[3]])==[1,2,3]\nassert flatten([])==[]\nassert flatten([[],[1]])==[1]"},
]

# fixed partition: tune the harness on SEARCH, judge on HELDOUT (never overlap)
SEARCH_TASKS = TASKS[:5]
HELDOUT_TASKS = TASKS[5:]
BY_ID = {t["id"]: t for t in TASKS}
