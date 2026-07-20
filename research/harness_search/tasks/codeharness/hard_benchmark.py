"""HARD coding tasks for the codeharness headroom check. Unlike benchmark.py (10
trivial tasks a 9B aces under any harness -> zero headroom), each task here has a
known EDGE-CASE TRAP a capable model often misses on the first attempt (overflow
clamping, truncate-toward-zero division, the abba sliding-window reset, spiral
boundary management, …). That first-attempt failure is what a TRACE-USING
self-repair harness can fix and a scalar-only harness cannot — so these tasks are
where "do traces beat scalar feedback" is actually testable.

Each task carries:
  prompt        – function stub shown to the model (the only thing the harness sees)
  reference     – a correct solution (runner self-test ONLY; never shown)
  public_tests  – a few asserts the SELF-REPAIR harness may run + read failures from
  tests         – the HIDDEN grading suite (superset, more edge cases); never shown

public_tests deliberately INCLUDE an edge case or two, so repairing against them can
actually raise the hidden pass rate. Pure data; correctness is verified on CPU by
verify_hard_benchmark() before any GPU run.
"""

TASKS = [
    {
        "id": "my_atoi", "entry_point": "my_atoi",
        "prompt": ('def my_atoi(s):\n'
                   '    """Convert string s to a 32-bit signed integer (C atoi / LeetCode 8):\n'
                   '    skip leading spaces, an optional single +/- sign, read digits until a\n'
                   '    non-digit, ignore the rest. Clamp the result to [-2**31, 2**31-1].\n'
                   '    Return 0 if no digits are read."""'),
        "reference": (
            "def my_atoi(s):\n"
            "    i, n = 0, len(s)\n"
            "    while i < n and s[i] == ' ':\n"
            "        i += 1\n"
            "    sign = 1\n"
            "    if i < n and s[i] in '+-':\n"
            "        sign = -1 if s[i] == '-' else 1\n"
            "        i += 1\n"
            "    num = 0\n"
            "    while i < n and s[i].isdigit():\n"
            "        num = num * 10 + int(s[i])\n"
            "        i += 1\n"
            "    num *= sign\n"
            "    return max(-2**31, min(2**31 - 1, num))\n"),
        "public_tests": (
            'assert my_atoi("42") == 42\n'
            'assert my_atoi("   -42") == -42\n'
            'assert my_atoi("4193 with words") == 4193\n'
            'assert my_atoi("words and 987") == 0\n'
            'assert my_atoi("-91283472332") == -2147483648\n'),
        "tests": (
            'assert my_atoi("42") == 42\n'
            'assert my_atoi("   -42") == -42\n'
            'assert my_atoi("4193 with words") == 4193\n'
            'assert my_atoi("words and 987") == 0\n'
            'assert my_atoi("-91283472332") == -2147483648\n'
            'assert my_atoi("91283472332") == 2147483647\n'
            'assert my_atoi("+-12") == 0\n'
            'assert my_atoi("  +0 123") == 0\n'
            'assert my_atoi("") == 0\n'
            'assert my_atoi("2147483648") == 2147483647\n'),
    },
    {
        "id": "eval_rpn", "entry_point": "eval_rpn",
        "prompt": ('def eval_rpn(tokens):\n'
                   '    """Evaluate a Reverse Polish Notation expression (list of string\n'
                   '    tokens). Operators: + - * /. Division TRUNCATES TOWARD ZERO\n'
                   '    (so -7 / 2 == -3, not -4). Return the integer result."""'),
        "reference": (
            "def eval_rpn(tokens):\n"
            "    st = []\n"
            "    for t in tokens:\n"
            "        if t in ('+', '-', '*', '/'):\n"
            "            b = st.pop(); a = st.pop()\n"
            "            if t == '+': st.append(a + b)\n"
            "            elif t == '-': st.append(a - b)\n"
            "            elif t == '*': st.append(a * b)\n"
            "            else:\n"
            "                q = abs(a) // abs(b)\n"
            "                st.append(q if (a < 0) == (b < 0) else -q)\n"
            "        else:\n"
            "            st.append(int(t))\n"
            "    return st[-1]\n"),
        "public_tests": (
            'assert eval_rpn(["2","1","+","3","*"]) == 9\n'
            'assert eval_rpn(["4","13","5","/","+"]) == 6\n'
            'assert eval_rpn(["-7","2","/"]) == -3\n'),
        "tests": (
            'assert eval_rpn(["2","1","+","3","*"]) == 9\n'
            'assert eval_rpn(["4","13","5","/","+"]) == 6\n'
            'assert eval_rpn(["-7","2","/"]) == -3\n'
            'assert eval_rpn(["3","-4","/"]) == 0\n'
            'assert eval_rpn(["-22","5","/"]) == -4\n'
            'assert eval_rpn(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]) == 22\n'),
    },
    {
        "id": "decode_string", "entry_point": "decode_string",
        "prompt": ('def decode_string(s):\n'
                   '    """Decode a string with the rule k[encoded] = encoded repeated k\n'
                   '    times; brackets may NEST and k may be multi-digit. Examples:\n'
                   "    '3[a2[c]]' -> 'accaccacc', '2[abc]3[cd]ef' -> 'abcabccdcdcdef'.\n"
                   '    Input is always valid."""'),
        "reference": (
            "def decode_string(s):\n"
            "    cur, num, st = '', 0, []\n"
            "    for ch in s:\n"
            "        if ch.isdigit():\n"
            "            num = num * 10 + int(ch)\n"
            "        elif ch == '[':\n"
            "            st.append((cur, num)); cur, num = '', 0\n"
            "        elif ch == ']':\n"
            "            prev, k = st.pop(); cur = prev + cur * k\n"
            "        else:\n"
            "            cur += ch\n"
            "    return cur\n"),
        "public_tests": (
            'assert decode_string("3[a]2[bc]") == "aaabcbc"\n'
            'assert decode_string("3[a2[c]]") == "accaccacc"\n'
            'assert decode_string("10[a]") == "aaaaaaaaaa"\n'),
        "tests": (
            'assert decode_string("3[a]2[bc]") == "aaabcbc"\n'
            'assert decode_string("3[a2[c]]") == "accaccacc"\n'
            'assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"\n'
            'assert decode_string("abc") == "abc"\n'
            'assert decode_string("10[a]") == "aaaaaaaaaa"\n'
            'assert decode_string("2[2[2[a]]]") == "aaaaaaaa"\n'),
    },
    {
        "id": "merge_intervals", "entry_point": "merge_intervals",
        "prompt": ('def merge_intervals(intervals):\n'
                   '    """Merge all overlapping intervals (a list of [start, end]) and\n'
                   '    return the merged list sorted by start. Touching intervals such as\n'
                   '    [1,4] and [4,5] merge into [1,5]. Input may be unsorted."""'),
        "reference": (
            "def merge_intervals(intervals):\n"
            "    if not intervals: return []\n"
            "    xs = sorted(intervals, key=lambda x: x[0])\n"
            "    out = [list(xs[0])]\n"
            "    for s, e in xs[1:]:\n"
            "        if s <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], e)\n"
            "        else:\n"
            "            out.append([s, e])\n"
            "    return out\n"),
        "public_tests": (
            'assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n'
            'assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n'
            'assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n'),
        "tests": (
            'assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n'
            'assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n'
            'assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n'
            'assert merge_intervals([]) == []\n'
            'assert merge_intervals([[1,4],[2,3]]) == [[1,4]]\n'
            'assert merge_intervals([[1,4],[5,6]]) == [[1,4],[5,6]]\n'),
    },
    {
        "id": "spiral_order", "entry_point": "spiral_order",
        "prompt": ('def spiral_order(matrix):\n'
                   '    """Return all elements of the m x n matrix in clockwise spiral order\n'
                   '    starting top-left. e.g. [[1,2,3],[4,5,6],[7,8,9]] ->\n'
                   '    [1,2,3,6,9,8,7,4,5]. Handle non-square and empty matrices."""'),
        "reference": (
            "def spiral_order(matrix):\n"
            "    if not matrix or not matrix[0]: return []\n"
            "    res = []\n"
            "    top, bot = 0, len(matrix) - 1\n"
            "    left, right = 0, len(matrix[0]) - 1\n"
            "    while top <= bot and left <= right:\n"
            "        for c in range(left, right + 1): res.append(matrix[top][c])\n"
            "        top += 1\n"
            "        for r in range(top, bot + 1): res.append(matrix[r][right])\n"
            "        right -= 1\n"
            "        if top <= bot:\n"
            "            for c in range(right, left - 1, -1): res.append(matrix[bot][c])\n"
            "            bot -= 1\n"
            "        if left <= right:\n"
            "            for r in range(bot, top - 1, -1): res.append(matrix[r][left])\n"
            "            left += 1\n"
            "    return res\n"),
        "public_tests": (
            'assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n'
            'assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]\n'),
        "tests": (
            'assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n'
            'assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]\n'
            'assert spiral_order([[1,2,3]]) == [1,2,3]\n'
            'assert spiral_order([[1],[2],[3]]) == [1,2,3]\n'
            'assert spiral_order([[1]]) == [1]\n'
            'assert spiral_order([]) == []\n'),
    },
    {
        "id": "longest_unique", "entry_point": "length_of_longest_substring",
        "prompt": ('def length_of_longest_substring(s):\n'
                   '    """Return the length of the longest substring of s containing no\n'
                   "    repeating characters. 'abcabcbb' -> 3, 'bbbbb' -> 1, 'pwwkew' -> 3.\n"
                   '    The sliding window start must never move backwards."""'),
        "reference": (
            "def length_of_longest_substring(s):\n"
            "    seen = {}\n"
            "    start = best = 0\n"
            "    for i, ch in enumerate(s):\n"
            "        if ch in seen and seen[ch] >= start:\n"
            "            start = seen[ch] + 1\n"
            "        seen[ch] = i\n"
            "        best = max(best, i - start + 1)\n"
            "    return best\n"),
        "public_tests": (
            'assert length_of_longest_substring("abcabcbb") == 3\n'
            'assert length_of_longest_substring("pwwkew") == 3\n'
            'assert length_of_longest_substring("abba") == 2\n'),
        "tests": (
            'assert length_of_longest_substring("abcabcbb") == 3\n'
            'assert length_of_longest_substring("bbbbb") == 1\n'
            'assert length_of_longest_substring("pwwkew") == 3\n'
            'assert length_of_longest_substring("") == 0\n'
            'assert length_of_longest_substring(" ") == 1\n'
            'assert length_of_longest_substring("abba") == 2\n'
            'assert length_of_longest_substring("tmmzuxt") == 5\n'),
    },
]

BY_ID = {t["id"]: t for t in TASKS}
