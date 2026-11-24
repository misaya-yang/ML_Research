import re

SYSTEM = "Published Qwen-Math plain completion template (Power sampling)."


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j : k - 1] if depth == 0 else None


def norm(s):
    if s is None:
        return None
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


def prompt_text(tok, row):
    return "Can you solve the following math problem? " + row["problem"] + r" Please reason step by step, and put your final answer within \boxed{{}}."


def prompt_ids(tok, row):
    return tok.encode(prompt_text(tok, row))
