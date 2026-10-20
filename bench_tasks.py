"""Canonical task names for the benchmarks that model cards report (`bench:*` fields).

Grouping only: raw names are always shown next to their group, and nothing here changes a score. Anything the rules do not
recognise is 'unclassified' and left out of the grouped views until a rule is added."""
import re

# (canonical task, regex over the lower-cased name with spaces/underscores/hyphens removed); first match wins
RULES = [
    ("AIME 2024", r"^aime(20)?24"), ("AIME 2025", r"^aime(20)?25"), ("AIME 2026", r"^aime(20)?26"), ("AIME (other)", r"^aime"),
    ("LiveCodeBench v5", r"^(livecodebench|lcb)v5"), ("LiveCodeBench v6", r"^(livecodebench|lcb)v6"), ("LiveCodeBench (version not stated)", r"^(livecodebench|lcb)"),
    ("GPQA Diamond", r"^gpqa.*diamond"), ("GPQA (subset not stated)", r"^gpqa"), ("MMLU-Pro", r"^mmlupro"), ("MMLU-Redux", r"^mmluredux"), ("MMLU", r"^mmlu(?!pro|redux|14k)"),
    ("GSM8K-Platinum", r"^gsm8kplatinum"), ("GSM8K", r"^gsm8k"), ("HumanEval+", r"^humaneval(plus|\+)"), ("HumanEval", r"^humaneval"), ("MBPP+", r"^mbpp(plus|\+)"), ("MBPP", r"^mbpp"),
    ("MATH-500", r"^math500"), ("MATH", r"^math(?!vision|vista)"), ("MathVision", r"^mathvision"), ("IFEval", r"^ifeval"), ("IFBench", r"^ifbench"),
    ("HellaSwag", r"^hellaswag"), ("ARC-Challenge", r"^arc(c|challenge)"), ("ARC-Easy", r"^arc(e|easy)"), ("PIQA", r"^piqa"), ("Winogrande", r"^winogrande"),
    ("TruthfulQA", r"^truthfulqa"), ("OpenBookQA", r"^openbookqa"), ("BoolQ", r"^boolq"), ("SciCode", r"^scicode"), ("Terminal-Bench", r"^terminalbench"),
    ("SWE-bench", r"^swebench"), ("BFCL", r"^bfcl"), ("tau-bench", r"^tau\d?bench|^taubench|^tau2"), ("MMMU-Pro", r"^mmmupro"), ("MMMU", r"^mmmu"),
    ("RealWorldQA", r"^realworldqa"), ("ERQA", r"^erqa"), ("AA-LCR", r"^aalcr"), ("SimpleQA", r"^simpleqa"), ("HLE", r"^(hle|humanitys?lastexam)"),
    ("MGSM", r"^mgsm"), ("MuSR", r"^musr"), ("OCRBench", r"^ocrbench"), ("DeepSWE", r"^deepswe"), ("CMMLU", r"^cmmlu"), ("C-Eval", r"^ceval"), ("MRCR", r"^mrcr"), ("HMMT", r"^hmmt"), ("SuperGPQA", r"^supergpqa"), ("RULER", r"^ruler"), ("Tool-Eval-Bench", r"^tool(eval|-eval)bench|^toolevalbench"), ("DROP", r"^drop"), ("BBH", r"^bbh|^bigbench"), ("CodeForces", r"^codeforces"), ("EvalPlus", r"^evalplus"),
]
NOISE = re.compile(r"first-?\d|smoke|asr|speed|tok/?s|tps|stderr|err|latency|ttft|kld|ppl|perplex|count|^n$|samples|time|memory|vram|overall|^avg$|average|^call$|abstain|divmed|knowledge$", re.I)

def canonical(name):
    """Canonical task for a raw benchmark name, None for non-task fields (sub-scores, rates, speeds), 'unclassified' if unknown."""
    if "/" in name or NOISE.search(name): return None
    key = re.sub(r"[\s_\-:()]", "", name.lower())
    for task, rx in RULES:
        if re.search(rx, key):
            version = re.search(r"(\d+(?:\.\d+)?)", name)
            if task == "Terminal-Bench" and version: return f"Terminal-Bench {version.group(1)}"
            if task == "BFCL" and re.search(r"v(\d)", name, re.I): return "BFCL v" + re.search(r"v(\d)", name, re.I).group(1)
            if task == "SWE-bench":
                for tag in ("Verified", "Pro", "Multilingual"):
                    if tag.lower() in name.lower(): return f"SWE-bench {tag}"
            return task
    return "unclassified"

