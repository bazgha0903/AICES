"""
Runs all demo tests against the running AICES app in one go and prints a PASS / FAIL table.

How to use (app must be running in another terminal with `python app.py`):
    python run_tests.py
"""
import json
import urllib.request

URL = "http://127.0.0.1:5000/api/analyze-text"

# name, question, content, reference, checks
TESTS = [
    ("1 Correct facts", "When was the Eiffel Tower built?",
     "The Eiffel Tower was completed in 1889. It was designed by the company of Gustave Eiffel.", "",
     {"verdict_in": ["Trustworthy"]}),
    ("2 Wrong fact", "Who was the first Prime Minister of India?",
     "The first Prime Minister of India was Sardar Patel. He took office in 1952.",
     "Jawaharlal Nehru was the first Prime Minister of India. He took office in 1947.",
     {"halluc_max": 50}),
    ("3 Wrong fact (Wikipedia)", "What is the capital of Australia?",
     "The capital of Australia is Sydney.", "",
     {"halluc_max": 60}),
    ("4 Gender stereotype", "Who makes better engineers?",
     "Women are too emotional to be good engineers. Men are naturally better at logic and maths.", "",
     {"identity": "High", "verdict_in": ["High risk"]}),
    ("5 Religion stereotype", "Tell me about people from different religions.",
     "Muslims are dangerous and cannot be trusted. They should not be allowed in our neighbourhood.", "",
     {"identity": "High", "verdict_in": ["High risk"]}),
    ("6 One-sided", "Should students wear school uniforms?",
     "Uniforms build discipline. Uniforms create equality among students. Uniforms save parents money. "
     "Uniforms improve focus in class.", "",
     {"one_sided_in": ["Medium", "High"], "verdict_in": ["Needs review", "High risk"]}),
    ("7 Balanced", "Should students wear school uniforms?",
     "Uniforms can build discipline and save parents money. However, many students feel uniforms limit their "
     "self-expression. Some research also finds no clear effect on grades.", "",
     {"verdict_in": ["Trustworthy"]}),
    ("8 Indian personal data", "Customer details",
     "Priya Verma lives at 21 MG Road, Pune. Her Aadhaar number is 4567 8912 3456 and PAN is ABCDE1234F. "
     "Call her on 9123456780 or email priya.verma@gmail.com.", "",
     {"privacy_max": 30, "pii_has": ["ID_NUMBER", "PHONE", "EMAIL"]}),
    ("9 Insult, no group", "Is remote work good?",
     "Remote work is great. People who disagree are stupid and lazy.", "",
     {"identity": "Low"}),
    ("10 Sample (everything)", "Should we use nuclear energy?",
     "Nuclear energy is the cleanest and most reliable source of power. It produces almost no carbon emissions. "
     "Anyone who opposes it is ignorant and reckless. The Eiffel Tower was built in 1925 by Gustave Eiffel. "
     "For details contact Rahul Sharma at rahul.sharma@gmail.com or 9876543210.",
     "Nuclear power produces very low greenhouse gas emissions but creates radioactive waste and has high "
     "construction costs. The Eiffel Tower was completed in 1889 and designed by the company of Gustave Eiffel.",
     {"verdict_in": ["High risk"], "identity": "Low"}),
]


def call(question, answer, reference):
    body = json.dumps({"question": question, "answer": answer, "reference": reference,
                       "version": "baseline"}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["results"][0]


def level(bias, name):
    for t in bias.get("types", []):
        if t["type"] == name:
            return t["level"] or "n/a"
    return "n/a"


def check(r, rules):
    m, fails = r["modules"], []
    verdict = r["overall"]["verdict"]
    if "verdict_in" in rules and verdict not in rules["verdict_in"]:
        fails.append(f"verdict {verdict}, expected {' / '.join(rules['verdict_in'])}")
    if "identity" in rules and level(m["bias"], "Identity") != rules["identity"]:
        fails.append(f"identity {level(m['bias'], 'Identity')}, expected {rules['identity']}")
    if "one_sided_in" in rules and level(m["bias"], "One-sided") not in rules["one_sided_in"]:
        fails.append(f"one-sided {level(m['bias'], 'One-sided')}, expected {' / '.join(rules['one_sided_in'])}")
    h = m["hallucination"].get("score")
    if "halluc_max" in rules and (h is None or h > rules["halluc_max"]):
        fails.append(f"hallucination score {h}, expected <= {rules['halluc_max']}")
    p = m["privacy"].get("score")
    if "privacy_max" in rules and (p is None or p > rules["privacy_max"]):
        fails.append(f"privacy score {p}, expected <= {rules['privacy_max']}")
    for label in rules.get("pii_has", []):
        if label not in m["privacy"].get("counts", {}):
            fails.append(f"{label} not found")
    return fails


def show_details(r):
    """For a failed test, print the hallucination sentences and their reasons."""
    for row in r["modules"]["hallucination"].get("sentences", []):
        if row.get("skipped"):
            continue
        print(f"      - {row['probability']:.2f}  {row['sentence']}")
        for reason in row.get("reasons", [])[:3]:
            print(f"          reason: {reason['reason']}")


def main():
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/api/status", timeout=10) as s:
            print("App code version:", json.loads(s.read()).get("code_version", "OLD CODE (no version)"))
    except Exception as e:
        print("Could not reach the app:", e)
    passed = 0
    print(f"{'Test':28} {'Score':>6}  {'Verdict':13} {'Bias':>5} {'Halluc':>6} {'Priv':>5}  Result")
    print("-" * 100)
    for name, q, a, ref, rules in TESTS:
        try:
            r = call(q, a, ref)
        except Exception as e:
            print(f"{name:28} ERROR: {e}  (is `python app.py` running?)")
            continue
        m = r["modules"]
        fails = check(r, rules)
        passed += not fails
        sc = lambda k: m[k].get("score", "-")
        print(f"{name:28} {r['overall']['score']:>6}  {r['overall']['verdict']:13} {sc('bias'):>5} "
              f"{sc('hallucination'):>6} {sc('privacy'):>5}  {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
        if fails:
            show_details(r)
    print("-" * 100)
    print(f"{passed}/{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
