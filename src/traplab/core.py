"""Strict proposal validation and exact, visible-data answer keys."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
import re
from statistics import NormalDist

FAMILIES = ("simpson", "collider", "rtm")
PROTOCOL = "c-lite-v1"
STORY_FIELDS = {"id", "domain", "unit", "outcome"}
FORBIDDEN = re.compile(r"\d|\b(before|after|because|cause\w*|leads?\s+to|due\s+to|then|previous\w*|subsequent\w*|improv\w*|wors\w*|treat\w*|select\w*)\b", re.I)
DEFAULT_STORIES = [
    {"id": "workshop", "domain": "workshop", "unit": "component", "outcome": "quality"},
    {"id": "garden", "domain": "garden", "unit": "plant", "outcome": "vigor"},
    {"id": "studio", "domain": "studio", "unit": "recording", "outcome": "clarity"},
    {"id": "classroom", "domain": "classroom", "unit": "learner", "outcome": "score"},
    {"id": "warehouse", "domain": "warehouse", "unit": "package", "outcome": "rating"},
]


class Rejected(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_story(story):
    if not isinstance(story, dict) or set(story) != STORY_FIELDS:
        raise Rejected("story_schema")
    for value in story.values():
        if not isinstance(value, str) or not value.strip() or len(value) > 60 or FORBIDDEN.search(value):
            raise Rejected("story_field")
        if not re.fullmatch(r"[A-Za-z][A-Za-z -]*", value):
            raise Rejected("story_characters")
    return story


def number(p, key, low, high, integer=False):
    value = p.get(key)
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise Rejected(f"bounds:{key}")
    if integer and type(value) is not int:
        raise Rejected(f"integer:{key}")
    return value


def keys(p, expected):
    if not isinstance(p, dict) or set(p) != set(expected.split()):
        raise Rejected("parameter_schema")


def classify(d, rtm=False):
    # Decimal prevents binary float drift at registered thresholds.
    d = Decimal(str(d))
    low, high = (Decimal("0.05"), Decimal("0.2")) if rtm else (Decimal("0.01"), Decimal("0.03"))
    if abs(d) < low:
        return 0
    if abs(d) < high:
        raise Rejected("ambiguity_band")
    return 1 if d > 0 else -1


@dataclass
class Item:
    id: str
    family: str
    params: dict
    story_id: str
    cell: str
    tag: str
    prompt: str
    options: list[str]
    answer: str
    visible: dict
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def build(proposal, stories=DEFAULT_STORIES):
    if not isinstance(proposal, dict) or set(proposal) != {"family", "params", "story_id"}:
        raise Rejected("proposal_schema")
    family, p = proposal["family"], proposal["params"]
    if family not in FAMILIES:
        raise Rejected("family")
    pool = {s["id"]: validate_story(s) for s in stories}
    if not isinstance(proposal["story_id"], str) or proposal["story_id"] not in pool:
        raise Rejected("story_id")
    story = pool[proposal["story_id"]]
    header = f"Domain: {story['domain']}. Unit: {story['unit']}. Outcome: {story['outcome']}.\n"
    if family == "simpson":
        keys(p, "n_a0 n_a1 n_b0 n_b1 p_a0 p_a1 p_b0 p_b1 z_role")
        if p["z_role"] not in ("confounder", "mediator", "unstated"):
            raise Rejected("z_role")
        n = {k: number(p, "n_" + k, 20, 2000, True) for k in ("a0", "a1", "b0", "b1")}
        rates = {k: number(p, "p_" + k, .05, .95) for k in n}
        # Counts are exact expectations, displayed with two decimal places.
        successes = {k: f"{n[k] * rates[k]:.2f}" for k in n}
        r = {k: Decimal(successes[k]) / n[k] for k in n}
        total = sum(n.values())
        adjusted = sum(Decimal(n['a'+z] + n['b'+z]) / total * (r['a'+z] - r['b'+z]) for z in ('0', '1'))
        aggregate = sum(Decimal(successes['a'+z]) for z in ('0', '1')) / (n['a0'] + n['a1']) - sum(Decimal(successes['b'+z]) for z in ('0', '1')) / (n['b0'] + n['b1'])
        sa, sg = classify(adjusted), classify(aggregate)
        answers = {1: "A_better", -1: "B_better", 0: "no_difference"}
        role = p["z_role"]
        answer = answers[sa] if role == "confounder" else answers[sg] if role == "mediator" else answers[sa] if sa == sg else "insufficient_info"
        assumptions = {
            "confounder": "The complete causal DAG is Z→T, Z→Y, T→Y, with no hidden common causes. The target population has P(Z=z) equal to the pooled table proportions. Treatment is exchangeable within Z.",
            "mediator": "Treatment T was randomized; Z is measured downstream of T. The complete causal DAG allows T→Z, Z→Y and T→Y, with no hidden common causes. Compare the total effects of assigning T.",
            "unstated": "Exactly one of two causal models holds: (1) complete DAG Z→T, Z→Y, T→Y, no hidden common causes, exchangeability within Z and target Z proportions equal to the pooled table; or (2) randomized T with downstream Z and DAG T→Z, Z→Y, T→Y. Which model holds is not stated.",
        }
        visible = {"n": n, "successes": successes, "z_role": role}
        table = "\n".join(f"Z={z}, T={t.upper()}: units={n[t+z]}, expected successes={successes[t+z]}" for z in ('0', '1') for t in ('a', 'b'))
        prompt = header + assumptions[role] + "\nThese are exact expected counts, not noisy sample estimates. Success is the favorable binary outcome.\n" + table + "\nWhich treatment gives a higher success rate if assigned to the entire target population? Use the stated causal assumptions. Differences below 0.01 count as no difference."
        options = ["A_better", "B_better", "no_difference", "insufficient_info"]
        tag = "paradox" if sa * sg == -1 else "control" if sa == sg else "boundary"
        # Fixed before data: role, four count bins, and four rate bins.
        bins = [role] + [n[k] // 500 for k in n] + [int(r[k] * 5) for k in n]
        diagnostics = {"adjusted_difference": str(adjusted), "aggregate_difference": str(aggregate)}
    elif family == "collider":
        keys(p, "px p0 beta selection s00 s01 s10 s11 scale")
        px = number(p, "px", .1, .9)
        p0 = number(p, "p0", .1, .9)
        beta = number(p, "beta", -.7, .7)
        if not .05 <= p0 + beta <= .95:
            raise Rejected("outcome_probability")
        scale = number(p, "scale", 1000, 100000, True)
        mode = p['selection']
        if mode not in ("random", "on_X", "on_X_and_Y"):
            raise Rejected("selection")
        sel = {k: number(p, 's'+k, .05, 1) for k in ('00', '01', '10', '11')}
        if mode == 'random' and len(set(sel.values())) != 1:
            raise Rejected("random_selection_mismatch")
        if mode == 'on_X' and (sel['00'] != sel['01'] or sel['10'] != sel['11']):
            raise Rejected("on_x_selection_mismatch")
        cells = {}
        for x in (0, 1):
            prob_y = p0 + beta * x
            for y in (0, 1):
                k = str(x)+str(y)
                cells[k] = f"{scale * (px if x else 1-px) * (prob_y if y else 1-prob_y) * sel[k]:.2f}"
                if Decimal(cells[k]) < 20:
                    raise Rejected("cell_below_twenty")
        risk = [Decimal(cells[str(x)+'1']) / (Decimal(cells[str(x)+'0']) + Decimal(cells[str(x)+'1'])) for x in (0, 1)]
        difference = risk[1]-risk[0]
        sign = classify(difference)
        answer = 'insufficient_info' if mode == 'on_X_and_Y' else {1:'positive', -1:'negative', 0:'none'}[sign]
        selection = {'random': 'Selection is random and independent of X and Y.', 'on_X': 'Selection depends only on X, with positive inclusion probability in both X groups.', 'on_X_and_Y': 'Selection depends jointly on X and Y. Inclusion probabilities are unknown; no other population information is available.'}[mode]
        visible = {'cells': cells, 'selection': mode}
        prompt = header + "X and Y are binary. This question asks for association, not a causal effect. " + selection + "\nExact expected counts in the selected sample (no sampling noise):\n" + '\n'.join(f"X={k[0]}, Y={k[1]}: {v}" for k,v in cells.items()) + "\nWhat is the association between X and Y in the full population? A population risk difference below 0.01 in magnitude counts as none."
        options = ['positive', 'negative', 'none', 'insufficient_info']
        tag = mode
        bins = [mode, int(px*5), int(p0*5), math.floor(beta*5)] + [int(sel[k]*4) for k in sorted(sel)]
        diagnostics = {'population_difference': beta, 'selected_difference': str(difference), 'dag': 'X→Y; X→S←Y when jointly selected'}
    else:
        keys(p, "rho threshold delta design n")
        rho = number(p, 'rho', .1, .95)
        threshold = number(p, 'threshold', .5, 2.5)
        delta = number(p, 'delta', -1, 1)
        n = number(p, 'n', 20, 2000, True)
        design = p['design']
        if design not in ('no_control', 'randomized_control'):
            raise Rejected('design')
        # Baseline SD=1: U~N(0,rho), independent errors~N(0,1-rho).
        # Select baseline < -threshold; conditional expected U = rho * baseline.
        normal = NormalDist()
        baseline = -math.exp(-threshold**2/2) / math.sqrt(2*math.pi) / normal.cdf(-threshold)
        visible = {'n': n, 'design': design, 'treated_baseline': f'{baseline:.3f}', 'treated_followup': f'{rho*baseline+delta:.3f}'}
        if design == 'randomized_control':
            visible.update(control_baseline=f'{baseline:.3f}', control_followup=f'{rho*baseline:.3f}')
            effect = (Decimal(visible['treated_followup']) - Decimal(visible['treated_baseline'])) - (Decimal(visible['control_followup']) - Decimal(visible['control_baseline']))
            answer = {1:'improved', -1:'worsened', 0:'no_effect'}[classify(effect, True)]
        else:
            effect = None
            classify(delta, True)
            answer = 'insufficient_info'
        desc = 'Eligible units were randomly split into intervention and untreated control groups, with equal baseline distributions. No differential changes other than intervention occur.' if design == 'randomized_control' else 'All eligible units received the intervention. There is no control group. The reliability, latent-score distribution, untreated follow-up mean and natural trend are unknown.'
        prompt = header + "Higher outcome values are better. Units with low observed baseline scores were eligible. Baselines include measurement noise. " + desc + "\nExact expected means, in population baseline SD units (no sampling noise):\n" + '\n'.join(f'{k}: {v}' for k,v in visible.items() if k not in ('design',)) + '\nDid the intervention improve the outcome relative to no intervention? Effects below 0.05 SD in magnitude count as no effect.'
        options = ['improved', 'no_effect', 'worsened', 'insufficient_info']
        tag = design
        bins = [design, int(rho*5), int(threshold*2), math.floor(delta*5)]
        diagnostics = {'latent_effect': delta, 'visible_effect': str(effect) if effect is not None else None, 'scm': 'U~N(0,rho); independent errors~N(0,1-rho); B=U+e0; F=U+e1+delta*T; select B<-threshold'}
    return Item(digest(proposal), family, p, story['id'], family+':'+json.dumps(bins, separators=(',', ':')), tag, prompt, options, answer, visible, diagnostics)


class Gate:
    def __init__(self, stories=DEFAULT_STORIES):
        self.stories = stories
        self.seen = set()

    def accept(self, proposal):
        item = build(proposal, self.stories)
        key = (item.family, item.cell, item.story_id)
        if key in self.seen:
            raise Rejected('duplicate_cell_story')
        self.seen.add(key)
        return item


def neutral_tools(item):
    """Only direct visible summaries. Never adjusted effects or conclusions."""
    v = item.visible
    if item.family == 'simpson':
        n, s = v['n'], v['successes']
        return {'cell_rates': {k: float(Decimal(s[k])/n[k]) for k in n}, 'population_cell_proportions': {k:n[k]/sum(n.values()) for k in n}}
    if item.family == 'collider':
        c = {k:float(x) for k,x in v['cells'].items()}
        return {'selected_row_rates': {str(x): c[str(x)+'1']/(c[str(x)+'0']+c[str(x)+'1']) for x in (0,1)}, 'selected_cell_proportions': {k:x/sum(c.values()) for k,x in c.items()}}
    return {'treated_change': str(Decimal(v['treated_followup'])-Decimal(v['treated_baseline'])), **({'control_change': str(Decimal(v['control_followup'])-Decimal(v['control_baseline']))} if 'control_baseline' in v else {})}
