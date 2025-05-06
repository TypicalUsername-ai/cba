from multiprocessing import Pool

import numpy as np


def cba(
    inputs: [any, 2],
    classes: [any],
    min_support: float = 0.01,
    min_confidence: float = 0.5,
    prune: bool = False,
):
    pool = Pool(6)
    in_lookup, token_ins = tokenize(inputs)
    cl_lookup, token_cls = tokenize_classes(classes)
    print(in_lookup, sep="\n")
    print(cl_lookup)
    rules = cba_rg(token_ins, cl_lookup, min_support, min_confidence, prune, pool)
    print(f"total of {len(rules)} generated")
    classifier, correct, error = cba_cb_m1(
        rules, token_ins.copy(), token_cls.copy(), pool
    )
    print(
        f"success rate {correct / len(token_cls) * 100:.2f}% | errors rate {error / len(token_cls) * 100:.2f}% | items {len(token_cls)}"
    )
    return classifier


def tokenize(ins: [str, 2]) -> ([dict[int, str]], [int, 2]):
    tokenized_cols = []
    col_dicts = []
    for column in np.transpose(ins):
        core = list(enumerate(set(column)))
        print(core)
        rev_lookup = dict((i[1], i[0]) for i in core)
        col_dicts.append(dict(core))
        tokenized_cols.append([rev_lookup.get(i) for i in column])

    return (col_dicts, np.transpose(tokenized_cols))


def tokenize_classes(classes: [str]) -> (dict[int, str], [int]):
    core = list(enumerate(set(classes)))
    print(core)
    rev_lookup = dict((i[1], i[0]) for i in core)
    lookup = dict(core)
    tokenized = [rev_lookup.get(i) for i in classes]
    return (lookup, tokenized)


class Rule:
    def __init__(
        self,
        rule: [tuple[int, int]],
        prediction: str,
        rule_support: int = 1,
        pred_support: int = 1,
    ):
        self.rule = frozenset(rule)
        self.prediction = prediction
        self.rule_support = rule_support
        self.pred_support = pred_support

    def support(self) -> int:
        return self.rule_support

    def confidence(self) -> float:
        return self.pred_support / self.rule_support

    def inputs_match(self, inputs: [int]) -> bool:
        inputset = set([i for i in enumerate(inputs)])
        return len(self.rule.intersection(inputset)) == len(self.rule)

    def __str__(self):
        return f"{self.rule} -> {self.prediction} (supp: {self.support()}|{self.pred_support}, conf: {self.confidence() * 100:.2f}%)"

    def __repr__(self):
        return self.__str__()


def cba_rg(
    inputs: [int, 2],
    classes: [int],
    min_support: float,
    min_confidence: float,
    prune: bool,
    threadpool: Pool,
):
    pool = threadpool
    total_items = len(classes)
    f = [[]]  # large k-ruleitems where occurence >= minsup
    for i, col in enumerate(np.transpose(inputs)):
        # print(set(col))
        # create a dict for
        # values -> classes -> counts
        pbs = dict(
            [value, dict((out, 0) for out in set(classes))] for value in set(col)
        )
        for entry, cl in zip(col, classes):
            pbs[entry][cl] += 1
        for r, c in pbs.items():
            for p, s in c.items():
                # check if support checks out
                if s / total_items >= min_support:
                    f[0].append(Rule([(i, r)], p, s, s))
    print("[0]", len(f[0]), "1-ruleitems where support >=", min_support)
    car = [generate_rules(f[0], min_confidence)]
    print("[0]", len(car[0]), "1-rule CARs with confidence >=", min_confidence)
    if prune:
        pr_car = [prune_rules(car)]

    k = 1
    while len(f[k - 1]) != 0 and k <= len(inputs[0]):
        candidates = generate_candidates(f[k - 1], inputs, pool)
        print(f"[{k}] {len(candidates)} {k + 1}-ruleitem candidates")
        ## multiprocessing
        candidates_item = pool.starmap(
            rule_subset, map(lambda i: (candidates, i), inputs)
        )

        for candidate_cases, cls in zip(candidates_item, classes):
            # candidate_cases = rule_subset(candidates, entry[0], pool)
            for candidate in candidate_cases:
                candidate.rule_support += 1
                candidate.pred_support += 1 if cls == candidate.prediction else 0
        f.append(list(filter(lambda c: c.support() >= min_support, candidates)))
        print(f"[{k}] {len(f[k])} {k + 1}-ruleitems where support >= {min_support}")
        car.append(generate_rules(f[k], min_confidence))
        print(
            f"[{k}] {len(car[k])} {k + 1}-rule CARs with confidence >= {min_confidence}"
        )
        if prune:
            pr_car.append(prune_rules(f[k]))
        k += 1

    if prune:
        return [j for sub in pr_car for j in sub]
    else:
        return [j for sub in car for j in sub]


def generate_rules(rules: [Rule], min_confidence: float) -> [Rule]:
    new_rules = []
    unique_rules = set([r.rule for r in rules])
    for ur in unique_rules:
        filtered = list(filter(lambda r: r.rule == ur, rules))
        best = max(filtered, key=Rule.support)
        total_support = sum(r.rule_support for r in filtered)

        nr = Rule(ur, best.prediction, total_support, best.pred_support)
        new_rules.append(nr)
    print(f"total {len(new_rules)} new rules")
    return list(filter(lambda r: r.confidence() >= min_confidence, new_rules))


def possible_extensions(rule: Rule, inputs: [int, 2]):
    return set(
        i for e in filter(rule.inputs_match, inputs) for i in enumerate(e)
    ).difference(rule.rule)


def generate_candidates(rules: [Rule], inputs: [int, 2], pool: Pool) -> [Rule]:
    candidate_rules = []
    available_exts = [
        *pool.starmap(possible_extensions, map(lambda r: (r, inputs), rules))
    ]
    for rule, exts in zip(rules, available_exts):
        for extension in exts:
            candidate_rules.append(Rule(rule.rule.union([extension]), rule.prediction))
    return candidate_rules


def rule_subset(rules: [Rule], data: [int]):
    # data = [*pool.starmap(Rule.inputs_match, map(lambda x: (x, data), rules))]
    # subset = map(lambda i: i[1], filter(lambda x: x[0], zip(data, rules)))
    subset = []
    for rule in rules:
        if rule.inputs_match(data):
            # print(rule.rule, "-", data, "=>", match)
            subset.append(rule)
    return subset


def cba_cb_m1(rules: [Rule], inputs: [int, 2], classes: [int], pool: Pool):
    rules.sort(reverse=True, key=lambda x: len(x.rule))
    rules.sort(reverse=True, key=Rule.support)
    rules.sort(reverse=True, key=Rule.confidence)
    classifier = []
    correct = 0
    error = 0
    for rule in rules:
        r_correct = 0
        r_err = 0
        temp = []
        for idx, (entry, cl) in enumerate(zip(inputs, classes)):
            if rule.inputs_match(entry):
                temp.append(idx)
                if rule.prediction == cl:
                    r_correct += 1
                else:
                    r_err += 1
        if r_correct >= 1:
            temp.reverse()
            correct += r_correct
            error += r_err
            for idx in temp:
                np.delete(inputs, idx, 0)
                classes.pop(idx)
                classifier.append(rule)
    return classifier, correct, error
