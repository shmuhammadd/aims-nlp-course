def mrr(ranks):
    return float(np.mean([0.0 if r is None else 1 / r for r in ranks])) if len(ranks) else 0.0


def validate_call(call, schema):
    if not isinstance(call, dict) or set(call) != {"tool", "args"}:
        return False
    spec = schema.get(call["tool"]) if isinstance(call["tool"], str) else None
    args = call["args"]
    if spec is None or not isinstance(args, dict) or set(args) != set(spec):
        return False
    # type(), not isinstance(): bool is a subclass of int and must not pass as a number
    return all(type(args[k]) in allowed and not (type(args[k]) is float and not math.isfinite(args[k]))
               for k, allowed in spec.items())
