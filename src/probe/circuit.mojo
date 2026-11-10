"""F2 circuit pins (docs/prereg-f2.md): load and validate `tests/circuits/<name>.json`, and the verdict arithmetic of `nx circuit-test`.
Pin = {"name","model","prompts":[{"text","expect"}],"units":[{"layer","neuron"}],"effect":E0,"fraction":0.5}.
The engine measurement of E (ablate the units with `gain L:N:0`, mean drop of the expected token's logprob) lands with A4 and A3."""
from json import Json, parse
from mneuron.store import read_text


struct Pin(Movable):
    var name: String
    var model: String
    var effect: Float64
    var fraction: Float64
    var units: List[Tuple[Int, Int]]
    var expect: List[Int]
    var prompts: List[String]

    def __init__(out self):
        self.name = String()
        self.model = String()
        self.effect = 0
        self.fraction = 0.5
        self.units = List[Tuple[Int, Int]]()
        self.expect = List[Int]()
        self.prompts = List[String]()


def load_pin(path: String, vocab: Int) raises -> Pin:
    """Refuses a pin with no unit, more than 128 (F2b groups; F2 pinned at most 8), a prompt-less pin, an `expect` outside 0..vocab-1, or E0 <= 0 (prereg bar 3)."""
    var j = parse(read_text(path))
    var p = Pin()
    p.name = j.text("name")
    p.model = j.text("model")
    p.effect = j.get("effect").num()
    p.fraction = j.get("fraction").num() if j.has("fraction") else 0.5
    for u in j.get("units").items():
        p.units.append((u.get("layer").int(), u.get("neuron").int()))
    for q in j.get("prompts").items():
        p.prompts.append(q.text("text"))
        p.expect.append(q.get("expect").int())
    if len(p.units) < 1 or len(p.units) > 128:
        raise Error("circuit: " + path + " pins " + String(len(p.units)) + " units, want 1 to 128")
    if len(p.prompts) < 1:
        raise Error("circuit: " + path + " has no prompt")
    for t in p.expect:
        if t < 0 or t >= vocab:
            raise Error("circuit: expect token " + String(t) + " outside the vocabulary 0.." + String(vocab - 1))
    if p.effect <= 0:
        raise Error("circuit: pinned effect " + String(p.effect) + " is not positive (the circuit does nothing to ablate)")
    if p.fraction <= 0 or p.fraction > 1:
        raise Error("circuit: fraction " + String(p.fraction) + " outside (0, 1]")
    return p^


def passes(p: Pin, measured: Float64) -> Bool:
    """The circuit-test verdict: the re-measured effect keeps at least `fraction` of the pinned one."""
    return measured >= p.fraction * p.effect
