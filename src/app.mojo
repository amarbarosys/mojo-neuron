"""Neuron explorer window on mojo-imkit: map of layers, ranked list, neuron view (lens, connections, fires, recipe, steered chat).
Every number comes from the same calls as the CLI and the MCP server (mneuron/calls.mojo). Test hooks: MNEURON_SCRIPT=<file> drives the
window (click X Y, text S, wheel D, frame, shot PNG, see docs/app.md); MNEURON_DIR picks the store. Needs a GPU for the UI compose only."""
from std.builtin.sort import sort
from std.collections import Dict
from std.os import getenv
from std.os.path import exists
from std.os import remove
from std.sys import argv
from max.gpu.host import DeviceContext
from json import Json, parse
from imkit.font import Font
from imkit.layer import Layer
from imkit.draw import UH, UW
from imkit.theme import ACCENT, GOOD, BAD, TEXT, MUTED, SURFACE, HOVER, flat, glass, neon
from imkit.ui import Ui, rgba
from imkit.audio import Audio
from imkit.window import EV_BUTTON_DOWN, EV_BUTTON_UP, EV_KEY_DOWN, EV_KEY_UP, EV_MOTION, EV_TEXT, EV_WHEEL, Input, Window
from mneuron.calls import call
from mneuron.store import mdir, models, model_info, layers_with_lens, Lens, tokens, root
from mneuron.pngw import png_write_rgba
from mneuron.rlog import Rlog
from sense.scan import Scan, TOP

comptime V_MAP = 0
comptime V_LIST = 1
comptime V_NEURON = 2
comptime V_DIAL = 3
comptime DIAL_MAX = 8
comptime V_SCAN = 4
comptime SCAN_COLS = 20
comptime SCAN_WIN = 60
comptime BUCKETS = 40
comptime FW = Float32(UW)
comptime FH = Float32(UH)
comptime GREEN = GOOD  # theme roles meet WCAG contrast on the glass panels
comptime RED = BAD


def asc(s: String) -> String:
    """Printable text: control bytes dropped, UTF-8 kept (the font rasterizes any codepoint, a fallback font covers CJK)."""
    var out = List[UInt8]()
    for b in s.as_bytes():
        if b >= 32 and b != 127:
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def clip(s: String, n: Int) -> String:
    """At most n codepoints, the last one '~' when cut; never splits a multi-byte character."""
    var a = asc(s)
    var starts = List[Int]()
    for i in range(a.byte_length()):
        if Int(a.as_bytes()[i]) & 0xC0 != 0x80:
            starts.append(i)
    if len(starts) <= n:
        return a^
    return String(a[byte=0 : starts[n - 1]]) + "~"


def fmt(x: Float64, d: Int = 2) -> String:
    var neg = x < 0
    var v = -x if neg else x
    var scale = 1
    for _ in range(d):
        scale *= 10
    var r = Int(v * Float64(scale) + 0.5)
    var frac = String(r % scale)
    while frac.byte_length() < d:
        frac = "0" + frac
    return ("-" if neg else "") + String(r // scale) + "." + frac


def ramp(t: Float32) -> UInt32:
    var u = min(max(Float64(t), 0.0), 1.0)
    return rgba(Int(30 + 200 * u), Int(36 + 90 * u), Int(70 + 120 * (1 - u)), 255)


struct App(Movable):
    var win: Window
    var ui: Ui
    var layer: Layer
    var frame: List[UInt32]
    var running: Bool
    var scripted_down: Int
    var view: Int
    var mods: List[String]
    var mi: Int
    var info: Json
    var lens_layers: List[Int]
    var L: Int  # current layer
    var tk: List[String]  # vocabulary of the current model
    var ffn: Int
    var zmax: List[Float32]  # top logit-lens score per neuron of the layer
    var t3: List[Int32]  # first three top token ids per neuron
    var order: List[Int]  # neurons shown in the list, best first
    var label: String  # what the order is (lens score, token search, bucket)
    var page: Int
    var off: Float32  # scroll offset of the list table
    var sel: Int  # selected neuron
    var dos: Json
    var fr: Json
    var query: String
    var text: String
    var recipe: String
    var status: String
    var chat_sig: String
    var chat_prompt: String
    var live: Json
    var pending: String
    var maps: List[List[Float32]]  # per lens layer: BUCKETS max scores
    var map_ready: Bool
    var show_r: Bool
    var dial: List[Int]  # persona dial units: layer, neuron pairs
    var dk: List[Float32]  # slider dose per unit
    var sc: List[Scan]  # the capture being tailed (0 or 1)
    var sc_ch: Int  # selected channel, -1 none
    var sc_ser: List[Float32]  # that channel over the tokens
    var sc_tick: Int
    var rp: Int  # F10 scrubber: the token the scan view starts at and plays, -1 follows the tail
    var au: List[Audio]  # the SDL audio stream, opened by the first play
    var dial_lens: List[Json]  # the dossier lens of each unit
    var dscore: Dict[String, Float64]  # token -> sum of dose * lens score over the units

    def __init__(out self) raises:
        self.win = Window("mojo-neuron", UW, UH, UW, UH)
        self.ui = Ui(Font(getenv("IMKIT_FONT", "/usr/share/fonts/TTF/Roboto-Medium.ttf"), [13, 16, 22], getenv("IMKIT_FALLBACK", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")))
        var th = getenv("MNEURON_THEME", "glass")
        self.ui.set_theme(neon() if th == "neon" else (flat() if th == "flat" else glass()))
        self.layer = Layer(DeviceContext(), self.ui)
        self.frame = List[UInt32](length=UW * UH, fill=0xFF16120F)
        self.running = True
        self.scripted_down = 0
        self.view = V_MAP
        self.mods = models()
        self.mi = 0
        self.info = Json.object()
        self.lens_layers = List[Int]()
        self.L = 0
        self.tk = List[String]()
        self.ffn = 0
        self.zmax = List[Float32]()
        self.t3 = List[Int32]()
        self.order = List[Int]()
        self.label = ""
        self.page = 0
        self.off = 0
        self.sel = -1
        self.dos = Json.object()
        self.chat_sig = String()
        self.chat_prompt = String("Who are you?")
        self.live = Json.object()
        self.fr = Json.object()
        self.query = String("snake")
        self.text = String("The snake slithered away")
        self.recipe = String()
        self.status = "ready"
        self.pending = String()
        self.maps = List[List[Float32]]()
        self.map_ready = False
        self.show_r = False
        self.dial = List[Int]()
        self.dk = List[Float32]()
        self.dial_lens = List[Json]()
        self.au = List[Audio]()
        self.sc = List[Scan]()
        self.sc_ch = -1
        self.sc_ser = List[Float32]()
        self.sc_tick = 0
        self.rp = -1
        self.dscore = Dict[String, Float64]()
        if len(self.mods) == 0:
            self.status = "no model in " + root() + ": run `lens MODEL LAYER 8 --out DIR` first"
        else:
            self.pick_model(0)

    def pick_model(mut self, i: Int) raises:
        self.mi = i
        self.info = model_info(self.mods[i])
        self.lens_layers = layers_with_lens(self.mods[i])
        self.tk = tokens(self.mods[i])
        self.map_ready = False
        self.sel = -1
        self.recipe = String()
        if len(self.lens_layers) > 0:
            self.load_layer(self.lens_layers[0])

    def load_layer(mut self, L: Int) raises:
        var l = Lens(self.mods[self.mi], L)
        self.L = L
        self.ffn = l.ffn
        self.zmax = List[Float32](length=l.ffn, fill=0)
        self.t3 = List[Int32](length=l.ffn * 3, fill=-1)
        var keys = List[Int]()
        for n in range(l.ffn):
            self.zmax[n] = l.z(n, False, 0)
            for i in range(min(3, l.k)):
                self.t3[n * 3 + i] = Int32(l.id(n, False, i))
            keys.append(Int(max(self.zmax[n], 0.0) * 1000000.0) * 1048576 + (l.ffn - 1 - n))
        sort(keys)
        self.order = List[Int]()
        for j in range(len(keys) - 1, -1, -1):
            self.order.append(l.ffn - 1 - keys[j] % 1048576)
        self.label = "lens score (top logit-lens z)"
        self.off = 0

    def build_map(mut self) raises:
        self.maps = List[List[Float32]]()
        for L in self.lens_layers:
            var l = Lens(self.mods[self.mi], L)
            var b = List[Float32](length=BUCKETS, fill=0)
            for n in range(l.ffn):
                var k = n * BUCKETS // l.ffn
                b[k] = max(b[k], l.z(n, False, 0))
            self.maps.append(b^)
        self.map_ready = True

    def select(mut self, n: Int):
        self.sel = n
        self.view = V_NEURON
        self.live = Json.object()
        self.dos = Json.object()
        self.fr = Json.object()
        self.status = "loading dossier..."
        self.pending = "dossier"

    def args(self) -> Json:
        var a = Json.object()
        a.set("model", Json.of(self.mods[self.mi]))
        a.set("layer", Json.of(self.L))
        a.set("neuron", Json.of(self.sel))
        return a^

    def run_pending(mut self):
        var p = self.pending
        self.pending = String()
        if p.byte_length() == 0:
            return
        try:
            if p == "dossier":
                self.dos = call("dossier", self.args())
                self.status = "dossier written for L" + String(self.L) + " N" + String(self.sel)
            elif p == "live":
                var a = self.args()
                a.set("prompt", Json.of(self.chat_prompt))
                self.live = call("live", a)
                self.status = "live sweep done in " + fmt(self.live.get("wall_s").num(), 1) + " s"
            elif p == "fires":
                var a = self.args()
                a.set("text", Json.of(self.text))
                self.fr = call("fires", a)
                self.status = "fires: " + String(len(self.fr.get("tokens").items())) + " tokens"
            elif p == "search":
                var a = Json.object()
                a.set("model", Json.of(self.mods[self.mi]))
                a.set("layer", Json.of(self.L))
                a.set("token", Json.of(self.query))
                a.set("k", Json.of(100000))  # the call's default is 50; the list wants every hit
                var r = call("search_lens", a)
                self.order = List[Int]()
                for e in r.items():
                    self.order.append(e.get("neuron").int())
                self.label = "lens search '" + clip(self.query, 20) + "': " + String(len(self.order)) + " hits"
                self.off = 0
                self.status = self.label
            elif p == "rank":
                var a = Json.object()
                a.set("model", Json.of(self.mods[self.mi]))
                a.set("layer", Json.of(self.L))
                a.set("k", Json.of(500))
                var r = call("rank", a)
                self.order = List[Int]()
                for e in r.items():
                    self.order.append(e.get("neuron").int())
                self.label = "sweep rank (cluster mass vs controls)"
                self.off = 0
                self.status = self.label
            elif p == "brain":
                var a = self.args()
                a.set("sync", Json.of(1))
                var r = call("export_brain", a)
                self.status = "brain note: " + r.text("path")
            elif p == "cvec":
                var a = Json.object()
                a.set("model", Json.of(self.mods[self.mi]))
                a.set("recipe", Json.of(self.dial_recipe()))
                var r = call("cvec", a)
                self.status = "cvec ok: " + String(r.get("layers").int()) + " layers, " + r.text("mode") + " -> " + r.text("gguf")
            elif p == "sound":
                var a = Json.object()
                if self.view == V_SCAN and self.rp >= 0 and len(self.sc) > 0:  # F10: only the scrubbed frame, from the capture or replay dir
                    a.set("cap", Json.of(self.sc[0].dir))
                    a.set("out", Json.of(self.sc[0].dir + ".frame" + String(self.rp) + ".wav"))
                    a.set("start", Json.of(self.rp))
                    a.set("tokens", Json.of(1))
                else:
                    a.set("cap", Json.of(String(getenv("MNEURON_CAP", root() + "/../cap/" + self.mods[self.mi]))))
                    a.set("out", Json.of(mdir(self.mods[self.mi]) + "/sound.wav"))
                    a.set("tokens", Json.of(96))
                var r = call("sound", a)
                self.status = "sound: " + fmt(r.get("seconds").num(), 1) + " s, peak " + fmt(r.get("peak").num()) + " -> " + r.text("wav")
                if not self.win.scripted:
                    var b = open(r.text("wav"), "r").read_bytes()
                    if len(self.au) == 0:
                        self.au.append(Audio(24000))
                    var smp = List[Float32](length=(len(b) - 44) // 2, fill=0)
                    for i in range(len(smp)):
                        var q = Int(b[44 + 2 * i]) | (Int(b[45 + 2 * i]) << 8)
                        smp[i] = Float32(q - 65536 if q >= 32768 else q) / 32768.0
                    self.au[0].put(smp.unsafe_ptr(), len(smp))
            elif p == "image":
                var a = Json.object()
                var turn = Json.object()
                turn.set("text", Json.of(self.chat_prompt))
                var ws = Json.array()
                for tok in self.dial_ranked(True):
                    ws.push(Json.of(tok))
                turn.set("words", ws)
                var turns = Json.array()
                turns.push(turn^)
                a.set("turns", turns^)
                a.set("out", Json.of(mdir(self.mods[self.mi]) + "/images/" + String(self.sc_tick)))
                var r = call("image", a)
                self.status = "image in " + fmt(r.get("wall_s").num(), 1) + " s: " + r.get("images").at(0).str()
            elif p == "recipe":
                var a = Json.object()
                a.set("model", Json.of(self.mods[self.mi]))
                a.set("recipe", Json.of(self.recipe))
                self.status = "recipe ok: " + call("recipe", a).text("recipe")
        except e:
            self.status = "FAIL " + p + ": " + clip(String(e), 110)

    def row_text(self, n: Int) -> String:
        var s = String()
        for i in range(3):
            var id = Int(self.t3[n * 3 + i])
            if id >= 0:
                s += (" | " if i > 0 else "") + clip(self.tk[id], 12)
        return s^

    def header(mut self):
        self.ui.rect(0, 0, FW, 48, 0, rgba(17, 19, 26, 230))
        _ = self.ui.text(16, 12, "mojo-neuron", 1, TEXT)
        self.ui.x = 170
        self.ui.y = 9
        if self.ui.button(1, "map", 70, 30, self.view == V_MAP):
            self.view = V_MAP
            if not self.map_ready and len(self.lens_layers) > 0:
                try:
                    self.build_map()
                except e:
                    self.status = "FAIL map: " + clip(String(e), 100)
        if self.ui.button(2, "list", 70, 30, self.view == V_LIST):
            self.view = V_LIST
        if self.ui.button(3, "neuron", 90, 30, self.view == V_NEURON) and self.sel >= 0:
            self.view = V_NEURON
        if self.ui.button(7, "dial", 70, 30, self.view == V_DIAL):
            self.view = V_DIAL
        if self.ui.button(9, "scan", 70, 30, self.view == V_SCAN):
            self.view = V_SCAN
            if len(self.sc) == 0:
                self.scan_open()
        self.ui.x = 570
        if len(self.mods) > 0 and self.ui.button(4, clip(self.mods[self.mi], 30), 250, 30):
            try:
                self.pick_model((self.mi + 1) % len(self.mods))
            except e:
                self.status = "FAIL model: " + clip(String(e), 100)
        if len(self.lens_layers) > 0:
            if self.ui.button(5, "<", 30, 30):
                self.step_layer(-1)
            _ = self.ui.text(self.ui.x, 14, "layer " + String(self.L), 0, TEXT)
            self.ui.x += 72
            if self.ui.button(6, ">", 30, 30):
                self.step_layer(1)

    def step_layer(mut self, d: Int):
        var at = 0
        for i in range(len(self.lens_layers)):
            if self.lens_layers[i] == self.L:
                at = i
        var to = self.lens_layers[(at + d + len(self.lens_layers)) % len(self.lens_layers)]
        try:
            self.load_layer(to)
            self.sel = -1
            if self.view == V_NEURON:
                self.view = V_LIST
        except e:
            self.status = "FAIL layer: " + clip(String(e), 100)

    def draw_map(mut self):
        self.ui.panel(16, 60, FW - 16, 690, "map: layers x neuron buckets (colour = best lens score; click a cell to open that bucket)")
        var nl = Int(self.info.get("n_layer").int()) if self.info.has("n_layer") else 1
        var cw = Float32(UW - 80) / Float32(max(nl, 1))
        var x0 = Float32(40)
        var y0 = Float32(112)
        var mx = Float32(0.001)
        for b in self.maps:
            for v in b:
                mx = max(mx, v)
        for li in range(nl):
            var at = -1
            for i in range(len(self.lens_layers)):
                if self.lens_layers[i] == li:
                    at = i
            for k in range(BUCKETS):
                var xa = x0 + Float32(li) * cw
                var ya = y0 + Float32(k) * 13
                var c = ramp(self.maps[at][k] / mx) if at >= 0 and self.map_ready else rgba(28, 31, 40, 255)
                var hc = self.ui.clickable(5000 + li * 64 + k, xa, ya, xa + cw - 1, ya + 12)
                self.ui.rect(xa, ya, xa + cw - 1, ya + 12, 2, HOVER if hc[0] and at >= 0 else c)
                if hc[1] and at >= 0:
                    self.open_bucket(li, k)
            if li % 4 == 0:
                _ = self.ui.text(x0 + Float32(li) * cw, y0 + Float32(BUCKETS) * 13 + 4, String(li), 0, MUTED)
        if not self.map_ready:
            _ = self.ui.text(40, 70, "press map again after a lens exists (layers with a lens are coloured)", 0, MUTED)

    def open_bucket(mut self, li: Int, k: Int):
        try:
            self.load_layer(li)
        except e:
            self.status = "FAIL layer: " + clip(String(e), 100)
            return
        var picked = List[Int]()
        for n in self.order:
            if n * BUCKETS // self.ffn == k:
                picked.append(n)
        self.order = picked^
        self.label = "layer " + String(li) + " bucket " + String(k) + " of " + String(BUCKETS)
        self.view = V_LIST

    def draw_list(mut self):
        self.ui.panel(16, 60, 700, 690, "list: " + self.label)
        self.ui.x = 28
        self.ui.y = 98
        var r = self.ui.table_begin(2000, ["neuron", "score", "promotes"], [90.0, 80.0, 490.0], 580, len(self.order), 33, self.off)
        for j in range(r[0], r[1]):
            var n = self.order[j]
            if self.ui.table_row(2100 + j % 64, j):
                self.select(n)
            self.ui.table_cell(j, 0, "N" + String(n))
            self.ui.table_cell(j, 1, fmt(Float64(self.zmax[n])), GREEN)
            self.ui.table_cell(j, 2, self.row_text(n), MUTED)
        self.ui.table_end()
        self.ui.panel(716, 60, FW - 16, 690, "find")
        _ = self.ui.field(10, self.query, 24)
        if self.ui.button(11, "search token", 150):
            self.pending = "search"
            self.status = "searching..."
        if self.ui.button(12, "by lens", 100):
            try:
                self.load_layer(self.L)
            except e:
                self.status = "FAIL layer: " + clip(String(e), 100)
        if self.ui.button(13, "by rank", 100):
            self.pending = "rank"
            self.status = "ranking..."
        _ = self.ui.text(732, self.ui.y + 8, String(len(self.order)) + " neurons, scroll the table", 0, MUTED)

    def lens_lines(mut self, key: String, x: Float32, y: Float32, c: UInt32):
        var yy = y
        var which = String("r_lens") if self.show_r and self.dos.has("r_lens") else String("lens")
        for t in self.dos.get(which).get(key).items():
            _ = self.ui.text(x, yy, clip(t.at(0).str(), 14) + "  " + fmt(t.at(1).num()), 0, c)
            yy += 21

    def conn_rows(mut self, key: String, x: Float32, y: Float32, id0: Int):
        var yy = y
        var i = 0
        for c in self.dos.get("connections").get(key).items():
            if i >= 7:
                break
            var L2 = c.get("layer").int()
            var n2 = c.get("neuron").int()
            var hc = self.ui.clickable(id0 + i, x, yy, x + 170, yy + 22)
            self.ui.rect(x, yy, x + 170, yy + 22, 5, HOVER if hc[0] else SURFACE)
            _ = self.ui.text(x + 6, yy + 3, "L" + String(L2) + " N" + String(n2) + " " + c.text("role") + " " + fmt(c.get("cos").num()), 0, TEXT)
            if hc[1]:
                var has = False
                for l in self.lens_layers:
                    if l == L2:
                        has = True
                if has:
                    try:
                        self.load_layer(L2)
                        self.select(n2)
                    except e:
                        self.status = "FAIL jump: " + clip(String(e), 100)
                else:
                    self.status = "no lens file for layer " + String(L2) + ": run lens first"
            yy += 25
            i += 1

    def draw_neuron(mut self):
        self.ui.panel(16, 60, 330, 330, "lens  L" + String(self.L) + " N" + String(self.sel))
        if self.dos.has("r_lens"):
            self.ui.x = 200
            self.ui.y = 62
            if self.ui.button(40, "R lens" if self.show_r else "logit lens", 120, 26):
                self.show_r = not self.show_r
        if self.dos.has("lens"):
            _ = self.ui.text(28, 98, "promotes", 0, GREEN)
            self.lens_lines("top", 28, 120, TEXT)
            _ = self.ui.text(180, 98, "suppresses", 0, RED)
            self.lens_lines("bottom", 180, 120, TEXT)
        self.ui.panel(342, 60, 680, 330, "connections (weight cosine)")
        if self.dos.has("connections"):
            _ = self.ui.text(354, 96, "upstream", 0, MUTED)
            self.conn_rows("up", 354, 118, 3000)
            _ = self.ui.text(520, 96, "downstream", 0, MUTED)
            self.conn_rows("down", 520, 118, 3100)
        self.ui.panel(696, 60, FW - 16, 330, "fires on your text (CPU forward)")
        _ = self.ui.field(20, self.text, 90)
        self.ui.x = 712
        if self.ui.button(21, "run fires", 110):
            self.pending = "fires"
            self.status = "running forward..."
        if self.fr.has("tokens") and self.fr.has("activations"):
            var xx = Float32(712)
            var yy = Float32(190)
            var acts = self.fr.get("activations").items()
            var mx = Float64(0.001)
            for a in acts:
                mx = max(mx, abs(a.num()))
            var toks = self.fr.get("tokens").items()
            for i in range(len(toks)):
                var s = clip(toks[i].str(), 14)
                var w = self.ui.tw(s, 0) + 10
                if xx + w > Float32(UW - 28):
                    xx = 712
                    yy += 40
                var v = acts[i].num()
                self.ui.rect(xx, yy, xx + w, yy + 24, 5, rgba(40 + Int(180 * max(v, 0) / mx), 46, 70 + Int(150 * max(-v, 0) / mx), 255))
                _ = self.ui.text(xx + 5, yy + 4, s, 0, TEXT)
                _ = self.ui.text(xx, yy + 26, fmt(v, 1), 0, MUTED)
                xx += w + 4
            self.ui.x = 712
            self.ui.y = yy + 48
            if 330 - self.ui.y > 70:
                self.ui.plot([Float32(a.num()) for a in acts], -Float32(mx), Float32(mx), 520, 330 - self.ui.y - 8, ACCENT)
        self.ui.panel(16, 342, 330, 690, "recipe (steer: engine lane)")
        self.ui.x = 32
        if self.ui.button(30, "+1.0", 70):
            self.recipe += ("," if self.recipe.byte_length() > 0 else "") + String(self.L) + ":" + String(self.sel) + ":+1.0"
        if self.ui.button(31, "-1.0", 70):
            self.recipe += ("," if self.recipe.byte_length() > 0 else "") + String(self.L) + ":" + String(self.sel) + ":-1.0"
        if self.ui.button(32, "check", 70):
            self.pending = "recipe"
        if self.ui.button(33, "clear", 70):
            self.recipe = String()
        _ = self.ui.text(32, 392, clip(self.recipe, 40), 0, TEXT)
        _ = self.ui.text(32, 412, clip(String(self.recipe[byte=min(40, self.recipe.byte_length()) :]), 40), 0, TEXT)
        self.ui.x = 32
        self.ui.y = 450
        if self.ui.button(34, "brain note", 130):
            self.pending = "brain"
            self.status = "exporting to Brain..."
        if self.ui.button(35, "dossier", 100):
            self.pending = "dossier"
        self.ui.x = 32
        self.ui.y = 490
        if self.ui.button(36, "add to dial", 130):
            self.dial_add()
        self.ui.panel(342, 342, FW - 16, 690, "steered chat   - dose | no steering | + dose")
        self.draw_chat()

    def wrap(mut self, s: String, x: Float32, y: Float32, cols: Int, rows: Int, c: UInt32):
        var a = asc(s)
        var i = 0
        for r in range(rows):
            if i >= a.byte_length():
                break
            var j = min(i + cols, a.byte_length())
            _ = self.ui.text(x, y + Float32(r * 19), String(a[byte=i:j]), 0, c)
            i = j

    def draw_chat(mut self):
        var neg = String()
        var pos = String()
        var base = String()
        var prompt = String()
        if self.dos.has("steer"):
            var st = self.dos.get("steer")
            var rolls = st.get("rollouts").items()
            if len(rolls) > 0:
                prompt = rolls[0].text("prompt")  # one prompt for all three columns, so they compare
            for r in rolls:
                if r.text("prompt") != prompt:
                    continue
                var d = r.get("dose").num()
                if d < 0 and neg.byte_length() == 0:
                    neg = r.text("text")
                if d > 0 and pos.byte_length() == 0:
                    pos = r.text("text")
            for b in st.get("base").items():
                if b.text("prompt") == prompt and base.byte_length() == 0:
                    base = b.text("text")
        if self.live.has("base"):  # a live run for this neuron beats the file rows
            prompt = self.live.text("prompt")
            neg = self.live.text("neg")
            base = self.live.text("base")
            pos = self.live.text("pos")
        self.chat_sig = "neg:" + String(asc(neg).byte_length()) + ",base:" + String(asc(base).byte_length()) + ",pos:" + String(asc(pos).byte_length())
        if prompt.byte_length() > 0:
            _ = self.ui.text(354, 668, clip("prompt: " + asc(prompt), 120), 0, MUTED)
        self.ui.x = 354
        self.ui.y = 376
        _ = self.ui.field(40, self.chat_prompt, 70)
        self.ui.x = 354
        if self.ui.button(41, "generate (engine sweep, GPU)", 260):
            self.pending = "live"
            self.status = "running the engine sweep on the GPU..."
        var cx = Float32(354)
        for col in range(3):
            self.ui.rect(cx, 476, cx + 290, 662, 8, SURFACE)
            _ = self.ui.text(cx + 8, 482, ["- dose", "no steering", "+ dose"][col], 0, ACCENT)
            var t = [neg, base, pos][col]
            if t.byte_length() > 0:
                self.wrap(t, cx + 8, 506, 40, 8, TEXT)
            cx += 300
        if not self.dos.has("steer"):
            _ = self.ui.text(362, 520, "No sweep rows for this neuron: press generate (needs live.json,", 0, MUTED)
            _ = self.ui.text(362, 540, "see docs/app.md) or run the engine's sweep into the store.", 0, MUTED)

    def dial_recipe(self) -> String:
        var r = String()
        for i in range(len(self.dk)):
            r += ("," if i > 0 else "") + String(self.dial[2 * i]) + ":" + String(self.dial[2 * i + 1]) + ":" + ("+" if self.dk[i] >= 0 else "") + fmt(Float64(self.dk[i]))
        return r^

    def dial_add(mut self):
        if not self.dos.has("lens") or len(self.dk) >= DIAL_MAX:
            self.status = "dial: open a neuron with a lens first (max " + String(DIAL_MAX) + " units)"
            return
        self.dial.append(self.L)
        self.dial.append(self.sel)
        self.dk.append(0.25)
        self.dial_lens.append(self.dos.get("lens").copy())
        self.status = "dial: " + String(len(self.dk)) + " units"

    def dial_blend(mut self):
        """The live preview: token -> sum over units of dose * lens score (top scores are +, suppressed are -), cheap, no engine."""
        self.dscore = Dict[String, Float64]()
        for i in range(len(self.dk)):
            for key in ["top", "bottom"]:
                for t in self.dial_lens[i].get(key).items():
                    var tok = clip(t.at(0).str(), 14)
                    self.dscore[tok] = self.dscore.get(tok, 0.0) + Float64(self.dk[i]) * t.at(1).num()

    def dial_ranked(self, top: Bool) -> List[String]:
        var items = List[Tuple[Float64, String]]()
        for e in self.dscore.items():
            items.append((e.value if top else -e.value, e.key))
        sort(items)
        var out = List[String]()
        for j in range(len(items) - 1, max(len(items) - 9, -1), -1):
            out.append(items[j][1])
        return out^

    def draw_dial(mut self):
        self.ui.panel(16, 60, 700, 690, "persona dial: sliders blend steering neurons (add them from the neuron view)")
        var drop = -1
        for i in range(len(self.dk)):
            var ya = Float32(104 + i * 46)
            _ = self.ui.text(32, ya + 4, "L" + String(self.dial[2 * i]) + " N" + String(self.dial[2 * i + 1]), 0, TEXT)
            self.ui.x = 130
            self.ui.y = ya
            self.dk[i] = self.ui.slider(6000 + i, self.dk[i], -1.0, 1.0, 360)
            _ = self.ui.text(510, ya + 4, ("+" if self.dk[i] >= 0 else "") + fmt(Float64(self.dk[i])), 0, TEXT)
            self.ui.x = 600
            self.ui.y = ya - 3
            if self.ui.button(6100 + i, "x", 40, 30):
                drop = i
        if drop >= 0:
            _ = self.dial.pop(2 * drop + 1)
            _ = self.dial.pop(2 * drop)
            _ = self.dk.pop(drop)
            _ = self.dial_lens.pop(drop)
        self.dial_blend()
        self.ui.panel(716, 60, FW - 16, 690, "blend preview (lens, no engine)")
        var rows = ["promotes", "suppresses"]
        for c in range(2):
            var x = Float32(732 + c * 270)
            _ = self.ui.text(x, 100, rows[c], 0, GREEN if c == 0 else RED)
            var y = Float32(124)
            for tok in self.dial_ranked(c == 0):
                _ = self.ui.text(x, y, tok + "  " + fmt(self.dscore.get(tok, 0.0) * (1.0 if c == 0 else 1.0)), 0, TEXT)
                y += 21
        _ = self.ui.text(732, 330, clip("recipe " + self.dial_recipe(), 60), 0, MUTED)
        self.ui.x = 732
        self.ui.y = 360
        if self.ui.button(6200, "export cvec", 150) and len(self.dk) > 0:
            self.pending = "cvec"
            self.status = "writing the control vector..."
        if self.ui.button(6202, "play capture", 150):
            self.pending = "sound"
            self.status = "rendering the capture..."
        self.ui.x = 732
        self.ui.y = 400
        if self.ui.button(6203, "draw image", 130) and len(self.dial_ranked(True)) > 0:
            self.pending = "image"
            self.status = "rendering one picture (ComfyUI through gpu-wait)..."
        if self.ui.button(6201, "clear", 90):
            self.dial = List[Int]()
            self.dk = List[Float32]()
            self.dial_lens = List[Json]()

    def scan_open(mut self):
        var dir = String(getenv("MNEURON_CAP", root() + "/../cap/" + (self.mods[self.mi] if len(self.mods) > 0 else String("none"))))
        try:
            self.sc = List[Scan]()
            self.sc.append(Scan(dir))
            self.sc_ch = -1
            self.status = "scan: tailing " + dir
        except e:
            self.sc = List[Scan]()
            self.status = "FAIL scan: " + clip(String(e), 100) + " (set MNEURON_CAP to a capture dir)"

    def scan_poll(mut self):
        self.sc_tick += 1
        if len(self.sc) == 0 or self.sc_tick % 10 != 0:
            return
        try:
            var got = self.sc[0].poll()
            if got > 0:
                var nm = List[Float64]()
                for x in self.sc[0].norm:
                    nm.append(Float64(x))
                Rlog("scan", self.sc[0].dir, "").done(self.sc[0].sig(), self.sc[0].dir, nm, 0.0, 1e9)
                self.status = "scan: " + self.sc[0].sig()
        except e:
            self.status = "FAIL scan: " + clip(String(e), 110)

    def word(self, id: Int32) -> String:
        return clip(self.tk[Int(id)], 7) if Int(id) >= 0 and Int(id) < len(self.tk) else "#" + String(id)

    def draw_scan(mut self):
        var tag = String("no capture: press open")
        if len(self.sc) > 0:
            tag = "L" + String(self.sc[0].layer) + ", " + String(self.sc[0].n) + " tokens, top " + String(TOP) + " channels of the FFN output per token (red +, blue -)"
        self.ui.panel(16, 60, FW - 16, 450, "live scan: " + tag)
        self.ui.x = 1090
        self.ui.y = 62
        if self.ui.button(8000, "open", 80, 26):
            self.scan_open()
        if len(self.sc) == 0:
            return
        if self.sc[0].n > 1:  # F10 scrubber: drag to a token; the strip window starts there and "play" renders that frame
            self.ui.x = 500
            self.ui.y = 64
            var cur = self.rp if self.rp >= 0 else self.sc[0].n - 1  # moving the thumb leaves the tail
            var nk = Int(self.ui.slider(8200, Float32(cur), 0.0, Float32(self.sc[0].n - 1), 520) + 0.5)
            if nk != cur:
                self.rp = nk
            self.ui.x = 1090
            self.ui.y = 94
            if self.ui.button(8001, "play frame", 120, 26) and self.rp >= 0:
                self.pending = "sound"
        var first = max(0, self.sc[0].n - SCAN_WIN) if self.rp < 0 else self.rp
        var mx = Float32(1e-6)
        for i in range(first * TOP, self.sc[0].n * TOP):
            mx = max(mx, abs(self.sc[0].val[i]))
        for t in range(first, min(self.sc[0].n, first + SCAN_WIN)):
            var k = t - first
            var x0 = Float32(28 + (k % SCAN_COLS) * 60)
            var y0 = Float32(100 + (k // SCAN_COLS) * 134)
            _ = self.ui.text(x0, y0, self.word(self.sc[0].ids[t]), 0, TEXT)
            for j in range(TOP):
                var v = self.sc[0].val[t * TOP + j] / mx
                var ya = y0 + 22 + Float32(j * 17)
                var hc = self.ui.clickable(8100 + k * TOP + j, x0, ya, x0 + 54, ya + 15)
                self.ui.rect(x0, ya, x0 + 54, ya + 15, 3, HOVER if hc[0] else rgba(40 + Int(200 * max(v, 0)), 40, 40 + Int(200 * max(-v, 0)), 255))
                if hc[1]:
                    self.sc_ch = Int(self.sc[0].top[t * TOP + j])
                    try:
                        self.sc_ser = self.sc[0].series(self.sc_ch)
                    except e:
                        self.status = "FAIL scan: " + clip(String(e), 100)
        self.ui.panel(16, 520, FW - 16, 750, "channel dossier" + ("" if self.sc_ch < 0 else "  L" + String(self.sc[0].layer) + " c" + String(self.sc_ch) + " over the tokens"))
        if self.sc_ch >= 0 and len(self.sc_ser) > 0:
            var m = Float32(1e-6)
            for v in self.sc_ser:
                m = max(m, abs(v))
            self.ui.x = 32
            self.ui.y = 560
            self.ui.plot(self.sc_ser, -m, m, 760, 170, ACCENT)
            _ = self.ui.text(830, 560, "strongest tokens", 0, MUTED)
            var used = List[Bool](length=len(self.sc_ser), fill=False)
            for r in range(min(6, len(self.sc_ser))):
                var bt = -1
                for t in range(len(self.sc_ser)):
                    if not used[t] and (bt < 0 or abs(self.sc_ser[t]) > abs(self.sc_ser[bt])):
                        bt = t
                used[bt] = True
                _ = self.ui.text(830, 584 + Float32(r * 22), "#" + String(bt) + " " + self.word(self.sc[0].ids[bt]) + "  " + fmt(Float64(self.sc_ser[bt])), 0, TEXT)

    def frame_once(mut self) raises:
        var inp = self.win.poll()
        self.ui.begin(inp)
        if inp.quit:
            self.running = False
            return
        if self.view == V_MAP:
            self.draw_map()
        elif self.view == V_LIST:
            self.draw_list()
        elif self.view == V_DIAL:
            self.draw_dial()
        elif self.view == V_SCAN:
            self.draw_scan()
        else:
            self.draw_neuron()
        self.header()
        _ = self.ui.text(16, FH - 24, clip(self.status, 150), 0, MUTED if self.status.find("FAIL") < 0 else RED)
        self.ui.finish()
        self.layer.render(self.ui)
        self.win.present(Int(self.frame.unsafe_ptr()), Int(self.layer.pixels.unsafe_ptr()))
        self.run_pending()
        self.scan_poll()

    def shot_bytes(self) -> List[UInt8]:
        var out = List[UInt8](length=UW * UH * 4, fill=0)
        var p = self.layer.pixels.unsafe_ptr().unsafe_bitcast[UInt8]()
        var bg = self.frame.unsafe_ptr().unsafe_bitcast[UInt8]()
        for px in range(UW * UH):  # the UI layer blended over the scene, as the window shows it
            var a = Int(p[unsafe_offset=px * 4 + 3])
            for c in range(3):
                out[px * 4 + c] = UInt8((Int(p[unsafe_offset=px * 4 + c]) * a + Int(bg[unsafe_offset=px * 4 + c]) * (255 - a)) // 255)
            out[px * 4 + 3] = 255
        return out^

    def event(mut self, line: String) raises:
        """One script line: frame, click X Y, text S, wheel D, shot PNG, wait N (frames)."""
        var w = line.split(" ")
        var cmd = String(w[0])
        if cmd == "frame":
            self.frame_once()
        elif cmd == "wait":
            for _ in range(Int(String(w[1]))):
                self.frame_once()
        elif cmd == "text":
            self.win.push(EV_TEXT, text=String(line[byte=5:]))
        elif cmd == "wheel":
            self.win.push(EV_WHEEL, 0, Float32(Int(String(w[1]))))
        elif cmd == "click":
            var x = Float32(Int(String(w[1])))
            var y = Float32(Int(String(w[2])))
            self.win.push(EV_MOTION, 0, x, y, x - self.win.x, y - self.win.y)
            self.frame_once()
            self.win.push(EV_BUTTON_DOWN, 1, x, y)
            self.frame_once()
            self.win.push(EV_BUTTON_UP, 1, x, y)
            self.frame_once()
            self.frame_once()
        elif cmd == "scrub":  # scrub K: the F10 scrubber at token K
            self.rp = Int(String(w[1]))
            self.frame_once()
        elif cmd == "play":  # the scrubbed frame as a WAV (and audio when not scripted)
            self.pending = "sound"
            self.frame_once()
        elif cmd == "state":  # state FILE: write the STATE line now (a gate reads it while the script is still running)
            with open(String(w[1]), "w") as f:
                f.write(self.state_line())
        elif cmd == "waitfile":  # frames until FILE exists (at most 900)
            for _ in range(900):
                if exists(String(w[1])):
                    break
                self.frame_once()
        elif cmd == "shot":
            var path = String(w[1])
            if exists(path):
                remove(path)
            png_write_rgba(path, self.shot_bytes(), UW, UH)
        else:
            raise Error("script: unknown command '" + line + "'")

    def report(self):
        print(self.state_line())

    def rp_sig(self) -> String:
        """k:token:channel=value,... of the scrubbed token's strip, exactly the stored values."""
        if self.rp < 0 or len(self.sc) == 0 or self.rp >= self.sc[0].n:
            return "-"
        var t = String(self.rp) + ":" + String(self.sc[0].ids[self.rp]) + ":"
        for j in range(TOP):
            t += String(self.sc[0].top[self.rp * TOP + j]) + "=" + String(Float64(self.sc[0].val[self.rp * TOP + j])) + ","
        return t

    def state_line(self) -> String:
        return String("STATE view=" + String(self.view) + " layer=" + String(self.L) + " sel=" + String(self.sel) + " order=" + String(len(self.order))
              + " recipe=" + self.recipe + " scan=" + (self.sc[0].sig() if len(self.sc) > 0 else String("-")) + " ch=" + String(self.sc_ch) + " rp=" + self.rp_sig() + " dial=" + self.dial_recipe() + " prev=" + (self.dial_ranked(True)[0] if len(self.dscore) > 0 else String("-")) + " chat=" + self.chat_sig + " dropped=" + String(self.ui.d.dropped) + " status=" + self.status)


def main() raises:
    if getenv("MNEURON_CLIPTEST", "") != "":  # gate: clip counts codepoints and never splits one
        print("CLIP " + clip("日本語のテキスト", 4) + " " + clip("abc", 4) + " " + clip("é€😀xyz", 4))
        return
    var app = App()
    var script = getenv("MNEURON_SCRIPT", "")
    if script.byte_length() > 0:
        app.win.scripted = True
        for line in open(script, "r").read().split("\n"):
            var l = String(line)
            if l.byte_length() == 0 or l.startswith("#"):
                continue
            app.event(l)
            if not app.running:
                break
    else:
        while app.running:
            app.frame_once()
    app.report()
    app.win.close()
