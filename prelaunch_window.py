"""The window before the game (the user, 2026-11-14: "a small Windows window at launch", the page "not explicit
enough"): three numbered steps read top to bottom, 1 where to play (a town, or a route from A to B) with the estimate
and Prepare, 2 the places ready for a teleport (progress, delete) and the cache's use, 3 Start the game.

tkinter (in Python itself: nothing to install). Every request to the world server runs on a thread of its own
(prelaunch.Bridge blocks); the answers come back to Tk's thread through a queue read every POLL_MS. The map page
(prelaunch.PAGE) stays one button away, for a box drawn by hand. Closing the window starts the game too: the
launcher never waits on a window nobody sees.
"""
import base64
import math
import queue
import threading
import webbrowser

import tkinter as tk
from tkinter import messagebox, ttk

import prelaunch

POLL_MS = 100
STATUS_MS = 2000
TOWN_KM = 4.0               # a town's area: a square this wide around its point
ROUTE_HALF_M = 300          # a route's area: this far either side
STATE = {"queued": "en attente", "running": "en cours", "done": "prêt", "paused": "en pause",
         "waiting": "en attente (quota du jour)", "failed": "en échec"}
FONT, SMALL, BOLD = ("Segoe UI", 10), ("Segoe UI", 9), ("Segoe UI Semibold", 10)
TITLE, BIG, BRAND = ("Segoe UI Semibold", 12), ("Segoe UI Semibold", 13), ("Segoe UI Black", 15)
BG, HEAD, PANEL, FIELD, LINE = "#121418", "#0b0c0f", "#1c1f25", "#14161a", "#2c3038"
TEXT, MUTED, OK, WARN = "#e8e9ec", "#9aa0aa", "#3fb27f", "#e0a63a"
BTN, BTN_HOVER = "#2a2e36", "#353a44"
ACCENT, ACCENT_HOVER, ACCENT_DARK = "#d7263d", "#e6404f", "#a81c2f"     # (Zyphro's red band)
WRAP = 330                  # px: the left column's text width
THUMB_W, THUMB_H = 260, 190  # px: the miniature
THUMB_TILES_MAX = 16        # tiles fetched for one miniature at most


def mb(b):
    b = float(b or 0)
    return f"{b / 1e9:.1f} Go".replace(".", ",") if b >= 1e9 else f"{max(1, round(b / 1e6))} Mo"


def duration(s):
    s = float(s or 0)
    if s < 90:
        return f"{round(s)} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h".replace(".", ",")


def box_around(lat, lon, km):
    import math
    dl = km / 2 / 111.32
    dn = dl / max(0.05, math.cos(math.radians(lat)))
    return {"s": lat - dl, "n": lat + dl, "w": lon - dn, "e": lon + dn}


def _merc(lat, lon, z):
    """Web-Mercator pixel (x, y) at zoom z (256 px tiles)."""
    lat = max(-85.0, min(85.0, lat))
    n = 256.0 * (1 << z)
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n


def render_thumb(tile, s, w, n, e, line=None, size=(THUMB_W, THUMB_H)):
    """A miniature of the box s, w, n, e on the satellite tiles (tile(layer, z, x, y) -> bytes or None): the box in
    red, or the route's points (line: [[lat, lon]...]) in red. PNG bytes, or None without Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    import io
    W, H = size
    z = 17
    while z > 2:
        x0, y0 = _merc(n, w, z)
        x1, y1 = _merc(s, e, z)
        if (x1 - x0) * 1.3 <= W and (y1 - y0) * 1.3 <= H:
            break
        z -= 1
    cx, cy = _merc((s + n) / 2, (w + e) / 2, z)
    left, top = cx - W / 2, cy - H / 2
    tx0, ty0, tx1, ty1 = int(left // 256), int(top // 256), int((left + W) // 256), int((top + H) // 256)
    keys = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)][:THUMB_TILES_MAX]
    got = {}

    def fetch(k):
        x, y = k
        if 0 <= y < (1 << z):
            got[k] = tile("sat", z, x % (1 << z), y)
    threads = [threading.Thread(target=fetch, args=(k,), daemon=True) for k in keys]   # (one map_tiles request)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    img = Image.new("RGB", (W, H), FIELD)
    for (x, y), data in got.items():
        if not data:
            continue
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        img.paste(im, (int(x * 256 - left), int(y * 256 - top)))
    d = ImageDraw.Draw(img)

    def px(lat, lon):
        x, y = _merc(lat, lon, z)
        return x - left, y - top
    if line:
        pts = [px(p[0], p[1]) for p in line]
        d.line(pts, fill=(0, 0, 0), width=6, joint="curve")
        d.line(pts, fill=ACCENT, width=3, joint="curve")
    else:
        (ax, ay), (bx, by) = px(n, w), px(s, e)
        d.rectangle((ax, ay, bx, by), outline=(0, 0, 0), width=4)
        d.rectangle((ax, ay, bx, by), outline=ACCENT, width=2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def bounds_of(points):
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    return min(lats), min(lons), max(lats), max(lons)


class Window:
    def __init__(self, bridge, launch, user_dir, url):
        self.bridge, self.launch, self.user_dir, self.url = bridge, launch, user_dir, url
        self.q = queue.Queue()
        self.pick = None            # the place being prepared: {name, start, box or route, mode}
        self.results = []
        self._shown = None
        self.root = tk.Tk()
        self.root.title("BeamNG.Earth – Avant de jouer")
        self.root.minsize(680, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.start_game)
        self._theme()
        self.thumb_img = None
        self.thumb_for = None
        self._build()

    # ---- look ---------------------------------------------------------------------------------------
    def _theme(self):
        """A dark theme (Zyphro's colours: near black, red for what to press, green for what is ready) on ttk's
        "clam", the one theme whose colours can all be set."""
        root = self.root
        root.configure(bg=BG)
        st = ttk.Style(root)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=TEXT, font=FONT, bordercolor=LINE, lightcolor=BG, darkcolor=BG,
                     troughcolor=PANEL, focuscolor=ACCENT, fieldbackground=FIELD, insertcolor=TEXT,
                     selectbackground=ACCENT, selectforeground="#fff")
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=PANEL, bordercolor=LINE, relief="solid", borderwidth=1)
        st.configure("Head.TFrame", background=HEAD)
        for name, bg in (("TLabel", BG), ("Card.TLabel", PANEL)):
            st.configure(name, background=bg, foreground=TEXT)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=SMALL)
        st.configure("Step.TLabel", background=PANEL, foreground=TEXT, font=TITLE)
        st.configure("Badge.TLabel", background=ACCENT, foreground="#fff", font=TITLE, anchor="center", width=2)
        st.configure("Brand.TLabel", background=HEAD, foreground="#fff", font=BRAND)
        st.configure("Sub.TLabel", background=HEAD, foreground=MUTED, font=FONT)
        st.configure("Ready.TLabel", background=PANEL, foreground=OK, font=SMALL)
        st.configure("Run.TLabel", background=PANEL, foreground=WARN, font=SMALL)
        st.configure("TButton", background=BTN, foreground=TEXT, bordercolor=LINE, padding=(10, 5), relief="flat")
        st.map("TButton", background=[("disabled", PANEL), ("pressed", LINE), ("active", BTN_HOVER)],
               foreground=[("disabled", "#5d626c")])
        st.configure("Accent.TButton", background=ACCENT, foreground="#fff", bordercolor=ACCENT, font=BOLD)
        st.map("Accent.TButton", background=[("disabled", PANEL), ("pressed", ACCENT_DARK), ("active", ACCENT_HOVER)],
               foreground=[("disabled", "#5d626c")], bordercolor=[("disabled", LINE)])
        st.configure("Big.TButton", background=ACCENT, foreground="#fff", bordercolor=ACCENT, font=BIG, padding=12)
        st.map("Big.TButton", background=[("pressed", ACCENT_DARK), ("active", ACCENT_HOVER)])
        st.configure("Small.TButton", padding=(8, 2), font=SMALL)
        st.configure("TRadiobutton", background=PANEL, foreground=TEXT, indicatorcolor=FIELD)
        st.map("TRadiobutton", indicatorcolor=[("selected", ACCENT)], background=[("active", PANEL)])
        st.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, bordercolor=LINE, padding=6)
        st.map("TEntry", bordercolor=[("focus", ACCENT)])
        st.configure("Horizontal.TProgressbar", background=OK, troughcolor=FIELD, bordercolor=LINE, thickness=8)
        st.configure("Run.Horizontal.TProgressbar", background=WARN, troughcolor=FIELD, bordercolor=LINE, thickness=8)

    def _step(self, parent, row, n, title):
        """A numbered card: its badge and title, and the frame below them for its content."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
        card.columnconfigure(0, weight=1)
        head = ttk.Frame(card, style="Card.TFrame", borderwidth=0)
        head.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(head, text=str(n), style="Badge.TLabel").pack(side="left", ipady=1)
        ttk.Label(head, text=title, style="Step.TLabel").pack(side="left", padx=(10, 0))
        body = ttk.Frame(card, style="Card.TFrame", borderwidth=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        return card, body

    # ---- layout ---------------------------------------------------------------------------------
    def _build(self):
        head = ttk.Frame(self.root, style="Head.TFrame", padding=(18, 12))
        head.pack(fill="x")
        ttk.Label(head, text="BeamNG.Earth", style="Brand.TLabel").pack(side="left")
        ttk.Label(head, text="  Avant de jouer : préparez vos lieux pour des téléports rapides",
                  style="Sub.TLabel").pack(side="left", pady=(4, 0))

        r = ttk.Frame(self.root, padding=(14, 12, 14, 14))
        r.pack(fill="both", expand=True)
        r.columnconfigure(0, weight=1)

        _c1, s1 = self._step(r, 0, 1, "Où voulez-vous jouer ?")
        s1.columnconfigure(1, weight=0)
        left = ttk.Frame(s1, style="Card.TFrame", borderwidth=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(s1, style="Card.TFrame", borderwidth=0)
        right.grid(row=0, column=1, sticky="n", padx=(14, 0))

        self.kind = tk.StringVar(value="town")
        kinds = ttk.Frame(left, style="Card.TFrame", borderwidth=0)
        kinds.grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(kinds, text="Une ville", value="town", variable=self.kind, command=self._kind).pack(side="left")
        ttk.Radiobutton(kinds, text="Un trajet de A à B", value="route", variable=self.kind,
                        command=self._kind).pack(side="left", padx=(14, 0))

        self.town = ttk.Frame(left, style="Card.TFrame", borderwidth=0)
        self.town.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.town.columnconfigure(0, weight=1)
        self.q_town = ttk.Entry(self.town, font=FONT)
        self.q_town.grid(row=0, column=0, sticky="ew")
        self.q_town.bind("<Return>", lambda e: self.search())
        ttk.Button(self.town, text="Chercher", command=self.search).grid(row=0, column=1, padx=(6, 0))
        self.res = tk.Listbox(self.town, height=4, font=FONT, activestyle="none", exportselection=False,
                              bg=FIELD, fg=TEXT, selectbackground=ACCENT, selectforeground="#fff",
                              highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT, bd=0,
                              relief="flat")
        self.res.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.res.bind("<<ListboxSelect>>", lambda e: self.choose())

        self.route = ttk.Frame(left, style="Card.TFrame", borderwidth=0)
        self.route.columnconfigure(1, weight=1)
        ttk.Label(self.route, text="Départ", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.q_a = ttk.Entry(self.route, font=FONT)
        self.q_a.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(self.route, text="Arrivée", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.q_b = ttk.Entry(self.route, font=FONT)
        self.q_b.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.q_b.bind("<Return>", lambda e: self.trace())
        ttk.Button(self.route, text="Tracer le trajet", command=self.trace).grid(row=2, column=1, sticky="e",
                                                                                 pady=(8, 0))

        self.est = ttk.Label(left, text="Tapez une ville puis Chercher.", style="Muted.TLabel", wraplength=WRAP,
                             justify="left")
        self.est.grid(row=3, column=0, sticky="w", pady=(10, 8))
        acts = ttk.Frame(left, style="Card.TFrame", borderwidth=0)
        acts.grid(row=4, column=0, sticky="w")
        self.b_now = ttk.Button(acts, text="Préparer maintenant", style="Accent.TButton",
                                command=lambda: self.prepare(True), state="disabled")
        self.b_now.pack(side="left")
        self.b_bg = ttk.Button(acts, text="Pendant le jeu", command=lambda: self.prepare(False), state="disabled")
        self.b_bg.pack(side="left", padx=(6, 0))
        ttk.Label(left, text="« Maintenant » : tout le processeur, avant de jouer. « Pendant le jeu » : doucement, "
                             "sans gêner.", style="Muted.TLabel", wraplength=WRAP,
                  justify="left").grid(row=5, column=0, sticky="w", pady=(6, 0))

        # the miniature: the place picked (or a prepared one clicked in step 2) on the satellite view
        self._blank = tk.PhotoImage(width=THUMB_W, height=THUMB_H)
        self.thumb = tk.Label(right, image=self._blank, bg=FIELD, fg=MUTED, cursor="hand2",
                              text="Aperçu du lieu", font=SMALL, compound="center", bd=0,
                              highlightthickness=1, highlightbackground=LINE)
        self.thumb.pack()
        self.thumb.bind("<Button-1>", lambda e: webbrowser.open(self.url))
        self.thumb_cap = ttk.Label(right, text="", style="Muted.TLabel", wraplength=THUMB_W)
        self.thumb_cap.pack(anchor="w", pady=(4, 0))
        ttk.Button(right, text="Voir la carte / encadrer une zone…",
                   command=lambda: webbrowser.open(self.url)).pack(fill="x", pady=(6, 0))

        c2, s2 = self._step(r, 1, 2, "Lieux prêts pour le téléport")
        r.rowconfigure(1, weight=1)
        self.list = ttk.Frame(s2, style="Card.TFrame", borderwidth=0)
        self.list.grid(row=0, column=0, sticky="ew")
        self.list.columnconfigure(1, weight=1)
        self.cache = ttk.Label(s2, text="Cache : lecture…", style="Muted.TLabel")
        self.cache.grid(row=1, column=0, sticky="w", pady=(10, 0))

        _c3, s3 = self._step(r, 2, 3, "Jouer")
        ttk.Button(s3, text="LANCER LE JEU", style="Big.TButton", command=self.start_game).grid(row=0, column=0,
                                                                                                sticky="ew")
        ttk.Label(s3, text="Vos lieux préparés sont dans le menu de la session libre (lieu de départ) et dans la "
                           "liste « Places » du jeu. La préparation continue pendant que vous jouez.",
                  style="Muted.TLabel", wraplength=WRAP + THUMB_W, justify="left").grid(row=1, column=0, sticky="w",
                                                                                       pady=(8, 0))
        self.q_town.focus_set()

    def _kind(self):
        if self.kind.get() == "town":
            self.route.grid_forget()
            self.town.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            self.q_town.focus_set()
        else:
            self.town.grid_forget()
            self.route.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            self.q_a.focus_set()
        self._set_pick(None, "Tapez une ville puis Chercher." if self.kind.get() == "town"
                       else "Tapez le départ et l'arrivée, puis Tracer le trajet.")

    # ---- work on threads, answers on Tk's ---------------------------------------------------------
    def _bg(self, fn, done):
        def work():
            try:
                out = fn()
            except Exception as e:      # (a refused request, the server gone: said in the window)
                out = {"error": f"{type(e).__name__}: {e}"}
            self.q.put((done, out))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, out = self.q.get_nowait()
                done(out)
        except queue.Empty:
            pass
        self._after(POLL_MS, self._poll)

    def _after(self, ms, fn):
        try:
            self.root.after(ms, fn)
        except tk.TclError:         # (the window is gone)
            pass

    def _set_pick(self, pick, text):
        self.pick = pick
        self.est.configure(text=text)
        state = "normal" if pick and pick.get("mode") else "disabled"
        self.b_now.configure(state=state)
        self.b_bg.configure(state=state)

    # ---- step 1 -----------------------------------------------------------------------------------
    def search(self):
        q = self.q_town.get().strip()
        if not q:
            return
        self.res.delete(0, "end")
        self._set_pick(None, "Recherche…")
        self._bg(lambda: self.bridge.ask({"type": "geocode", "q": q, "full": True}), self._found)

    def _found(self, out):
        self.results = (out.get("results") or [])[:6]
        self.res.delete(0, "end")
        for it in self.results:
            self.res.insert("end", f"{it.get('name')}, {it['detail']}" if it.get("detail") else str(it.get("name")))
        if not self.results:
            self._set_pick(None, out.get("text") or out.get("error") or "Rien trouvé.")
        else:
            self._set_pick(None, "Choisissez le bon lieu dans la liste.")
            if len(self.results) == 1:
                self.res.selection_set(0)
                self.choose()

    def choose(self):
        sel = self.res.curselection()
        if not sel:
            return
        it = self.results[sel[0]]
        lat, lon = float(it["lat"]), float(it["lon"])
        pick = dict(name=str(it.get("name") or "Lieu"), start=[lat, lon], box=box_around(lat, lon, TOWN_KM))
        self._estimate(pick)

    def trace(self):
        a, b = self.q_a.get().strip(), self.q_b.get().strip()
        if not (a and b):
            return
        self._set_pick(None, "Calcul du trajet…")

        def work():
            pa = (self.bridge.ask({"type": "geocode", "q": a, "full": True}).get("results") or [None])[0]
            pb = (self.bridge.ask({"type": "geocode", "q": b, "full": True}).get("results") or [None])[0]
            if not (pa and pb):
                return {"error": "Départ ou arrivée introuvable."}
            r = self.bridge.ask({"type": "route", "lat0": pa["lat"], "lon0": pa["lon"], "lat1": pb["lat"],
                                 "lon1": pb["lon"], "download": "off"})
            if r.get("error") or not r.get("points"):
                return {"error": "Trajet impossible : " + str(r.get("error") or "aucune route")}
            return {"pick": dict(name=f"{a} → {b}", start=r["points"][0], route=True, line=r["points"],
                                 info=f"{round(r.get('dist', 0) / 1000)} km de route")}
        self._bg(work, lambda out: self._estimate(out["pick"]) if "pick" in out else self._set_pick(None, out["error"]))

    def _area(self, pick):
        return {"route": "current", "half": ROUTE_HALF_M} if pick.get("route") else dict(pick["box"])

    def show_thumb(self, name, s=None, w=None, n=None, e=None, line=None):
        """The miniature of a place (a box, or a route's points), made on a thread."""
        if line:
            s, w, n, e = bounds_of(line)
        key = (name, round(s, 5), round(w, 5), round(n, 5), round(e, 5))
        self.thumb_for = key
        self.thumb.configure(text="Chargement de l'aperçu…", image=self._blank)
        self.thumb_cap.configure(text=name)

        def done(png):
            if self.thumb_for != key:           # (another place was picked meanwhile)
                return
            if not isinstance(png, bytes):
                self.thumb.configure(text="Aperçu indisponible")
                return
            self.thumb_img = tk.PhotoImage(data=base64.b64encode(png).decode("ascii"), format="png")
            self.thumb.configure(image=self.thumb_img, text="")
        self._bg(lambda: render_thumb(self.bridge.tile, s, w, n, e, line), done)

    def _estimate(self, pick):
        self._set_pick(None, f"{pick['name']} : estimation…")
        if pick.get("line"):
            self.show_thumb(pick["name"], line=pick["line"])
        elif pick.get("box"):
            b = pick["box"]
            self.show_thumb(pick["name"], b["s"], b["w"], b["n"], b["e"])

        def work():
            for mode in ("world", "data"):
                r = self.bridge.ask(dict(self._area(pick), type="area_plan", mode=mode))
                if r.get("ok"):
                    return dict(r, mode=mode)
            return r

        def done(r):
            if not r.get("ok"):
                self._set_pick(None, "Pas d'estimation : " + str(r.get("error") or "erreur"))
                return
            pick["mode"] = r["mode"]
            text = (f"{pick['name']} : {round(r.get('km2') or 0)} km²"
                    + (f", {pick['info']}" if pick.get("info") else "")
                    + f" · {mb(r.get('downloadBytes'))} à télécharger · {mb(r.get('diskBytes'))} sur le disque"
                    + f" · environ {duration(r.get('seconds'))}")
            if r["mode"] == "data":
                text += "\nTrop grand pour tout construire d'avance : seulement les téléchargements."
            self._set_pick(pick, text)
        self._bg(work, done)

    def prepare(self, fast):
        pick = self.pick
        if not pick:
            return
        self._set_pick(None, f"{pick['name']} : envoi…")
        msg = dict(self._area(pick), mode=pick["mode"], name=pick["name"], fast=fast, start=pick["start"])

        def done(out):
            if out.get("ok"):
                self._set_pick(None, f"{pick['name']} (préparé) est ajouté aux lieux de départ.")
                self.refresh()
            else:
                self._set_pick(pick, "Refusé : " + str(out.get("error") or "erreur"))
        self._bg(lambda: prelaunch.prepare(self.bridge, self.user_dir, msg), done)

    # ---- step 2 -----------------------------------------------------------------------------------
    def refresh(self):
        self._bg(self.bridge.status, self._show)

    def _tick(self):
        self.refresh()
        self._after(STATUS_MS, self._tick)

    def _show(self, st):
        try:
            self.root.winfo_exists()
        except tk.TclError:
            return
        c, a = st.get("cache") or {}, st.get("areas") or {}
        if c.get("capBytes") is not None:
            pinned = c.get("pinnedBytes") or a.get("pinnedBytes")
            self.cache.configure(text=f"Cache : {mb(c.get('usedBytes'))} sur {mb(c['capBytes'])}"
                                      + (f", dont {mb(pinned)} de lieux préparés (jamais effacés)" if pinned else ""))
        areas = a.get("areas") or []
        shown = [(it.get("id"), it.get("name"), it.get("state"), it.get("done"), it.get("total"), a.get("fast") == it.get("id"))
                 for it in areas]
        if shown == getattr(self, "_shown", None):      # (rebuilt only when it changed: no flicker, no lost click)
            return
        self._shown = shown
        for w in self.list.winfo_children():
            w.destroy()
        if not areas:
            ttk.Label(self.list, text="Aucun lieu préparé pour l'instant.", style="Muted.TLabel").grid(
                row=0, column=0, sticky="w")
        for i, it in enumerate(areas):
            total = it.get("total") or 0
            pct = min(100, round(100 * (it.get("done") or 0) / total)) if total else (100 if it.get("state") == "done" else 0)
            state = STATE.get(it.get("state"), str(it.get("state")))
            if a.get("fast") == it.get("id"):
                state += ", à fond"
            done = it.get("state") == "done"
            name = ttk.Label(self.list, text=f"{it.get('name') or 'Zone'} (préparé)", style="Card.TLabel",
                             cursor="hand2")
            name.grid(row=i, column=0, sticky="w", pady=3)
            name.bind("<Button-1>", lambda e, it=it: self.show_area(it))
            ttk.Progressbar(self.list, value=pct, maximum=100, length=150,
                            style="Horizontal.TProgressbar" if done else "Run.Horizontal.TProgressbar").grid(
                row=i, column=1, sticky="ew", padx=10)
            ttk.Label(self.list, text="✓ prêt" if done else f"{state} · {pct} %",
                      style="Ready.TLabel" if done else "Run.TLabel", width=18).grid(row=i, column=2, sticky="w")
            ttk.Button(self.list, text="Voir", style="Small.TButton",
                       command=lambda it=it: self.show_area(it)).grid(row=i, column=3, padx=(6, 0))
            ttk.Button(self.list, text="Supprimer", style="Small.TButton",
                       command=lambda it=it: self.delete(it)).grid(row=i, column=4, padx=(4, 0))

    def show_area(self, it):
        """A prepared place in the miniature."""
        name = f"{it.get('name') or 'Zone'} (préparé)"
        if it.get("route"):
            self.show_thumb(name, line=it["route"])
        elif all(isinstance(it.get(k), (int, float)) for k in "swne"):
            self.show_thumb(name, it["s"], it["w"], it["n"], it["e"])

    def delete(self, it):
        name = it.get("name") or "Zone"
        if not messagebox.askyesno("Supprimer", f"Supprimer « {name} (préparé) » ?\n\nSon lieu de départ est retiré ; "
                                   "ses données ne sont plus protégées et le cache les efface quand il manque de place.",
                                   parent=self.root):
            return
        self._bg(lambda: prelaunch.delete(self.bridge, self.user_dir, it.get("id"), name), lambda out: self.refresh())

    # ---- step 3 -----------------------------------------------------------------------------------
    def start_game(self):
        try:
            prelaunch.launch_game(self.bridge, self.launch)
        finally:
            self.root.destroy()

    def run(self):
        self.root.after(POLL_MS, self._poll)
        self.root.after(0, self._tick)
        self.root.mainloop()
        self.launch.set()       # (however the window ended)
