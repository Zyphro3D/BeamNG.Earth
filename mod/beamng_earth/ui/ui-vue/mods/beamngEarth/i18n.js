// BeamNG.Earth's screens in the player's language (the France fork; the user, 2026-11-14: "the Earth interface in
// French"). The screens stay written in English; this turns their texts at display, from the dictionary Lua reads
// (lua/ge/extensions/beamngEarth/lang.lua: the language setting, the game's own language, lang_<code>.json):
// a text node whose text (spaces round it kept) is an entry, and the title / placeholder / aria-label attributes.
// A MutationObserver keeps up with what Vue draws or changes; each node's English is remembered, so switching back to
// English (or another language) restores it. Only whole texts are turned: a sentence built with numbers in it stays
// English unless its pieces are entries of their own (tools/lang_extract.py lists the screens' texts).

const ATTRS = ["title", "placeholder", "aria-label"]
const POLL_MS = 3000          // the language setting looked at again this often
let exact = null              // the dictionary in use (null: English)
let code = "en"
let observer = null
let timer = null
const original = new WeakMap()      // text node -> its English; element -> { attr: English }

function engineLua(script, cb) {
  if (window.bngApi && window.bngApi.engineLua) window.bngApi.engineLua(script, cb)
}

function turn(text) {
  if (!exact || !text) return null
  const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(text)
  const key = m[2].replace(/\s+/g, " ")
  const got = exact[key]
  return got === undefined ? null : m[1] + got + m[3]
}

function textNode(n) {
  const now = n.nodeValue
  let en = original.get(n)
  if (en === undefined || (now !== en && now !== turn(en))) {     // (new, or Vue wrote a new English text)
    en = now
    original.set(n, en)
  }
  const want = turn(en) ?? en
  if (now !== want) n.nodeValue = want
}

function element(el) {
  let saved = original.get(el)
  for (const a of ATTRS) {
    if (!el.hasAttribute || !el.hasAttribute(a)) continue
    const now = el.getAttribute(a)
    saved = saved || {}
    let en = saved[a]
    if (en === undefined || (now !== en && now !== turn(en))) {
      en = now
      saved[a] = en
    }
    const want = turn(en) ?? en
    if (now !== want) el.setAttribute(a, want)
  }
  if (saved) original.set(el, saved)
}

function walk(root) {
  if (!root) return
  if (root.nodeType === 3) return textNode(root)
  if (root.nodeType !== 1 && root.nodeType !== 9 && root.nodeType !== 11) return
  if (root.nodeType === 1) {
    if (root.tagName === "SCRIPT" || root.tagName === "STYLE") return
    element(root)
  }
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT)
  for (let n = w.nextNode(); n; n = w.nextNode()) {
    if (n.nodeType === 3) textNode(n)
    else element(n)
  }
}

function observe() {
  if (observer) return
  observer = new MutationObserver(list => {
    for (const r of list) {
      if (r.type === "characterData") textNode(r.target)
      else if (r.type === "attributes") element(r.target)
      else for (const n of r.addedNodes) walk(n)
    }
  })
  observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true,
                                    attributeFilter: ATTRS })
}

function apply(got) {
  const next = got && got.code ? got.code : "en"
  const dict = got && got.exact && typeof got.exact === "object" ? got.exact : null
  if (next === code && (dict !== null) === (exact !== null)) return
  code = next
  exact = next === "en" ? null : dict
  observe()
  walk(document.body)                 // (both ways: a dictionary turns, English restores)
}

function poll() {
  // (the dictionary only when the language changed: one table of a few hundred entries)
  engineLua("(function() local ok, L = pcall(require, 'beamngEarth/lang'); if not ok then return nil end " +
            "return L.current() end)()", c => {
    const next = typeof c === "string" ? c : "en"
    if (next === code && (next === "en" || exact)) return
    if (next === "en") return apply({ code: "en" })
    engineLua("(function() local L = require('beamngEarth/lang'); local d = L.dict(); " +
              "return { code = L.current(), exact = d and d.exact or nil } end)()", apply)
  })
}

export function start() {
  if (timer) return
  poll()
  timer = setInterval(poll, POLL_MS)
}

export function stop() {
  if (timer) clearInterval(timer)
  timer = null
  if (observer) observer.disconnect()
  observer = null
  if (exact) { exact = null; code = "en"; walk(document.body) }      // (English back where we turned it)
}

export const _test = { turn, setDict: d => { exact = d; code = d ? "fr" : "en" }, walk }
