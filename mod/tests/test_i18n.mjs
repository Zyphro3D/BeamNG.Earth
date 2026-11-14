// node mod/tests/test_i18n.mjs  (i18n.js: the screens' texts turned by the dictionary, spaces kept)
import * as i18n from "../beamng_earth/ui/ui-vue/mods/beamngEarth/i18n.js"
import assert from "node:assert"
import { readFileSync } from "node:fs"
const { turn, setDict } = i18n._test
const fr = JSON.parse(readFileSync(new URL("../beamng_earth/lua/ge/extensions/beamngEarth/lang_fr.json", import.meta.url)))
setDict(fr.exact)
assert.equal(turn("Tree density"), "Densité des arbres")
assert.equal(turn("  Tree density\n  "), "  Densité des arbres\n  ")      // (a template's text node: its spaces kept)
assert.equal(turn("Tree   density"), "Densité des arbres")               // (spaces inside collapsed, as the extractor)
assert.equal(turn("Not a text of ours"), null)
setDict(null)
assert.equal(turn("Tree density"), null)                                  // (English: nothing turned)
console.log("i18n ok:", Object.keys(fr.exact).length, "entries")
