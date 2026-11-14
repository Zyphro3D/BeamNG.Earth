-- luajit mod/tests/test_lang.lua  (the interface's language: lang.lua, lang_fr.json)
-- The language from the setting, the game's own and the fallback; exact entries with their spaces; built lines
-- through "subs"; English untouched; the French dictionary file loads and holds the settings' labels.
local dir = (arg[0]:match('^(.*)[/\\]') or '.')
local T = dofile(dir .. '/testlib.lua')
package.loaded['beamngEarth/settings'] = { get = function(k) return _G._lang_setting end }
local L = require('beamngEarth/lang')
L.CACHE_S = 0

-- the language in use
_G.settings = { getValue = function(k) return k == 'userLanguage' and _G._game_lang or nil end }
_G._lang_setting, _G._game_lang = 'auto', 'fr_FR'
T.check(L.current() == 'fr', 'auto: a French game is French')
_G._game_lang = 'de_DE'
T.check(L.current() == 'en', 'auto: a language with no dictionary is English')
_G._game_lang = nil
T.check(L.current() == 'fr', 'auto: the game language unread: French (this fork)')
_G._lang_setting = 'en'
T.check(L.current() == 'en', 'the setting en wins')
_G._lang_setting, _G._game_lang = 'fr', 'en_US'
T.check(L.current() == 'fr', 'the setting fr wins over an English game')

-- translations
L._force('fr', { exact = { ['Tree density'] = 'Densité des arbres', ['Loading 100%'] = 'Chargement 100 %' },
                 subs = { { 'Loading', 'Chargement' }, { 'distant land', 'paysage lointain' }, { '50%', '50 %' } } })
T.check(L.t('Tree density') == 'Densité des arbres', 'an exact entry')
T.check(L.t('  Tree density ') == '  Densité des arbres ', 'its spaces kept')
T.check(L.t('Not in it') == 'Not in it', 'no entry: unchanged')
T.check(L.tr('Loading 100%') == 'Chargement 100 %', 'a built line: its exact entry first')
T.check(L.tr('Loading 50%  |  distant land 3/9') == 'Chargement 50 %  |  paysage lointain 3/9', 'else the subs, plain text')
T.check(L.t(nil) == nil and L.tr('') == '', 'nothing stays nothing')
L._force('en')
T.check(L.t('Tree density') == 'Tree density' and L.tr('Loading 5%') == 'Loading 5%', 'English: untouched')

-- the French dictionary
local f = io.open(dir .. '/../beamng_earth/lua/ge/extensions/beamngEarth/lang_fr.json', 'r')
local text = f and f:read('*a') or ''
if f then f:close() end
T.check(#text > 1000, 'lang_fr.json is there')
for _, label in ipairs({ 'Tree density', 'Language', 'Quality' }) do
  T.check(text:find('"' .. label .. '"', 1, true) ~= nil, 'the dictionary has "' .. label .. '"')
end

T.done('test_lang')
