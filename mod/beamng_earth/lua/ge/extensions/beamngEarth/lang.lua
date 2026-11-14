-- The mod's interface in the player's language (the France fork; the user, 2026-11-14: "the Earth interface in
-- French"). The texts stay written in English in the code; a dictionary per language (lang_<code>.json beside this
-- file: { "exact": { "English text": "translation" } }, tools/lang_extract.py lists the texts) turns them at display:
-- here for what Lua draws or sends (the banner, the HUD, the loading line, messages), in the UI by i18n.js (the
-- screens' text nodes and titles, from the same file through current() and dict()).
--
-- The language: the setting 'language' (settings.lua): 'fr', 'en', or 'auto' = the game's own language when it can be
-- read (French when it starts with "fr"), else French (this fork is France's). A text with no entry stays English.
-- Engine-light: the dictionary is read once (jsonReadFile), t() is one table lookup; runs under plain luajit (tests).

local M = {}

M.DIR = '/lua/ge/extensions/beamngEarth/'
M.FALLBACK = 'fr'           -- 'auto' when the game's language cannot be read
M.KNOWN = { fr = true }     -- the languages with a dictionary (English is the code's own)

local dicts = {}            -- code -> { exact = {...} } (false: tried, none)
local forced = nil          -- tests: M._force(code)

local function gameLanguage()
  local ok, v = pcall(function()
    local s = rawget(_G, 'settings')
    return s and s.getValue and s.getValue('userLanguage')
  end)
  if ok and type(v) == 'string' and v ~= '' then return v end
  ok, v = pcall(function()
    local L = rawget(_G, 'Lang')
    return L and L.getSelectedLanguage and L.getSelectedLanguage()
  end)
  if ok and type(v) == 'string' and v ~= '' then return v end
  return nil
end

M.CACHE_S = 1.0             -- s the language in use is kept before the setting is read again (it is asked every frame)
local cached, cachedAt = nil, -math.huge

-- The language in use: 'fr' or 'en'.
function M.current()
  if forced then return forced end
  local now = os.clock()
  if cached and now - cachedAt < M.CACHE_S then return cached end
  cached, cachedAt = M._resolve(), now
  return cached
end

function M._resolve()
  local want = 'auto'
  local ok, s = pcall(require, 'beamngEarth/settings')
  if ok and s and s.get then want = s.get('language') or 'auto' end
  if want == 'auto' then
    local g = gameLanguage()
    if g == nil then want = M.FALLBACK
    else want = g:lower():sub(1, 2) end
  end
  return M.KNOWN[want] and want or 'en'
end

local function load(code)
  if dicts[code] ~= nil then return dicts[code] end
  local read = rawget(_G, 'jsonReadFile')
  local d = false
  if read then
    local ok, got = pcall(read, M.DIR .. 'lang_' .. code .. '.json')
    if ok and type(got) == 'table' and type(got.exact) == 'table' then d = got end
  end
  dicts[code] = d
  return d
end

-- The dictionary of the language in use ({ exact = {...} }), or nil for English.
function M.dict()
  local code = M.current()
  if code == 'en' then return nil end
  return load(code) or nil
end

-- text in the language in use (unchanged when it has no entry).
function M.t(text)
  if type(text) ~= 'string' or text == '' then return text end
  local d = M.dict()
  if not d then return text end
  local got = d.exact[text]
  if got then return got end
  -- (a text with spaces round it, as some lines are built: the same entry, its spaces kept)
  local lead, core, tail = text:match('^(%s*)(.-)(%s*)$')
  got = core ~= text and d.exact[core]
  return got and (lead .. got .. tail) or text
end

-- A built line (the loading line, the banner: numbers in them): its exact entry, else the dictionary's "subs" (pairs
-- [English, translation], in order, plain text) replaced in it. Only for lines this mod builds itself.
function M.tr(text)
  if type(text) ~= 'string' or text == '' then return text end
  local d = M.dict()
  if not d then return text end
  local got = d.exact[text]
  if got then return got end
  if type(d.subs) ~= 'table' then return text end
  for _, p in ipairs(d.subs) do
    if type(p) == 'table' and type(p[1]) == 'string' and type(p[2]) == 'string' then
      local a = p[1]:gsub('[%^%$%(%)%%%.%[%]%*%+%-%?]', '%%%0')
      text = text:gsub(a, (p[2]:gsub('%%', '%%%%')))
    end
  end
  return text
end

function M._force(code, dict)     -- tests
  forced = code
  if dict then dicts[code] = dict end
end

return M
