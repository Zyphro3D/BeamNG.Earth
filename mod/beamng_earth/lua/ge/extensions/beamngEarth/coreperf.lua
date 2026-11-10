-- The core's frame timings (core.lua's onUpdate drives them; split out of core.lua, whose main chunk sat at Lua's
-- 200-local limit): the mod's parts timed every frame (mark), the hitch frames kept with the engine's breakdown
-- (noteHitch, hitches), the performance log's context and lines (frame: perflog.lua) and the anonymous performance
-- summary (psum: perfsummary.lua), the Lua garbage per minute (alloc).
local M = {}

-- A frame watcher. ctx: state (a function: core's runtime state, which resetState replaces), dropHold (core's
-- drophold), simPaused (a function). Its modules are the ones core.lua required when it made it.
function M.new(ctx)
  local W = { HITCH_KEEP = 64 }
  local stopwatch = require('beamngEarth/stopwatch')
  local net = require('beamngEarth/net')
  local roads = require('beamngEarth/roads')
  local trees = require('beamngEarth/trees')
  local farfield = require('beamngEarth/farfield')
  local beammp = require('beamngEarth/beammp')
  local perflog = require('beamngEarth/perflog')
  local dropHold, simPaused = ctx.dropHold, ctx.simPaused

  -- Hitch attribution (the user, 2026-10-14: "there's definitely some performance issues sometimes"): each frame
  -- times the mod's own parts; a frame over core's HITCH_MS keeps them and the engine's per-subsystem breakdown, so a
  -- recorded drive (tools/autodrive_perf.py) says whether we or the engine (rendering, collision, streaming) took it.
  local hitchClock = stopwatch.clock()
  W.clock = hitchClock
  local marks, prevMarks = {}, {}
  local hitches = {}
  function W.mark(name, t0)
    local t = hitchClock()
    marks[name] = (marks[name] or 0) + (t - t0)
    -- (the Lua garbage the part made since the last mark: W.alloc's per-part split for the perf lines)
    local al = W.alloc
    if al and al.markKb then
      local kb = collectgarbage('count')
      local d = kb - al.markKb
      if d > 0 then al.cur.marks[name] = (al.cur.marks[name] or 0) + d end
      al.markKb = kb
    end
    return t
  end
  -- The mod's parts' ms in the last whole frame (diagnostics: a per-frame probe; not to be changed by the caller)
  function W.frameMarks() return prevMarks end
  function W.noteHitch(frameMs)
    local st = ctx.state()
    local h = { ms = math.floor(frameMs + 0.5), at = math.floor(st.clock * 10 + 0.5) / 10, mod = {} }
    -- marks: the mod's parts in the frame that just ended (onUpdate swaps them to prevMarks after this call; reading
    -- prevMarks named the frame before it, and most Lua-led hitches showed no part at all, 2026-10-18)
    for k, v in pairs(marks) do if v >= 1 then h.mod[k] = math.floor(v * 10 + 0.5) / 10 end end
    if Engine and Engine.Debug and Engine.Debug.getLastPerformanceMetrics then
      local m = {}
      if pcall(Engine.Debug.getLastPerformanceMetrics, m) then
        local arr = {}
        for k, v in pairs(m) do
          if type(v) == 'number' and not k:find('^phys') and not k:find('^virt') then arr[#arr + 1] = { k, v } end
        end
        table.sort(arr, function(a, b) return a[2] > b[2] end)
        h.engine = {}
        for i = 1, math.min(6, #arr) do h.engine[arr[i][1]] = math.floor(arr[i][2] * 10 + 0.5) / 10 end
      end
    end
    hitches[#hitches + 1] = h
    if #hitches > W.HITCH_KEEP then table.remove(hitches, 1) end
    return h
  end

  -- The performance log (perflog.lua): where the player is and what the mod is doing, for its lines.
  local perfMetrics = {}
  -- The mod's Lua garbage over the last whole minute (perf lines' "alloc"): what the frames grew the heap by (a lower
  -- bound: collection steps inside a frame hide some), what decoding the server's messages made, and the message type
  -- whose handling made the most. LuaGC frames of up to 183 ms on a fast drive (Fells Point, 2026-11-07): which part
  -- of the mod feeds the collector. One collectgarbage('count') per step: microseconds.
  W.alloc = { at = nil, kb = nil, markKb = nil, cur = { total = 0, decode = 0, types = {}, marks = {} }, last = nil }
  function W.alloc.roll(now)
    local alloc = W.alloc
    if not alloc.at then alloc.at = now end
    if now - alloc.at >= 60 then
      alloc.last, alloc.at = alloc.cur, now
      alloc.cur = { total = 0, decode = 0, types = {}, marks = {} }
    end
  end
  function W.alloc.frame()
    local alloc = W.alloc
    local kb = collectgarbage('count')
    if alloc.kb and kb > alloc.kb then alloc.cur.total = alloc.cur.total + (kb - alloc.kb) end
    alloc.kb = kb
  end
  function W.alloc.text()
    local a = W.alloc.last
    if not a then return '' end
    local top, topKb = nil, 0
    for k, v in pairs(a.types) do if v > topKb then top, topKb = k, v end end
    -- the frame's parts that made the most (mark names: roads, sea+trees, far, ...)
    local parts = {}
    for k, v in pairs(a.marks or {}) do parts[#parts + 1] = { k, v } end
    table.sort(parts, function(x, y) return x[2] > y[2] end)
    local ps = {}
    for i = 1, math.min(4, #parts) do ps[#ps + 1] = string.format('%s %.0f', parts[i][1], parts[i][2] / 1024) end
    -- (total: the whole game Lua's heap growth, the game's own UI / vehicles / gameplay too (a 2026-11-07 probe: ~20 of 108
    -- MB/min ours, parked); decode, msg and parts: ours)
    return string.format(', GE Lua garbage %.0f MB/min (ours: decode %.0f%s%s)', a.total / 1024, a.decode / 1024,
      top and string.format(', msg %s %.0f', top, topKb / 1024) or '', #ps > 0 and ('; parts ' .. table.concat(ps, ', ')) or '')
  end
  W.perfOut = {}        -- (perfEngine's answer, refilled: perflog only sums it into its window; not a table a call)
  local function perfEngine()
    local dbg = Engine and Engine.Debug
    if not (dbg and dbg.getLastPerformanceMetrics) then return nil end
    for k in pairs(perfMetrics) do perfMetrics[k] = nil end
    if not pcall(dbg.getLastPerformanceMetrics, perfMetrics) then return nil end
    local out = W.perfOut
    for k in pairs(out) do out[k] = nil end
    for k, v in pairs(perfMetrics) do
      if type(v) == 'number' and not k:find('^phys') and not k:find('^virt') then out[k] = v end
    end
    return out
  end
  local function perfContext()
    local st = ctx.state()
    local c = {}
    local p = st.player
    if p then
      c.lat, c.lon = p.lat, p.lon
      c.mph = math.sqrt((p.vx or 0) ^ 2 + (p.vy or 0) ^ 2) * 2.23694
    end
    if st.block then c.block = string.format('%d,%d', st.block.cx, st.block.cy) end
    local s = {}
    if dropHold.active then s[#s + 1] = 'held until its ground loads' end
    if st.guard and st.guard.pausedByUs then s[#s + 1] = 'held at the edge of the ground' end
    if st.rebasing then s[#s + 1] = 'moving the world origin' end
    if st.clock < (st.loadingUntil or 0) or st.placeFrames then s[#s + 1] = 'loading after a teleport' end
    if not net.isConnected() then s[#s + 1] = 'world server not connected' end
    if beammp.inSession() then s[#s + 1] = 'BeamMP' end
    c.state = table.concat(s, ', ')
    local d = {}
    if st.commitFrames then d[#d + 1] = 'ground block swap' end
    if st.terrainPending and st.terrainPending.sentAt then
      d[#d + 1] = string.format('next block asked %.1fs ago', st.clock - st.terrainPending.sentAt)
    end
    local okF, fs = pcall(farfield.stats)
    if okF and fs then
      if (fs.queued or 0) > 0 then d[#d + 1] = 'far tiles ' .. fs.queued .. ' queued' end
      if not fs.done then d[#d + 1] = 'far field loading' end
    end
    local okR, rs = pcall(roads.status)
    if okR and rs and ((rs.queued or 0) + (rs.walksQueued or 0)) > 0 then
      d[#d + 1] = 'road tiles ' .. ((rs.queued or 0) + (rs.walksQueued or 0)) .. ' queued'
    end
    local okT, ts = pcall(trees.status)
    if okT and ts and (ts.pending or 0) > 0 then d[#d + 1] = 'tree tiles ' .. ts.pending .. ' pending' end
    local okW, fw = pcall(trees.frameWork)
    if okW and fw then
      for i = 1, 2 do
        local w = fw[i]
        if w.made + w.removed + w.forests > 0 then
          d[#d + 1] = string.format('trees %s: +%d -%d items, %d shapes (%d items in the level)%s%s',
            i == 1 and 'last pass' or 'pass before', w.made, w.removed, w.shapes, w.items,
            w.new > 0 and (', ' .. w.new .. ' new shapes') or '', w.forests > 0 and (', ' .. w.forests .. ' new Forests') or '')
        end
      end
    end
    if (st.serverBacklog or 0) > 0 then d[#d + 1] = 'server backlog ' .. st.serverBacklog end
    c.doing = table.concat(d, ', ')
    c.extra = string.format('Lua %.0f MB%s%s', collectgarbage('count') / 1024,
      st.terrainLatency and string.format(', last block %.1fs', st.terrainLatency) or '', W.alloc.text())
    return c
  end
  local machineSent = false       -- the machine's lines sent to this server connection
  local machineTryAt = nil
  local function perfMachine()
    local rep = rawget(_G, 'beamngEarth_report')
    local okH, hw = pcall(function() return rep and rep.collectHardware and rep.collectHardware() end)
    if not okH or not hw then return {} end
    local okF, md = pcall(require('beamngEarth/issuefmt').formatHardware, hw)
    if not okF or type(md) ~= 'table' then return {} end
    local out = {}
    for _, l in ipairs(md) do                -- its Markdown list lines, without the list and bold marks
      local item = type(l) == 'string' and l:match('^%- (.*)$')
      if item then out[#out + 1] = 'machine: ' .. (item:gsub('%*%*', '')) end
    end
    return out
  end
  -- The anonymous performance summary (perfsummary.lua, TELEMETRY.md sections 2-4): fed here, a window sent to the server
  -- every 5 minutes and the setup once a connection, only while the player has "Performance data" switched on (the server
  -- also keeps it only then). Counts and rounded steps; no position. 
  W.psum = { mod = require('beamngEarth/perfsummary'), sampleAt = 0, setupSent = false }
  function W.psum.on()
    local S = rawget(_G, 'beamngEarth_settings')
    if not (S and S.get) then return false end
    return S.get('telemetry') ~= false and S.get('telemetryPerf') ~= false and not (S.privacyNeeded and S.privacyNeeded())
  end
  function W.psum.sample()
    local st = ctx.state()
    local s = { heapMB = collectgarbage('count') / 1024 }
    if W.alloc.last then s.garbageMB = W.alloc.last.total / 1024 end
    local okS, sc = pcall(roads.scene)
    local loaded = {}
    if okS and sc then s.buildings = sc.buildings; loaded.roadTiles = sc.tiles; loaded.buildings = sc.buildings end
    local okR, rs = pcall(roads.status)
    if okR and rs then s.roadsQueued = (rs.queued or 0) + (rs.walksQueued or 0); loaded.farBuildings = rs.farBuildings end
    local okT, ts = pcall(trees.status)
    if okT and ts then loaded.trees = ts.items end
    local okF, fs = pcall(farfield.stats)
    if okF and fs then loaded.farTiles = fs.live end
    local P = rawget(_G, 'beamngEarth_parking') or package.loaded['beamngEarth/parking']
    local okP, ps = pcall(function() return P and P.status() end)
    if okP and ps then loaded.parked = ps.spots end
    local N = package.loaded['beamngEarth/nightlights']
    local okN, ns = pcall(function() return N and N.status() end)
    if okN and ns then loaded.lamps = ns.lit end
    local okV, tv = pcall(function()
      local api = rawget(_G, 'gameplay_traffic')
      return api and api.getNumOfTraffic and api.getNumOfTraffic()
    end)
    if okV and type(tv) == 'number' then loaded.traffic = tv end
    s.loaded = loaded
    -- held at the edge: the guard holding the car back (mode 'hold') or the rare full pause; the first night counted
    -- only the pause (0.0 % over 233 hours)
    s.edge = (st.guard and (st.guard.mode == 'hold' or st.guard.pausedByUs)) and true or false
    s.drop = dropHold.active or false
    s.loading = st.clock < (st.loadingUntil or 0) or st.placeFrames ~= nil
    local m3 = rawget(_G, 'beamngEarth_map3d')
    s.mapOpen = (m3 and m3.isActive and m3.isActive()) or false
    return s
  end
  function W.psum.frame(ms, hrec)
    local st = ctx.state()
    local ps = W.psum
    if not net.isConnected() then ps.setupSent = false; return end
    if not ps.on() then return end
    local p = st.player
    local mph = p and math.sqrt((p.vx or 0) ^ 2 + (p.vy or 0) ^ 2) * 2.23694 or 0
    local cause = ms >= ps.mod.CAUSE_MS and perflog.cause(ms, hrec) or nil
    -- only frames the player is playing: not paused (menus, the edge guard's pause: that time is counted as held) and
    -- not in the background, where the game slows its own frames down (the game's lowPerformance sampler does the same)
    local focused = not Engine.isProgramFocused or Engine.isProgramFocused()
    if focused and not simPaused() then
      ps.mod.frame(ms, mph, cause ~= perflog.UNMEASURED and cause or nil,
        st.clock < (st.loadingUntil or 0) or st.placeFrames ~= nil)
    end
    if st.clock < ps.sampleAt then return end
    ps.sampleAt = st.clock + 1
    local okS, smp = pcall(ps.sample)
    if okS then ps.mod.sample(st.clock, smp) end
    if not ps.setupSent then
      local rep = rawget(_G, 'beamngEarth_report')
      local okH, hw = pcall(function() return rep and rep.collectHardware and rep.collectHardware() end)
      local S = rawget(_G, 'beamngEarth_settings')
      if okH and hw then
        local okU, setup = pcall(ps.mod.setup, hw, S and S.get)
        if okU then net.send({ type = 'perf_setup', setup = setup }); ps.setupSent = true end
      end
    end
    local summary = ps.mod.take(st.clock, perflog.takeDrops)
    if summary then net.send({ type = 'perf_summary', summary = summary }) end
  end
  function W.frame(ms, hrec)
    local st = ctx.state()
    local lines = perflog.frame(ms, st.clock, perfContext, hrec, marks, perfEngine)
    pcall(W.psum.frame, ms, hrec)
    if not net.isConnected() then
      machineSent = false
    elseif not machineSent and st.clock >= (machineTryAt or 0) then
      local m = perfMachine()
      if #m > 0 then
        machineSent = true
        net.send({ type = 'perf_log', lines = m })
      else
        machineTryAt = st.clock + 10           -- (the game's report extension not loaded yet)
      end
    end
    if #lines == 0 then return end
    for _, l in ipairs(lines) do log('I', 'beamngEarth', l) end
    if net.isConnected() then net.send({ type = 'perf_log', lines = lines }) end
  end
  -- The hitches since the last call: [{ms, at (s), mod = {part = ms}, engine = {subsystem = ms}}].
  function W.hitches()
    local out = hitches
    hitches = {}
    return out
  end

  -- The previous frame's engine tick (ms), 0 without the metrics.
  local tickMetrics = {}
  function W.engineTickMs()
    local dbg = Engine and Engine.Debug
    if not (dbg and dbg.getLastPerformanceMetrics) then return 0 end
    if not pcall(dbg.getLastPerformanceMetrics, tickMetrics) then return 0 end
    return tonumber(tickMetrics.Ticking) or 0
  end

  -- The marks of the frame being timed (profiler.lua reads them); endFrame makes them the last whole frame's.
  function W.marks() return marks end
  function W.endFrame()
    prevMarks, marks = marks, prevMarks
    for k in pairs(marks) do marks[k] = nil end
    if W.alloc then W.alloc.markKb = collectgarbage('count') end      -- (the frame's parts start counting here)
  end
  return W
end

return M
