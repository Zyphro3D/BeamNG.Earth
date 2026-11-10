-- luajit mod/tests/test_coreperf.lua  (the core's frame timings, coreperf.lua)
-- The mod's parts timed per frame and handed over at the frame's end, the hitch frames kept with the engine's
-- breakdown (at most HITCH_KEEP, drained by hitches()), the performance log's lines logged and sent, the engine tick.
local T = dofile((arg[0]:match('^(.*)[/\\]') or '.') .. '/testlib.lua')

local sent, logged = {}, {}
local connected = true
package.loaded['beamngEarth/net'] = { isConnected = function() return connected end,
  send = function(m) sent[#sent + 1] = m end }
package.loaded['beamngEarth/roads'] = { status = function() return { queued = 3 } end, scene = function() return {} end }
package.loaded['beamngEarth/trees'] = { status = function() return {} end, frameWork = function() return nil end }
package.loaded['beamngEarth/farfield'] = { stats = function() return { queued = 0, done = true } end }
package.loaded['beamngEarth/beammp'] = { inSession = function() return false end }
local perfLines, perfCtx = {}, nil
package.loaded['beamngEarth/perflog'] = {
  UNMEASURED = 'unmeasured',
  cause = function() return 'x' end,
  frame = function(_, _, context) perfCtx = context(); return perfLines end,
}
log = function(level, _, line) logged[#logged + 1] = level .. ' ' .. line end
Engine = { Debug = { getLastPerformanceMetrics = function(m) m.Ticking = 7.5; m.Render = 30; m.physA = 99 end } }

local st = { clock = 12.34, player = { lat = 1, lon = 2, vx = 10, vy = 0 }, guard = {} }
local W = require('beamngEarth/coreperf').new({ state = function() return st end, dropHold = { active = false },
  simPaused = function() return false end })

-- the parts: summed within a frame, the last whole frame's after endFrame
local t = W.mark('roads', W.clock())
W.mark('roads', t)
T.check(W.marks().roads ~= nil, "the frame's parts are kept while it runs")
W.endFrame()
T.check(W.frameMarks().roads ~= nil, "after endFrame they are the last whole frame's")
T.check(next(W.marks()) == nil, 'and the new frame starts empty')

-- a hitch: the engine's breakdown without the physics lines, at most HITCH_KEEP kept, drained by hitches()
local h = W.noteHitch(80.4)
T.check(h.ms == 80 and h.at == 12.3, 'a hitch keeps its ms and when')
T.check(h.engine.Render == 30 and h.engine.physA == nil, "the engine's subsystems, not its physics")
W.HITCH_KEEP = 3
for _ = 1, 5 do W.noteHitch(60) end
T.check(#W.hitches() == 3, 'at most HITCH_KEEP hitches kept')
T.check(#W.hitches() == 0, 'hitches() drains them')

-- the performance log: its context (where, the road tiles queued), its lines logged and sent while connected
perfLines = { 'perf hitch 80 ms' }
W.frame(80, h)
T.check(perfCtx and perfCtx.lat == 1 and perfCtx.doing:find('road tiles 3 queued', 1, true) ~= nil, 'the context')
T.check(logged[#logged] == 'I perf hitch 80 ms', "the lines go to the game's log")
local perfLogs = 0
for _, m in ipairs(sent) do if m.type == 'perf_log' then perfLogs = perfLogs + 1 end end
T.check(perfLogs >= 1, '... and to the server')
connected = false
local before = #sent
W.frame(80, h)
T.check(#sent == before, 'nothing sent without a connection')

T.near(W.engineTickMs(), 7.5, 1e-9, "the engine's tick")
Engine = nil
T.check(W.engineTickMs() == 0, 'no metrics: 0')

T.done('test_coreperf')
