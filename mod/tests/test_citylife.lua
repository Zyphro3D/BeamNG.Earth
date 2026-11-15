-- luajit mod/tests/test_citylife.lua  (pedestrians: citylife.lua's segments from the road graph, walkers.lua on them)
local dir = (arg[0]:match('^(.*)[/\\]') or '.')
local T = dofile(dir .. '/testlib.lua')
package.path = dir .. '/../beamng_earth/lua/ge/extensions/?.lua;' .. package.path
local C = require('beamngEarth/citylife')
local walkers = require('beamngEarth/walkers')

-- a street of three edges (two plain nodes in it) crossed by a fast road at its middle node
local g = {
  nodes = { a = { 0, 0, 10, 4 }, b = { 50, 0, 10, 4 }, c = { 100, 0, 10, 4 }, d = { 150, 0, 10, 4 },
            n = { 50, 60, 10, 6 }, s = { 50, -60, 10, 6 } },
  edges = {
    { 'a', 'b', 0, 1, 1, 13.9, 1, 4 }, { 'b', 'c', 0, 1, 1, 13.9, 1, 4 }, { 'c', 'd', 0, 1, 1, 13.9, 1, 4 },
    { 'n', 'b', 0, 2, 2, 25, 1, 7 }, { 'b', 's', 0, 2, 2, 25, 1, 7 },
  },
}
local segs = C.segments(g)
local n, street, fast = 0, nil, 0
for _, s in pairs(segs) do
  n = n + 1
  if s[5] < 20 then street = s else fast = fast + 1 end
end
-- b has four edges (no chaining through it); c has two that agree: b-c-d is one segment
T.check(n == 4, 'a-b, b-c-d, n-b and b-s: b has four edges, nothing chained through it (' .. n .. ')')
local long = nil
for _, s in pairs(segs) do if s[5] < 20 and (#s - 7) / 3 == 3 then long = s end end
T.check(long ~= nil, 'b-c-d chained through c (3 points)')
T.check(long and long[4] == '+-' and long[6] == 4, 'lanes and half width carried')
T.check(C.segments(nil) and next(C.segments({})) == nil, 'no graph: no segments')
-- edges that differ (width) are not chained
local g2 = { nodes = { a = { 0, 0, 0 }, b = { 40, 0, 0 }, c = { 80, 0, 0 } },
  edges = { { 'a', 'b', 0, 1, 1, 13.9, 1, 4 }, { 'b', 'c', 0, 1, 1, 13.9, 1, 6 } } }
local k2 = 0
for _ in pairs(C.segments(g2)) do k2 = k2 + 1 end
T.check(k2 == 2, 'a change of width ends a segment')

T.check(C.countOf('off') == 0 and C.countOf('few') < C.countOf('normal') and C.countOf('normal') < C.countOf('many'),
  'the counts by setting')
T.check(C.countOf('nonsense') == C.COUNTS.normal, 'an unknown value: normal')

-- walkers.lua runs on them (pure: sync and step, no drawing)
local W = walkers.new({ seed = 7 })
walkers.sync(W, segs, 75, 0)
T.check(#W.list > 0, 'sidewalk pieces built (' .. #W.list .. ')')
local fastWalk = false
for _, piece in ipairs(W.list) do if piece.rec.src[5] > 20 then fastWalk = true end end
T.check(not fastWalk, 'no sidewalk on the 90 km/h road')
walkers.setTarget(W, 20)
for _ = 1, 200 do walkers.step(W, 0.05, 75, 0, {}, nil) end
local asked, walking = walkers.counts(W)
T.check(asked == 20 and walking > 0, 'walkers out (' .. walking .. ' of ' .. asked .. ')')
local ok = true
for _, w in ipairs(W.slots) do
  if not w.free and (w.x ~= w.x or math.abs(w.x) > 1000 or math.abs(w.y) > 1000) then ok = false end
end
T.check(ok, 'every walker somewhere sensible')

-- the game map's nodes (navgraph.toNodes' shape) back to the graph: each shared edge once
local e1 = { inNode = 'a', outNode = 'b', oneWay = false, lanes = '-+', speedLimit = 13.9, drivability = 1, inRadius = 4 }
local e2 = { inNode = 'b', outNode = 'c', oneWay = true, lanes = '++', speedLimit = 13.9, drivability = 1, inRadius = 4 }
local mapNodes = {
  a = { pos = { x = 0, y = 0, z = 1 }, radius = 4, links = { b = e1 } },
  b = { pos = { x = 40, y = 0, z = 1 }, radius = 4, links = { a = e1, c = e2 } },
  c = { pos = { x = 80, y = 0, z = 1 }, radius = 4, links = { b = e2 } },
}
local gm = C.graphOfMap(mapNodes)
T.check(#gm.edges == 2, 'each map edge once (' .. #gm.edges .. ')')
local one = nil
for _, e in ipairs(gm.edges) do if e[1] == 'b' then one = e end end
T.check(one and one[2] == 'c' and one[3] == 1 and one[4] == 2 and one[5] == 0, 'direction, one way and lanes kept')
T.check(gm.nodes.b[1] == 40 and gm.nodes.b[4] == 4, 'node position and radius')
T.check(#C.graphOfMap(nil).edges == 0, 'no map: no roads')

-- a graph that says where sidewalks are drawn (edge[13]): walkers only there
local g3 = { nodes = { a = { 0, 0, 0 }, b = { 60, 0, 0 }, c = { 0, 60, 0 } },
  edges = { { 'a', 'b', 0, 1, 1, 13.9, 1, 4, nil, nil, nil, nil, 2.0 }, { 'a', 'c', 0, 1, 1, 13.9, 1, 4, nil, nil, nil, nil, 0 } } }
local withWalk, without
for _, sg in pairs(C.segments(g3)) do
  if sg[3] == 'b' or sg[2] == 'b' then withWalk = sg else without = sg end
end
T.check(withWalk and withWalk[5] == 13.9 and math.abs(withWalk[6] - (2.0 - C.SIDE_GAP > 0.5 and 2.0 - C.SIDE_GAP or 0.5)) < 1e-9
  and withWalk[4] == '', 'a drawn sidewalk: walkers in its middle (edge[13] its offset)')
T.check(without and without[5] == C.NO_WALK_SPEED, 'no sidewalk drawn: no walkers there')
local W3 = walkers.new({ seed = 3 })
walkers.sync(W3, C.segments(g3), 0, 0)
local onC = false
for _, piece in ipairs(W3.list) do if piece.rec.src[3] == 'c' or piece.rec.src[2] == 'c' then onC = true end end
T.check(#W3.list > 0 and not onC, 'sidewalk pieces only along the road with one')
local e = { inNode = 'a', outNode = 'b', lanes = '-+', speedLimit = 13.9, drivability = 1, inRadius = 4, walk = 1.5 }
local gw = C.graphOfMap({ a = { pos = { x = 0, y = 0, z = 0 }, links = { b = e } }, b = { pos = { x = 30, y = 0, z = 0 }, links = { a = e } } })
T.check(gw.edges[1][13] == 1.5, 'the map edge carries its sidewalk width')

-- the traffic (flow.lua) on the raw segments: the AI lanes' half widths, every road
local flow = require('beamngEarth/flow')
local raw = C.segments(g3, false)
local okRaw = true
for _, sg in pairs(raw) do if sg[5] ~= 13.9 or sg[6] ~= 4 then okRaw = false end end
T.check(okRaw, 'the traffic gets the roads as they are (no sidewalk changes)')
local grid = { nodes = {}, edges = {} }
for i = 0, 4 do for j = 0, 4 do grid.nodes['n' .. i .. '_' .. j] = { i * 80, j * 80, 0, 4 } end end
for i = 0, 4 do for j = 0, 4 do
  if i < 4 then grid.edges[#grid.edges + 1] = { 'n' .. i .. '_' .. j, 'n' .. (i + 1) .. '_' .. j, 0, 1, 1, 13.9, 1, 4 } end
  if j < 4 then grid.edges[#grid.edges + 1] = { 'n' .. i .. '_' .. j, 'n' .. i .. '_' .. (j + 1), 0, 1, 1, 13.9, 1, 4 } end
end end
local F = flow.new({ leftHand = false, seed = 5, paints = flow.SHARE, bodies = flow.BODY_SHARE })
flow.sync(F, C.segments(grid, false), 160, 160, {}, {})
flow.setTarget(F, 30)
for k = 1, 600 do flow.step(F, 0.05, 160, 160, {}, k * 0.05, nil) end
T.check(F.count > 0, 'cars on the roads (' .. tostring(F.count) .. ')')
local sane = true
for _, car in pairs(F.cars) do
  local x, y = flow.pose(F, car.link, car.s, car.lane)
  if not x or x ~= x or x < -50 or x > 370 or y < -50 or y > 370 then sane = false end
end
T.check(sane, 'every car on the grid')
T.check(C.flowCountOf('off') == 0 and C.flowCountOf('light') < C.flowCountOf('dense'), 'the traffic counts by setting')

-- the kerbs measured on the road as drawn: a street along +x, its right (walkers' side 1) is -y
local street = { 1, 'a', 'b', '', 13.9, 2, 1, 0, 0, 0, 40, 0, 0, 80, 0, 0 }
local function height(x, y, z)
  if y <= -3.5 then return 0.14 end         -- a kerb 3.5 m right of the line
  if y >= 6 then return 0.03 end            -- a low step 6 m left: no kerb
  return 0
end
local sides = C.measureSides(street, height)
T.check(sides and math.abs(sides[1] - (3.5 + C.WALK_IN)) < 0.26, 'the right kerb found, the walker past it (' .. tostring(sides and sides[1]) .. ')')
T.check(sides and sides[-1] == false, 'no kerb on the left: no walkers there')
T.check(C.measureSides(street, function() return nil end) == nil, 'nothing read: unmeasured')
-- walkers.lua honours them: pieces on the right side only, at the measured offset
local Wm = walkers.new({ seed = 9 })
local s2 = { 1, 'a', 'b', '', 13.9, 2, 1, 0, 0, 0, 40, 0, 0, 80, 0, 0 }
s2.off = { [1] = 4.3, [-1] = false }
walkers.sync(Wm, { [1] = s2 }, 40, 0)
local okSide, n = true, 0
for _, piece in ipairs(Wm.list) do
  n = n + 1
  if piece.side ~= 1 or math.abs(piece.line[2] + 4.3) > 0.01 then okSide = false end
end
T.check(n > 0 and okSide, 'pieces only on the measured side, at its offset (' .. n .. ')')

-- a roundabout (one-way ring split at its four arms): the ring has the way, no gridlock (the original rules gave
-- 18 of 40 cars up in 10 minutes, Seilh 2026-11-15)
local rg = { nodes = {}, edges = {} }
for k = 0, 3 do
  local a = k * math.pi / 2
  rg.nodes['r' .. k] = { 20 * math.cos(a), 20 * math.sin(a), 0, 4 }
  rg.nodes['m' .. k] = { 20 * math.cos(a + math.pi / 4), 20 * math.sin(a + math.pi / 4), 0, 4 }
  rg.nodes['e' .. k] = { 170 * math.cos(a), 170 * math.sin(a), 0, 4 }
end
for k = 0, 3 do
  local nx = (k + 1) % 4
  rg.edges[#rg.edges + 1] = { 'r' .. k, 'm' .. k, 1, 1, 0, 8, 1, 4 }
  rg.edges[#rg.edges + 1] = { 'm' .. k, 'r' .. nx, 1, 1, 0, 8, 1, 4 }
  rg.edges[#rg.edges + 1] = { 'e' .. k, 'r' .. k, 0, 1, 1, 13.9, 1, 4 }
end
local FR = flow.new({ leftHand = false, seed = 3, paints = flow.SHARE, bodies = flow.BODY_SHARE })
flow.sync(FR, C.segments(rg, false), 0, 0, {}, {})
flow.setTarget(FR, 40)
for k = 1, 12000 do flow.step(FR, 0.05, 0, 0, {}, k * 0.05, nil) end
T.check((FR.stats.gaveUp or 0) == 0 and FR.count >= 35, 'a roundabout flows (' .. tostring(FR.stats.gaveUp) .. ' given up)')

-- the same roundabout with its arms split into a one-way entry and exit where they meet the ring (as the graph has
-- them): the ring still has the way
local rf = { nodes = {}, edges = {} }
for k = 0, 3 do
  local a = k * math.pi / 2
  rf.nodes['r' .. k] = { 20 * math.cos(a), 20 * math.sin(a), 0, 4 }
  rf.nodes['x' .. k] = { 20 * math.cos(a + 0.35), 20 * math.sin(a + 0.35), 0, 4 }   -- (the exit, past the entry)
  rf.nodes['m' .. k] = { 20 * math.cos(a + math.pi / 4), 20 * math.sin(a + math.pi / 4), 0, 4 }
  rf.nodes['j' .. k] = { 45 * math.cos(a + 0.2), 45 * math.sin(a + 0.2), 0, 4 }
  rf.nodes['e' .. k] = { 170 * math.cos(a), 170 * math.sin(a), 0, 4 }
end
for k = 0, 3 do
  local nx = (k + 1) % 4
  rf.edges[#rf.edges + 1] = { 'r' .. k, 'x' .. k, 1, 1, 0, 8, 1, 4 }
  rf.edges[#rf.edges + 1] = { 'x' .. k, 'm' .. k, 1, 1, 0, 8, 1, 4 }
  rf.edges[#rf.edges + 1] = { 'm' .. k, 'r' .. nx, 1, 1, 0, 8, 1, 4 }
  rf.edges[#rf.edges + 1] = { 'j' .. k, 'r' .. k, 1, 1, 0, 8, 1, 3 }       -- the entry, one way
  rf.edges[#rf.edges + 1] = { 'x' .. k, 'j' .. k, 1, 1, 0, 8, 1, 3 }       -- the exit, one way
  rf.edges[#rf.edges + 1] = { 'e' .. k, 'j' .. k, 0, 1, 1, 13.9, 1, 4 }
end
local FF = flow.new({ leftHand = false, seed = 4, paints = flow.SHARE, bodies = flow.BODY_SHARE })
flow.sync(FF, C.segments(rf, false), 0, 0, {}, {})
flow.setTarget(FF, 40)
for k = 1, 12000 do flow.step(FF, 0.05, 0, 0, {}, k * 0.05, nil) end
T.check((FF.stats.gaveUp or 0) <= 2, 'a roundabout with one-way entries flows (' .. tostring(FF.stats.gaveUp) .. ' given up)')
T.check(C.flowRoom(FF) > 0 and C.flowRoom(FF) < 1000, 'room by lane length (' .. C.flowRoom(FF) .. ')')

-- the game's signals on the traffic: a red on the edge into a junction stops its link's cars there
local sg = { nodes = { a = { 0, 0, 0, 4 }, b = { 60, 0, 0, 4 }, c = { 120, 0, 0, 4 }, n = { 120, 80, 0, 4 }, s = { 120, -80, 0, 4 },
  d = { 200, 0, 0, 4 } },
  edges = { { 'a', 'b', 0, 1, 1, 13.9, 1, 4 }, { 'b', 'c', 0, 1, 1, 13.9, 1, 4 }, { 'c', 'd', 0, 1, 1, 13.9, 1, 4 },
            { 'n', 'c', 0, 1, 1, 13.9, 1, 4 }, { 'c', 's', 0, 1, 1, 13.9, 1, 4 } } }
local segsS = C.segments(sg, false)
local FS = flow.new({ leftHand = false, seed = 8, paints = flow.SHARE, bodies = flow.BODY_SHARE })
flow.sync(FS, segsS, 60, 0, {}, {})
local red = { action = 2 }
local got = C.attachSignals(FS, segsS, { b = { c = { red } } })
local lit, litDir = 0, nil
for _, link in pairs(FS.links) do if type(link.light) == 'table' and link.light.bne == red then lit = lit + 1; litDir = link.to end end
T.check(got == 1 and lit == 1 and litDir == 'c', 'the signal on b -> c lights the link into c only')
T.check(flow.lightOf(FS, (function() for _, l in pairs(FS.links) do if type(l.light) == 'table' then return l end end end)()) == 'red', 'red is red')
red.action = 0
T.check(flow.lightOf(FS, (function() for _, l in pairs(FS.links) do if type(l.light) == 'table' then return l end end end)()) == 'green', 'its live state followed')
C.attachSignals(FS, segsS, {})
local still = false
for _, link in pairs(FS.links) do if type(link.light) == 'table' and link.light.bne then still = true end end
T.check(not still, 'a signal gone: its link unlit')

T.done('test_citylife')
