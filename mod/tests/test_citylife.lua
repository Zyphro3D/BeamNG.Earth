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

T.done('test_citylife')
