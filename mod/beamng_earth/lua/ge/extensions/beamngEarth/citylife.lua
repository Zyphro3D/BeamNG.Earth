-- City life: pedestrians on the sidewalks and traffic on the roads (the France fork; the user, 2026-11-15: "pedestrians, traffic ... light and
-- multiplayer-friendly"). The walkers are EarthNG's (walkers.lua, MIT, Alezx311): static pose shapes drawn on each
-- player's own computer, no physics, nothing sent to BeamMP, so a server carries no extra vehicles and every player
-- sees their own crowd. Cars pass through them (no collision); a walker waits at the kerb while a car comes.
--
-- The roads are the game's road map as BeamNG.Earth applies it (navgraph.lua hands the server's graph of the roads
-- round the car to map.reset: the map the game's traffic drives), read back with map.getMap() (M.graphOfMap) and
-- turned into walkers.lua's segments by M.segments: the edges chained through the nodes where a road simply goes on.
-- Read again each time the map is reloaded (onNavgraphReloaded: every 400 m driven, a teleport, a rebase), synced in a
-- coroutine, a few roads a frame. (navgraph.currentGraph() is dropped when the anchor moves, until the next graph:
-- the first try read it and found no roads, 2026-11-15.) Setting 'pedestrians' (off / few / normal / many); the Low preset turns them off.

local M = {}
local walkers = require('beamngEarth/walkers')
local flow = require('beamngEarth/flow')

M.DIR = '/levels/beamng_earth/art/shapes/beamngEarth/people/'
M.MATERIALS = M.DIR .. 'people.materials.json'
M.GROUP = 'beamngEarth_walkers'
M.COUNTS = { off = 0, few = 80, normal = 200, many = 400 }
-- the traffic (EarthNG's flow.lua: cars without physics, drawn here only; ~2.7 ms of the frame for 400, its docs)
M.FLOW_DIR = '/levels/beamng_earth/art/shapes/beamngEarth/flow/'
M.FLOW_MATERIALS = M.FLOW_DIR .. 'flow.materials.json'
M.FLOW_GROUP = 'beamngEarth_flow'
M.FLOW_COUNTS = { off = 0, light = 100, normal = 200, dense = 400 }
M.SIGNALS_S = 2.0          -- s between two looks at the game's signals for the traffic (attachSignals)
M.FLOW_LANE_M = 80         -- m of lane per car at most (400 cars on a suburb's few roads were one jam, Seilh 2026-11-15)
M.SIDE_GAP = 1.7          -- m: walkers.lua's SIDE_GAP (its walkers' line past the road's edge)
M.NO_WALK_SPEED = 99      -- m/s: a road walkers.lua gives no sidewalk (over its MAX_SPEED)
M.WALK_MAX_SPEED = 16.7   -- m/s: walkers.lua's MAX_SPEED (faster roads have no sidewalk: not measured)
M.KERB_MIN = 0.07          -- m: a step up this high from the road is a kerb (the drawn ones are 12-15 cm)
M.KERB_STEP = 0.25         -- m between the height reads across a street ...
M.KERB_MAX = 12            -- ... out to this far from its centre line
M.WALK_IN = 0.8            -- m past the kerb a walker walks
M.KERB_AT = { 0.25, 0.5, 0.75 }   -- where along a street its kerbs are looked for (two of three must agree)
M.REMEASURE_S = { 15, 45 } -- s after a sync the kerbs are measured again (the road tiles come after the graph)
M.CARS_S = 0.25           -- s between the reads of the vehicles (the walkers wait for moving ones at the kerb)
M.SYNC_MS = 2.0           -- ms of sidewalk building a frame (walkers.sync's pauses yield only past it: one yield per
                          -- pause took over a thousand frames for a town's 500 roads, the first try in game)

local W, V = nil, nil     -- walkers.lua's state and its view
local F, FV = nil, nil    -- flow.lua's state and its view
local flowCount, flowLeft = nil, nil
local flowRoom = math.huge    -- the cars the roads round have room for (M.flowRoom)
local flowSegs, signalsAt = nil, -math.huge
local syncCo = nil
local cars, carsAt = {}, -math.huge
local obstacles = {}       -- (flow.step's list, reused)
local clock = 0
local at = nil            -- (a vec3 reused for the ground reads)
local lastCount = nil
local lastAnchor = nil
local mapDirty = true      -- the game's road map reloaded since the last read
local mapSerial = 0
local sliceAt = 0          -- (the time this frame's slice of the sync started)
local remeasure, remeasureAt, lastSig = nil, nil, nil
local startSync

-- Pure: walkers.lua's segments ({[id] = {id, a, b, lanes, speed, half, drive, x, y, z, ...}}) of a road graph
-- ({nodes = {name = {x, y, z, r}}, edges = {{in, out, oneway, lanes_out, lanes_in, speed, drivability, radius ...}}}).
-- Edges are chained through a node with exactly two edges that agree on speed, width and lanes (a road going on),
-- so a sidewalk runs the street's length instead of stopping at every graph node.
function M.segments(g, forWalkers)
  local out = {}
  if type(g) ~= 'table' or type(g.nodes) ~= 'table' or type(g.edges) ~= 'table' then return out end
  local nodes, edges = g.nodes, g.edges
  local at = {}           -- node -> list of edge indices
  local function key(e)
    return string.format('%.1f|%.1f|%d|%d|%d|%.1f', tonumber(e[6]) or 0, tonumber(e[8]) or 0, tonumber(e[4]) or 0,
      tonumber(e[5]) or 0, tonumber(e[3]) or 0, tonumber(e[13]) or 0)
  end
  -- (a graph that says where sidewalks are drawn, edge[13] (navgraph.py edge[12]): walkers only there; an older one
  -- without it: every road under walkers' 60 km/h, as EarthNG)
  local knows = false
  for _, e in ipairs(edges) do if e[13] ~= nil then knows = true; break end end
  for i, e in ipairs(edges) do
    if nodes[e[1]] and nodes[e[2]] and e[1] ~= e[2] then
      at[e[1]] = at[e[1]] or {}
      at[e[2]] = at[e[2]] or {}
      table.insert(at[e[1]], i)
      table.insert(at[e[2]], i)
    end
  end
  local used = {}
  local function through(n, from)       -- the other edge at n that continues edge `from`, or nil
    local list = at[n]
    if not list or #list ~= 2 then return nil end
    local other = list[1] == from and list[2] or list[1]
    if other == from or used[other] or key(edges[other]) ~= key(edges[from]) then return nil end
    return other
  end
  local id = 0
  for i, e in ipairs(edges) do
    if not used[i] and at[e[1]] and nodes[e[1]] and nodes[e[2]] and e[1] ~= e[2] then
      used[i] = true
      -- grow both ways: names along the chain, from one end to the other
      local names = { e[1], e[2] }
      local cur, n = i, e[2]
      while true do
        local nxt = through(n, cur)
        if not nxt then break end
        used[nxt] = true
        local ne = edges[nxt]
        n = ne[1] == n and ne[2] or ne[1]
        if n == names[1] then break end           -- (a loop)
        names[#names + 1] = n
        cur = nxt
      end
      cur, n = i, e[1]
      while true do
        local nxt = through(n, cur)
        if not nxt then break end
        used[nxt] = true
        local ne = edges[nxt]
        n = ne[1] == n and ne[2] or ne[1]
        if n == names[#names] then break end
        table.insert(names, 1, n)
        cur = nxt
      end
      local fwd, back = tonumber(e[4]) or 1, tonumber(e[5]) or 0
      local lanes = string.rep('+', math.max(0, fwd)) .. string.rep('-', math.max(0, back))
      if lanes == '' then lanes = '+-' end
      id = id + 1
      local speed, half, walk = tonumber(e[6]) or 13, tonumber(e[8]) or 3.5, tonumber(e[13]) or 0
      if knows and forWalkers ~= false then
        if walk > 0 then
          -- the walker in the middle of the drawn sidewalk, `walk` m off the centre line (walkers.lua puts it SIDE_GAP
          -- past the road's edge, the half width or its lanes' reach: no lanes, the half width alone)
          half, lanes = math.max(0.5, walk - M.SIDE_GAP), ''
        else
          speed = M.NO_WALK_SPEED       -- (no sidewalk drawn: no walkers; the road still cuts the others' sidewalks)
        end
      end
      local s = { id, names[1], names[#names], lanes, speed, half, tonumber(e[7]) or 1 }
      s.names = names                 -- (the chain's nodes: the game's signals on its edges, attachSignals)
      for _, name in ipairs(names) do
        local p = nodes[name]
        s[#s + 1], s[#s + 2], s[#s + 3] = tonumber(p[1]) or 0, tonumber(p[2]) or 0, tonumber(p[3]) or 0
      end
      out[id] = s
    end
  end
  return out
end

-- Pure: the road graph ({nodes = {name = {x, y, z, r}}, edges = {{in, out, oneway, lanes_out, lanes_in, speed, driv,
-- radius}}}) of the game map's nodes ({name = {pos = {x, y, z}, radius, links = {other = edge}}}, navgraph.toNodes:
-- an edge {inNode, outNode, oneWay, lanes '--++', speedLimit, drivability, inRadius} shared by both its nodes).
function M.graphOfMap(mapNodes)
  local g = { nodes = {}, edges = {} }
  if type(mapNodes) ~= 'table' then return g end
  local seen = {}
  for name, n in pairs(mapNodes) do
    local p = n.pos
    if p then g.nodes[name] = { p.x, p.y, p.z, n.radius or 3 } end
  end
  for name, n in pairs(mapNodes) do
    for other, e in pairs(n.links or {}) do
      if type(e) == 'table' and not seen[e] then
        seen[e] = true
        local a, b = e.inNode or name, e.outNode or other
        local lanes = type(e.lanes) == 'string' and e.lanes or ''
        local fwd, back = select(2, lanes:gsub('%+', '')), select(2, lanes:gsub('%-', ''))
        if fwd + back == 0 then fwd, back = 1, e.oneWay and 0 or 1 end
        g.edges[#g.edges + 1] = { a, b, e.oneWay and 1 or 0, fwd, back, tonumber(e.speedLimit) or 13,
          tonumber(e.drivability) or 1, tonumber(e.inRadius) or n.radius or 3, nil, nil, nil, nil, e.walk }
      end
    end
  end
  return g
end

-- Pure (height(x, y, z) -> the surface's height or nil): each side's sidewalk middle of segment s (walkers.lua's
-- side 1 is the right of its line, -1 the left), measured on the road as drawn: the first step up of KERB_MIN or more
-- going out from the centre line, at KERB_AT of its length; a side where two of the three agree within a metre gets
-- that kerb + WALK_IN, else false (no kerb: a street drawn wider than the graph says, flush ground, a square).
-- {[1] = m or false, [-1] = m or false}, or nil when nothing could be read. (The France fork, 2026-11-15: by the
-- graph's widths alone the walkers of Toulouse's rue de Metz walked on the carriageway, 2 m short of the kerb.)
function M.measureSides(s, height)
  local pts = {}
  for k = 8, #s - 2, 3 do pts[#pts + 1] = { s[k], s[k + 1], s[k + 2] } end
  if #pts < 2 then return nil end
  local cum = { 0 }
  for i = 2, #pts do
    cum[i] = cum[i - 1] + math.sqrt((pts[i][1] - pts[i - 1][1]) ^ 2 + (pts[i][2] - pts[i - 1][2]) ^ 2)
  end
  local L = cum[#pts]
  if L < 4 then return nil end
  local found = { [1] = {}, [-1] = {} }
  local read = false
  for _, f in ipairs(M.KERB_AT) do
    local d = f * L
    local i = 2
    while i < #pts and cum[i] < d do i = i + 1 end
    local a, b = pts[i - 1], pts[i]
    local seg = cum[i] - cum[i - 1]
    local t = seg > 1e-6 and (d - cum[i - 1]) / seg or 0
    local x, y, z = a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t
    local dx, dy = (b[1] - a[1]) / math.max(seg, 1e-6), (b[2] - a[2]) / math.max(seg, 1e-6)
    local rx, ry = dy, -dx                      -- (the right: walkers.lua's offsetLine)
    local h0 = height(x, y, z)
    if h0 then
      read = true
      for _, side in ipairs({ 1, -1 }) do
        local k = 1
        while k * M.KERB_STEP <= M.KERB_MAX do
          local o = k * M.KERB_STEP
          local h = height(x + side * rx * o, y + side * ry * o, z)
          if h and h - h0 >= M.KERB_MIN then
            table.insert(found[side], o)
            break
          end
          k = k + 1
        end
      end
    end
  end
  if not read then return nil end
  local out = {}
  for _, side in ipairs({ 1, -1 }) do
    local list = found[side]
    table.sort(list)
    local pick = false
    for i = 1, #list - 1 do
      if list[i + 1] - list[i] <= 1.0 then pick = list[i + 1] + M.WALK_IN; break end
    end
    out[side] = pick
  end
  return out
end

-- Pure: the walkers asked for a setting value.
function M.countOf(value) return M.COUNTS[value] or M.COUNTS.normal end

local function group()
  local g = scenetree.findObject(M.GROUP)
  if g then return g end
  pcall(loadJsonMaterialsFile, M.MATERIALS)
  g = createObject('SimGroup')
  g.canSave = false
  g:registerObject(M.GROUP)
  if scenetree.MissionGroup then scenetree.MissionGroup:addObject(g) end
  return g
end

local function flowGroup()
  local g = scenetree.findObject(M.FLOW_GROUP)
  if g then return g end
  pcall(loadJsonMaterialsFile, M.FLOW_MATERIALS)
  g = createObject('SimGroup')
  g.canSave = false
  g:registerObject(M.FLOW_GROUP)
  if scenetree.MissionGroup then scenetree.MissionGroup:addObject(g) end
  return g
end

local function flowShapes()
  local list = {}
  for b, body in ipairs(flow.BODIES) do
    list[b] = {}
    for k, paint in ipairs(flow.PAINTS) do list[b][k] = M.FLOW_DIR .. 'car_' .. body .. '_' .. paint .. '.dae' end
  end
  return list
end

-- Pure: how many cars flow.lua's roads have room for: their lanes' length over FLOW_LANE_M.
function M.flowRoom(F)
  local m = 0
  for _, link in pairs(F.links or {}) do m = m + (link.len or 0) * (link.lanes or 1) end
  return math.floor(m / M.FLOW_LANE_M)
end

-- Pure: the cars asked for a setting value.
function M.flowCountOf(value) return M.FLOW_COUNTS[value] or M.FLOW_COUNTS.normal end

-- The game's traffic signals on the traffic's links (core_trafficSignals.getMapNodeSignals: {n1 = {n2 = {entry ...}}},
-- an entry's action its live state): a link whose chain runs over a signal's edge n1 -> n2 in its own direction stops
-- at its end for that signal's red (flow.lightOf). Again every SIGNALS_S: the signals are made a frame after the map
-- (navgraph.lua) and made again with it. Returns how many links got one.
function M.attachSignals(F, segs, sigs)
  if not F or type(sigs) ~= 'table' then return 0 end
  local n = 0
  for _, link in pairs(F.links) do
    local seg = segs and segs[tonumber(link.sid) or link.sid]
    local names = seg and seg.names
    local found = nil
    if names then
      local a, b, st = 1, #names, 1
      if link.dir < 0 then a, b, st = #names, 1, -1 end
      for i = a, b - st, st do
        local row = sigs[names[i]]
        local list = row and row[names[i + st]]
        if list and list[1] then found = list[1] end
      end
    end
    if found then
      if not (type(link.light) == 'table' and link.light.bne == found) then link.light = { bne = found } end
      n = n + 1
    elseif type(link.light) == 'table' and link.light.bne then
      link.light = nil
    end
  end
  return n
end

local function flowClear()
  if F then pcall(flow.clear, F) end
  F, FV, flowCount = nil, nil, nil
  local g = scenetree and scenetree.findObject and scenetree.findObject(M.FLOW_GROUP)
  if g then pcall(function() g:delete() end) end
end

local function shapes()
  local list = {}
  for k = 0, walkers.POSES - 1 do list[k + 1] = M.DIR .. 'ped_walk' .. k .. '.dae' end
  list[walkers.POSES + 1] = M.DIR .. 'ped_stand.dae'
  return list
end

local function ground(x, y, z)
  at = at or vec3(0, 0, 0)
  at.x, at.y, at.z = x, y, z + 2
  local h = be:getSurfaceHeightBelow(at)
  if h and h > -1e6 then return h end
end

-- the vehicles with physics, {x, y, vx, vy} each (tables reused), every CARS_S
local function readCars()
  if clock - carsAt < M.CARS_S then return cars end
  carsAt = clock
  local n = 0
  local count = be and be.getObjectCount and be:getObjectCount() or 0
  for i = 0, count - 1 do
    local o = be:getObject(i)
    if o and o.getPosition then
      local okP, p = pcall(o.getPosition, o)
      local okV, v = pcall(o.getVelocity, o)
      if okP and p then
        n = n + 1
        local c = cars[n] or {}
        c.x, c.y = p.x, p.y
        c.vx, c.vy = okV and v and v.x or 0, okV and v and v.y or 0
        cars[n] = c
      end
    end
  end
  for k = n + 1, #cars do cars[k] = nil end
  return cars
end

local function focus()
  local veh = getPlayerVehicle and getPlayerVehicle(0)
  if veh then
    local p = veh:getPosition()
    return p.x, p.y
  end
  local ok, p = pcall(core_camera.getPosition)
  if ok and p then return p.x, p.y end
end

function M.reset()
  flowClear()
  remeasure, remeasureAt, lastSig = nil, nil, nil
  if W then pcall(walkers.clear, W, V) end
  W, V, syncCo, lastCount = nil, nil, nil, nil
  mapDirty = true
  local g = scenetree and scenetree.findObject and scenetree.findObject(M.GROUP)
  if g then pcall(function() g:delete() end) end
end

local function slice()
  if (os.clock() - sliceAt) * 1000 >= M.SYNC_MS then coroutine.yield() end
end

-- The sidewalks' kerbs measured (a few roads a frame), then walkers.lua's sync; a second pass whose kerbs came out the
-- same as the last (sig) keeps the walkers as they are.
function startSync(segs, fx, fy, onlyIfChanged)  -- (local: declared above)
  return coroutine.create(function()
    local sig = {}
    for id, sg in pairs(segs) do
      if sg[5] <= M.WALK_MAX_SPEED then
        local got = M.measureSides(sg, ground)
        if got then
          sg.off = got
          sig[#sig + 1] = id .. ':' .. tostring(got[1]) .. ',' .. tostring(got[-1])
        else
          sg.off = { [1] = false, [-1] = false }    -- (nothing drawn there yet: no walkers until it is measured)
          sig[#sig + 1] = id .. ':-'
        end
      end
      slice()
    end
    table.sort(sig)
    local key = table.concat(sig, ';')
    if onlyIfChanged and key == lastSig then return end
    lastSig = key
    walkers.sync(W, segs, fx, fy, slice)
  end)
end

-- The game's road map was reloaded (core.lua onNavgraphReloaded): read it again.
function M.onMapReloaded() mapDirty, mapSerial = true, mapSerial + 1 end

-- Every frame (core.lua, our level): the count from the setting, the road map synced when it changed, the walkers
-- moved and drawn. anchorId: the world origin (a new one: the walkers start again); paused: the game paused.
function M.update(dtReal, anchorId, setting, paused, trafficSetting, leftHand)
  local want, wantCars = M.countOf(setting), M.flowCountOf(trafficSetting)
  if want == 0 and wantCars == 0 then
    if W or F then M.reset() end
    return
  end
  clock = clock + (dtReal or 0)
  local fx, fy = focus()
  if not fx then return end
  -- (a new anchor, a rebase or a teleport: the world's coordinates moved, everyone starts again in the new frame)
  if (W or F) and anchorId ~= lastAnchor then M.reset() end
  lastAnchor = anchorId
  -- (nothing made before the game has a road map: no shapes loaded for a level without roads, nor in a test's stubs)
  if not W and not F then
    local m = map and map.getMap and map.getMap()
    if not (m and type(m.nodes) == 'table' and next(m.nodes)) then return end
  end
  -- pedestrians
  if want == 0 and W then
    pcall(walkers.clear, W, V)
    W, V, lastCount = nil, nil, nil
    local g = scenetree.findObject(M.GROUP)
    if g then pcall(function() g:delete() end) end
  elseif want > 0 and not W then
    W = walkers.new({ seed = 4242 })
    V = walkers.view(W, shapes(), group())
    mapDirty = true
  end
  if W and want ~= lastCount then walkers.setTarget(W, want); lastCount = want end
  -- traffic (a new side of the road: started again)
  if F and (wantCars == 0 or flowLeft ~= (leftHand == true)) then flowClear() end
  if wantCars > 0 and not F then
    flowLeft = leftHand == true
    F = flow.new({ leftHand = flowLeft, seed = 777, paints = flow.SHARE, bodies = flow.BODY_SHARE })
    FV = flow.view(F, flowShapes(), flowGroup())
    mapDirty = true
  end
  if F and wantCars ~= flowCount then flow.setTarget(F, math.min(wantCars, flowRoom)); flowCount = wantCars end
  -- the road map: read again when it changed; the traffic syncs at once, the sidewalks over a few frames
  if mapDirty and not syncCo then
    mapDirty = false
    local m = map and map.getMap and map.getMap()
    local g = M.graphOfMap(m and m.nodes)
    local any = next(g.edges) ~= nil
    if not any then
      mapDirty = true           -- (no roads yet: the map comes with the level's first graph)
    else
      if F then
        flowSegs = M.segments(g, false)
        local ok, err = pcall(flow.sync, F, flowSegs, fx, fy, {}, {})
        signalsAt = -math.huge
        if not ok then log('W', 'beamngEarth', 'traffic: sync: ' .. tostring(err)) end
        flowRoom = M.flowRoom(F)
        flow.setTarget(F, math.min(wantCars, flowRoom))
      end
      if W then
        local segs = M.segments(g)
        remeasure = 1
        remeasureAt = clock + M.REMEASURE_S[1]
        syncCo = startSync(segs, fx, fy)
      end
    end
  end
  if syncCo then
    sliceAt = os.clock()
    local ok, err = coroutine.resume(syncCo)
    if not ok then log('W', 'beamngEarth', 'pedestrians: sync: ' .. tostring(err)); syncCo = nil end
    if syncCo and coroutine.status(syncCo) == 'dead' then syncCo = nil end
  end
  -- the kerbs again once the road tiles are in (REMEASURE_S after a sync): the walkers move only if they changed
  if W and not syncCo and not mapDirty and remeasure and clock >= remeasureAt then
    local m = map and map.getMap and map.getMap()
    local segs = M.segments(M.graphOfMap(m and m.nodes))
    remeasure = remeasure + 1
    remeasureAt = M.REMEASURE_S[remeasure] and (clock + M.REMEASURE_S[remeasure] - M.REMEASURE_S[remeasure - 1]) or nil
    if not remeasureAt then remeasure = nil end
    if next(segs) then syncCo = startSync(segs, fx, fy, true) end
  end
  if paused then return end
  local cars = readCars()
  if F then
    -- the obstacles: the vehicles with physics and the walkers crossing a road
    local obs = obstacles
    for k = 1, #cars do obs[k] = cars[k] end
    local n = #cars
    if W then
      for k = 1, W.onRoadN or 0 do
        local w = W.onRoad[k]
        n = n + 1
        local o = obs[n] or {}
        o.x, o.y, o.vx, o.vy = w.x, w.y, 0, 0
        obs[n] = o
      end
    end
    for k = n + 1, #obs do obs[k] = nil end
    if clock - signalsAt >= M.SIGNALS_S then
      signalsAt = clock
      local ts = rawget(_G, 'core_trafficSignals')
      local ok, sigs = pcall(function() return ts and ts.getMapNodeSignals and ts.getMapNodeSignals() end)
      if ok and sigs then M.attachSignals(F, flowSegs, sigs) end
    end
    flow.step(F, dtReal, fx, fy, obs, clock, nil)
    flow.draw(F, FV, fx, fy)
  end
  if W then
    walkers.step(W, dtReal, fx, fy, cars, F and F.cars or nil)
    walkers.draw(W, V, fx, fy, ground)
  end
end

-- (tests, the console) asked, walking, room
function M.counts() if not W then return 0, 0, 0 end return walkers.counts(W) end
-- (probes) whether the sidewalks are being built, and how many metres of them there are
function M.debug() return { syncing = syncCo ~= nil, dirty = mapDirty, pieces = W and #W.list or 0, length = W and W.length or 0,
  cars = F and F.count or 0, links = F and F.linkList and #F.linkList or 0 } end

return M
