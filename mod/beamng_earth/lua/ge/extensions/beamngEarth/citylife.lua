-- City life: pedestrians on the sidewalks (the France fork; the user, 2026-11-15: "pedestrians, traffic ... light and
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

M.DIR = '/levels/beamng_earth/art/shapes/beamngEarth/people/'
M.MATERIALS = M.DIR .. 'people.materials.json'
M.GROUP = 'beamngEarth_walkers'
M.COUNTS = { off = 0, few = 80, normal = 200, many = 400 }
M.CARS_S = 0.25           -- s between the reads of the vehicles (the walkers wait for moving ones at the kerb)
M.SYNC_YIELD = 1          -- coroutine.yield at each pause walkers.sync asks for (a few roads a frame)

local W, V = nil, nil     -- walkers.lua's state and its view
local syncCo = nil
local cars, carsAt = {}, -math.huge
local clock = 0
local at = nil            -- (a vec3 reused for the ground reads)
local lastCount = nil
local lastAnchor = nil
local mapDirty = true      -- the game's road map reloaded since the last read
local mapSerial = 0

-- Pure: walkers.lua's segments ({[id] = {id, a, b, lanes, speed, half, drive, x, y, z, ...}}) of a road graph
-- ({nodes = {name = {x, y, z, r}}, edges = {{in, out, oneway, lanes_out, lanes_in, speed, drivability, radius ...}}}).
-- Edges are chained through a node with exactly two edges that agree on speed, width and lanes (a road going on),
-- so a sidewalk runs the street's length instead of stopping at every graph node.
function M.segments(g)
  local out = {}
  if type(g) ~= 'table' or type(g.nodes) ~= 'table' or type(g.edges) ~= 'table' then return out end
  local nodes, edges = g.nodes, g.edges
  local at = {}           -- node -> list of edge indices
  local function key(e)
    return string.format('%.1f|%.1f|%d|%d|%d', tonumber(e[6]) or 0, tonumber(e[8]) or 0, tonumber(e[4]) or 0,
      tonumber(e[5]) or 0, tonumber(e[3]) or 0)
  end
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
      local s = { id, names[1], names[#names], lanes, tonumber(e[6]) or 13, tonumber(e[8]) or 3.5,
        tonumber(e[7]) or 1 }
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
          tonumber(e.drivability) or 1, tonumber(e.inRadius) or n.radius or 3 }
      end
    end
  end
  return g
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
  if W then pcall(walkers.clear, W, V) end
  W, V, syncCo, lastCount = nil, nil, nil, nil
  mapDirty = true
  local g = scenetree and scenetree.findObject and scenetree.findObject(M.GROUP)
  if g then pcall(function() g:delete() end) end
end

-- The game's road map was reloaded (core.lua onNavgraphReloaded): read it again.
function M.onMapReloaded() mapDirty, mapSerial = true, mapSerial + 1 end

-- Every frame (core.lua, our level): the count from the setting, the road map synced when it changed, the walkers
-- moved and drawn. anchorId: the world origin (a new one: the walkers start again); paused: the game paused.
function M.update(dtReal, anchorId, setting, paused)
  local want = M.countOf(setting)
  if want == 0 then
    if W then M.reset() end
    return
  end
  clock = clock + (dtReal or 0)
  local fx, fy = focus()
  if not fx then return end
  -- (a new anchor, a rebase or a teleport: the world's coordinates moved, the walkers start again in the new frame)
  if W and anchorId ~= lastAnchor then M.reset() end
  lastAnchor = anchorId
  if not W then
    W = walkers.new({ seed = 4242 })
    V = walkers.view(W, shapes(), group())
    mapDirty = true
  end
  if want ~= lastCount then walkers.setTarget(W, want); lastCount = want end
  -- the road map: read again when it changed, synced over a few frames
  if mapDirty and not syncCo then
    mapDirty = false
    local m = map and map.getMap and map.getMap()
    local segs = M.segments(M.graphOfMap(m and m.nodes))
    if next(segs) then
      syncCo = coroutine.create(function()
        walkers.sync(W, segs, fx, fy, M.SYNC_YIELD and coroutine.yield or nil)
      end)
    else
      mapDirty = true           -- (no roads yet: the map comes with the level's first graph)
    end
  end
  if syncCo then
    local ok, err = coroutine.resume(syncCo)
    if not ok then log('W', 'beamngEarth', 'pedestrians: sync: ' .. tostring(err)); syncCo = nil end
    if syncCo and coroutine.status(syncCo) == 'dead' then syncCo = nil end
  end
  if paused then return end
  walkers.step(W, dtReal, fx, fy, readCars(), nil)
  walkers.draw(W, V, fx, fy, ground)
end

-- (tests, the console) asked, walking, room
function M.counts() if not W then return 0, 0, 0 end return walkers.counts(W) end

return M
