-- luajit mod/tests/test_net_hello.lua  (connected, the hello never answered: the connection starts over)
local dir = (arg[0]:match('^(.*)[/\\]') or '.')
local T = dofile(dir .. '/testlib.lua')
local connects, closes = 0, 0
local fake = { settimeout = function() end, setoption = function() end, send = function(_, d) return #d end,
  connect = function() connects = connects + 1; return 1 end, close = function() closes = closes + 1 end,
  getpeername = function() return '127.0.0.1' end, receive = function() return nil, 'timeout' end }
package.loaded.socket = { tcp = function() return fake end }
jsonEncode = function() return '{}' end
jsonDecode = function() return {} end
local N = require('beamngEarth/net')

local function run(seconds, step)
  local kinds = {}
  for _ = 1, math.floor(seconds / step) do
    for _, e in ipairs(N.update(step)) do kinds[e.type] = (kinds[e.type] or 0) + 1 end
  end
  return kinds
end
local got = run(5, 0.1)
T.check(connects == 1 and N.isConnected() and got._connected == 1, 'connected, once')
run(N.HELLO_TIMEOUT_S - 6, 0.1)
T.check(closes == 0, 'not given up on before the timeout')
got = run(1.5, 0.1)
T.check(closes == 1 and got._disconnected == 1, 'no answer: the connection closes: ' .. closes)
got = run(3, 0.1)
T.check(connects == 2 and got._connected == 1, 'then a new one, with a new hello: ' .. connects)
N.serverHello('s', 'id')
closes = 0
run(N.HELLO_TIMEOUT_S * 2, 0.1)
T.check(closes == 0 and N.isReady(), 'answered: it stays open')
T.done('net_hello')
