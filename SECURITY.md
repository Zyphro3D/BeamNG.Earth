# Security

## Reporting a vulnerability

Please report it privately through this repository's **Security > Report a vulnerability** (a GitHub private
advisory), not in a public issue. Describe what is affected, how to reproduce it, and the version you used.

## Verifying a download

Every release publishes `SHA256SUMS` beside its files and a signed build provenance attestation for each of them
(GitHub's attestations, signed through Sigstore). Either check works:

```
sha256sum -c SHA256SUMS --ignore-missing                     # the file you downloaded is the one published
gh attestation verify BeamNG.Earth-<version>.exe -R <owner>/<repository>   # it was built by this repository's
                                                                           # release workflow, from its source
```

The release builds install their Python packages from lock files with hashes (`server/requirements-build.lock`,
`pip --require-hashes`), pin every GitHub action by commit, and check the tools they download (appimagetool, the
BeamMP server) against their SHA-256.

## What the app exposes

- **The world server** listens on `127.0.0.1` only (the game on the same machine talks to it).
- **The companion feed** (a phone or second screen following the car) listens on `127.0.0.1` by default;
  `--companion-feed lan` opens it to the local network, and `off` closes it. On the LAN it shares the car's position.
- **The compute helper** (`server/peerwork.py serve`, shared far-tile building, off unless started by hand) listens
  on all interfaces when run. Its protocol carries JSON and numpy arrays without pickle, checks every job's type and
  ranges, and refuses a peer of another version.
- **Map data, elevation and imagery** are fetched over HTTPS with certificate verification (the system's store on
  Windows). Remote file names from the map data's index are escaped in every database query.

## Accepted risks

- **Local cache files.** A few files the server writes and reads back in its own cache and staging folders use Python's
  pickle (a road geotile's skin inputs, the parsed airport database). Loading a tampered one would run code, but
  writing there needs the same user's rights, which already allow far more. They are never sent to or received from
  another machine.
- **The game's own MCP server** (BeamNG.drive's Settings > Advanced) runs any Lua it is sent from this machine. It is
  the game's, used by the development tools in `tools/`; turn it off when you are not using them.
