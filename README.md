# LongiTrack-backend

[![arXiv](https://img.shields.io/badge/arXiv-2605.23118-b31b1b.svg)](https://arxiv.org/abs/2605.23118)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A standalone GPU service for point-prompted longitudinal lesion tracking: model loading,
[uniGradICON](https://github.com/uncbiag/uniGradICON) registration and
[LongiSeg](https://github.com/MIC-DKFZ/LongiSeg) segmentation, served over a Unix socket
(local) or TCP (remote) so a client process never has to import torch itself.

It has no GUI of its own and is not tied to one. Any viewer that speaks the protocol below
can drive it, which is the point: the GPU work lives here, once, and a front end for MITK,
3D Slicer or anything else is a client against the same interface.

## Install

LongiTrack-backend is **Linux only**. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install. The default install includes the Linux GPU backend:

```bash
git clone https://github.com/MIC-DKFZ/LongiTrack-backend.git
cd LongiTrack-backend
uv sync
```

`uv sync --no-default-groups` installs the same package without torch or LongiSeg. That cannot
serve, but it is all a client machine needs to talk to a server that can.


## Running it

A client running on the same machine spawns its own server on a Unix socket, so there is
nothing to start by hand. Run it yourself to serve remote clients, or to point a socket client
at it for debugging:

```bash
uv run longitrack-backend --socket /tmp/longitrack.sock     # Unix socket, for debugging
uv run longitrack-backend                                   # remote, TCP, Ed25519-authenticated
```

The TCP listener binds to every interface on port **8765**, which is fixed. It refuses to start
without an explicit `y` (or `--confirm-tcp-listener yes`), because it authenticates callers by
Ed25519 key but does not encrypt the connection. Put it behind an SSH tunnel or a loopback-only
bind for anything beyond a trusted LAN. On start it prints a `CONNECT <host>:8765` line; that whole
line can be pasted straight into the client's address field.

### Running it for other people

Serving remote clients is an administrator's job, and it has to be done once on the server before
anyone can connect:

1. **Start the TCP listener once, interactively.** The server reads public keys from
   `/srv/longitrack/authorized_keys`; on that first start it asks for your `sudo` password and
   creates the `longitrack` group and that directory for you. Reconnect your SSH session
   afterwards so the new group membership applies.
2. **Add each approved user's SSH account to the `longitrack` group.** Without this their key
   cannot be installed and they will not get past the handshake.
3. **Give them the server address.**

Each user then registers their own key from their own machine:

```bash
uv run create_and_push_remote_id --name gpu-cluster --ssh them@this-server --owner them
```

`--owner` is the person the key belongs to, and is recorded next to it on the server so you can
tell whose is whose. The private key never leaves their machine; only the public half is
installed here. Key changes take effect without restarting the backend.


## Building a client against it

Every request is a length-prefixed JSON frame; large payloads, like scans and masks, follow as
raw binary frames. `protocol.py` has the exact framing. `health` needs no authentication and
identifies the service; a TCP connection then needs one `authenticate` call signing the server's
challenge with an authorized Ed25519 key, a Unix socket needs none.

Past that handshake, the interface a client drives is:

- **`initialize`** -- load a model folder (local or Hugging Face Hub) and pick a device.
- **`load_scan(s)`** / **`preload_registration_scans`** -- prepare one or more scans ahead of use.
- **`propagate`** -- map a point from a baseline scan to a follow-up scan via registration.
- **`track`** -- given a point on each scan, segment that lesion in both.
- **`export`** -- write masks and a JSON record of the prompts behind them to a folder.
- **`open_session`** / **`upload_scan`** / **`materialize_model`** -- a remote connection's
  content-addressed file transfer, so scans and model folders only cross the network once.

## The model

A LongiSeg model folder (`dataset.json`, `plans.json`, `fold_*/checkpoint_final.pth`),
either on the Hugging Face Hub or on disk:

```bash
uv run longitrack-model info                       # what is in the resolved model?
uv run longitrack-model download --folds 0         # pull one fold
uv run longitrack-model upload /path/to/folder -r owner/name
uv run longitrack-model sample -p PanTrack_029 -i 2  # a PanTrack scan pair for testing
```

`$LONGITRACK_MODEL_DIR`, else `$LONGITRACK_HF_REPO`, else `ykirchhoff/LongiTrack`
picks the default when no source is given.

New checkpoints are published as their own `LongiTrack_v<version>` folder alongside older
ones, so a repo (or a local folder with several such subfolders) can hold more than one.
Without `--version`, the newest is used; add `--version 1.0`, or append `@1.0` directly to
a repo id or path (e.g. `ykirchhoff/LongiTrack@1.0`), to pin an older one.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Hooks run `ruff --fix` and a few cheap file checks on every commit, and the tests on every push:

```bash
uv tool install pre-commit
pre-commit install -t pre-commit -t pre-push
```

## Citing

This backend implements **Exploiting Longitudinal Context in Clinician-Verified
Interactive Lesion Tracking** ([arXiv:2605.23118](https://arxiv.org/abs/2605.23118)). Please cite it
together with [LongiSeg](https://doi.org/10.1007/978-3-031-72069-7_7) for the segmentation
framework and [uniGradICON](https://arxiv.org/abs/2403.05780) for the registration.

```bibtex
@article{kirchhoff2026longitrack,
  title   = {Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking},
  author  = {Kirchhoff, Yannick and Rokuss, Maximilian and Mertens, Daniel Philipp and
             F{\"u}ller, David and Hamm, Benjamin and Schreyer, Andreas and Ritter, Oliver and
             Maier-Hein, Klaus},
  journal = {arXiv preprint arXiv:2605.23118},
  year    = {2026}
}

@inproceedings{rokuss2024longitudinal,
  title     = {Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting},
  author    = {Rokuss, Maximilian R. and Kirchhoff, Yannick and Roy, Saikat and Kovacs, Balint and
               Ulrich, Constantin and Wald, Tassilo and Zenk, Maximilian and Denner, Stefan and
               Isensee, Fabian and Vollmuth, Philipp and Kleesiek, Jens and Maier-Hein, Klaus},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages     = {64--74},
  year      = {2024},
  publisher = {Springer}
}

@inproceedings{tian2024unigradicon,
  title     = {uniGradICON: A Foundation Model for Medical Image Registration},
  author    = {Tian, Lin and Greer, Hastings and Kwitt, Roland and Vialard, Fran{\c{c}}ois-Xavier and
               San Jos{\'e} Est{\'e}par, Ra{\'u}l and Bouix, Sylvain and Rushmore, Richard and
               Niethammer, Marc},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages     = {749--760},
  year      = {2024},
  publisher = {Springer}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
