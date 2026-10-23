# LOC3D private GPU workbench

This dependency-free Cloudflare Worker serves the LOC3D point-cloud UI and
queues work for an outbound GPU worker. Checkpoints stay on the GPU machine.
The source is intended for an owner-private Site; the hosting platform must
protect every browser route. Do not publish this app with anonymous access.

Configure the `BUCKET` R2 binding and the `DELIMIT3D_BRIDGE_TOKEN` server secret.
The GPU worker uses the same secret and the platform's private service-access
header. Keep both credentials outside source control and browser code.

Start the worker with `python -m delimit3d.inference.remote_worker --help` in the
LOC3D CUDA environment. Provide its HTTPS URL, model registry, example scene,
and output directory. GPU availability and the allocation deadline appear in
the UI. A disconnected GPU cannot process requests.

On clusters that require an outbound proxy, set `HTTPS_PROXY` and `HTTP_PROXY`
in the private GPU launch environment, and exclude `localhost`, `127.0.0.1`,
and `::1` with `NO_PROXY`. Use the cluster's documented proxy; keep TLS
certificate verification enabled. These settings do not expose the local GPU
server to the internet.

The queue holds at most eight pending requests, with one active GPU worker.
Uploads are limited to 128 MiB; requests expire after 30 minutes. Downloadable files
expire after 24 hours and are removed by bounded cleanup on later heartbeats.
Cleanup pauses while no GPU worker is connected. Scene uploads are removed when
their processing completes; the GPU keeps at most two loaded scenes.

Run checks and build with Node 22 or newer:

```bash
node --test tests/bridge.test.mjs
bash scripts/build.sh
node scripts/validate-artifact.mjs
```

The build copies only `worker/index.js` and `.openai/hosting.json` into `dist`.
The UI is embedded in the Worker. After changing UI source, run:

```bash
python scripts/embed-assets.py /path/to/delimit3d/src/delimit3d/inference/static
```

Publish the exact tested source and generated artifact through the Sites
source/build workflow. For a new Site, let registration set `project_id`; never
copy another deployment's identity. Preserve the owner-private access policy.
