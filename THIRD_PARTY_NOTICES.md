# Third-party notices

Original Delimit3D code retains the status in `LICENSE.md`. External projects, copied code and public pretrained weights keep their own licenses. Weights and full backbone/teacher checkouts are external to this package.

## Copied implementations

- **AGILE3D**: official dataset, simulated clicks, evaluation and decoder-derived code at revision `b73638da41edbabe52a1b578d52ddeb8fa552173`. Copyright Yuanwen Yue / ETH Zurich, MIT; full text in `third_party/AGILE3D.LICENSE`. Namespaced dataset/evaluator modules live in `tasks/agile3d/protocol/`; the native PyTorch decoder and explicit NumPy quantization bridge live in `evaluation/`. See `manifests/agile3d_extraction.json` and [upstream](https://github.com/ywyue/AGILE3D).
- **Mask3D**: decoder, losses and segment/task logic derived from the executed Mask3D transfer implementation. Copyright Jonas Schult, MIT; full text in `third_party/Mask3D.LICENSE`. See `manifests/mask3d_extraction.json` and [upstream](https://github.com/JonasSchult/Mask3D).
- **LitePT**: mesh normal preparation and runtime adaptations derive from LitePT. The full upstream notice is in `third_party/LitePT.LICENSE`. The external source and applied patch are locked in `src/delimit3d/runtime/assets/sources.lock.json`. See [upstream](https://github.com/Neilus03/LitePT).

## External sources and weights

| Project | Locked source | Terms |
| --- | --- | --- |
| LitePT | `055c3fa4285a4feef679cffa69c9542e8b09e922` plus recorded patch | Upstream license and notices |
| Concerto | `5b41117160cdd92c3538be1911277b17e7ef5871` | Upstream source/weight terms |
| Sonata | `18c09ff8d713494f78a8213792262b910977a65d` | Apache 2.0 source; CC BY-NC 4.0 public weights |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Upstream source/weight terms |
| UnSAMv2 | `4f2ce5148bc9c769169b7c45ad28c7af28399ff0` | Upstream source/weight terms, including SAM2 components |

Source/weight manifests record exact revisions and content hashes. Fetching a dependency does not change its license. Official Structured3D, ScanNet and AGILE3D benchmark data are obtained separately under their dataset terms.

PyTorch, NumPy, SciPy, OpenCV, Pillow, imageio, plyfile, Open3D, spconv, FlashAttention, torch-scatter, torch-cluster, PointROPE and pointops remain governed by their respective licenses.
