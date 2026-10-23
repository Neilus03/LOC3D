# LOC3D: research direction

Updated 23 October 2026. LOC3D stands for **Learning Object-centric 3D
Representations**. Earlier project material uses Delimit3D or SCEPT3R. Historical
experiment names identify their original recipes and are retained for provenance.

LOC3D studies whether grouping relations from 2D region proposals can teach a
reusable 3D encoder to group objects and separate individual instances. The main
setting trains LitePT-S* from scratch on **2,930 synthetic scenes from
Structured3D**, with a disjoint 326-scene validation partition. Pretraining uses
no 3D instance, part or semantic annotations. Frozen SAM2 supplies region
proposals; geometry, camera parameters and depth associate those proposals with
visible 3D points.

Object-centric here describes dense point features that support grouping and
instance separation. It does not mean an explicit decomposition into object
slots. The main downstream questions concern automatic class-agnostic instance
segmentation and interactive object segmentation. Semantic labeling of recovered
objects is a supporting adaptation study.

## Main formulation

SAM2 proposals remain separate within each rendered view and may overlap. The
training signal does not require matching proposal identities across views or
constructing a globally consistent 3D instance partition. A point is known for a
view when at least one proposal covers it; unseen and uncovered points remain
unknown.

For a selected proposal, positives come from its members and negatives from
known points outside it. **A negative remains eligible when another proposal
contains it and either positive endpoint.** At the voxel-token level, positive
and negative memberships must be pure for the selected proposal; mixed and
unknown tokens are excluded. Uniform, spatially hard and feature-hard sampling
provide the negative families.

LitePT-S* receives coordinates, RGB and normals. A projection head on its finest
features supports the symmetric relative contrastive objective following
PartField. The main recipe supervises this finest resolution directly, without
the earlier hierarchy or VICReg objectives. After pretraining, the projection
head is discarded and the native encoder features are reused. Feature extraction
requires no SAM2 inference on the test scene. Task decoders and semantic adapters
retain their own inputs, training and checkpoint requirements.

The [pretraining guide](pretraining.md#sam2-dec0) records executable recipes,
policy-specific fixed evaluation and recovery requirements. The formulation is
the main LOC3D method; that choice does not establish superiority on every task.

## Preserve recipe identity

| Family | Negative eligibility and scope |
| --- | --- |
| **LOC3D: SAM2 selected-proposal negatives** | Known, pure negatives outside the selected proposal; membership in another shared proposal is allowed. Records use `allow_shared_proposal_negatives=true`. |
| **SAM2 overlap veto** | Adds a veto when a negative shares another relevant proposal with either positive endpoint. Overlapping proposals are still retained. |
| **Historical UnSAMv2 variants** | Earlier proposal sources and multigranular, multiscale or hierarchy recipes, with their own grids, regularizers and training histories. |

The veto is an alternative negative policy to investigate under matched
conditions. Existing checkpoints may differ in point cap, training exposure or
other settings; comparing those checkpoints does not isolate the policy alone.
An epoch-150 selected-proposal checkpoint with a 100k training cap must remain
distinct from an overlap-veto checkpoint trained with another cap or budget.
Historical UnSAMv2 and overlap-veto scores must retain those labels rather than
being reassigned to the main LOC3D formulation.

## What each evaluation can establish

The [23 October research update](research_update_20261023.md) records the latest
verified evidence and protocol boundaries included in this release. Current
release verification is indexed in [the release status](../CURRENT.md).
The [11–17 October evidence record](weekly_update_20261017.md) remains a dated
record of the historical recipes it names, including completed UnSAMv2 transfer
and semantic-labeling studies. Newer evidence does not change those identities.

| Evaluation | Interpretation |
| --- | --- |
| Frozen cosine retrieval and spherical k-means | Representation diagnostics. Report ground-truth query selection or oracle cluster counts when used, as well as grids, native feature interfaces and scoring support. These are separate from decoder-backed instance AP. |
| Articulate3D part retrieval and matched-pair AUROC | Custom frozen-feature selectivity within the documented parent/part cohorts, even when scenes come from the official validation split. These are not official Articulate3D segmentation AP. |
| Mask3D adaptation | Automatic instance segmentation under the exact task recipe. The release reports custom class-agnostic raw-query CA-AP, distinct from class-aware ScanNet leaderboard AP. |
| AGILE3D adaptation | Interactive segmentation at specified click budgets. Single-object and multi-object results have separate coverage and metrics. |
| Fewer labeled scenes | Downstream annotation efficiency on the named subset and training budget. Custom seed-42 scene subsets remain separate from official ScanNet Data-Efficient lists. |
| Sparse point annotations | A separate low-annotation protocol. Distinguish pilots from full-budget evaluations and record how instance as well as semantic supervision is restricted. |
| Semantic adaptation | Recognition under a named label budget, input modality and classifier. Accuracy conditional on geometrically matched masks excludes missed objects and is not whole-scene segmentation performance. |

See [task protocols](tasks.md), [feature probing](probes.md) and
[trained-model inference](inference.md) for the corresponding release interfaces.
The public probe commands accept user-selected queries or cluster counts; a
research diagnostic can additionally use annotations to construct those inputs
and must disclose that choice.

Comparisons of native pretrained systems are valid for their stated interfaces.
Different architectures, feature widths, input signals or pretraining data limit
attribution to the pretraining objective alone. A scratch-versus-pretrained
LitePT-S* comparison supports that attribution only when decoder initialization,
data, input channels, preprocessing, updates, learning rates, seed, normalization
and evaluation match. Frozen, reduced-encoder-LR and equal-LR arms remain separate.

## Research questions and limits

The working hypothesis is that learning grouping relations provides a useful
basis for object-oriented tasks, with semantic recognition added through a
separate adaptation stage. Semantic recognition and instance separation are
different capabilities. The project does not establish that semantic learning
generally harms grouping, that existing 3D representations lack instance
awareness, or that semantic adaptation is universally inexpensive.

The main open questions are how the negative policy affects transfer under
matched pretraining conditions, how performance changes with pretraining data
and exposure, and how much annotation and compute semantic adaptation requires.
Single-source training alone establishes neither data efficiency nor compute
efficiency. Those claims need measured budgets and appropriate comparisons.

The annotation claim concerns 3D task labels during pretraining. SAM2 has its own
training history, synthetic scenes are designed assets, and downstream adaptation
can use annotations. Class-agnostic proposals also do not imply semantics-free
features: the encoder uses RGB and normals as well as coordinates.

Only completed, verified evaluations belong in result tables. Launches, partial
training, feature illustrations and favorable hypotheses do not establish final
downstream results. Research plans and dated snapshots should retain their date
and completion status.
