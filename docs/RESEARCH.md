# Starfall Breakout: approach and evidence

Research date: 12 October 2026. Hardware: Intel i5-10210U (4 cores / 8 threads), 16 GB RAM, Intel UHD integrated graphics. CUDA is unavailable; OpenVINO reports CPU and GPU devices.

## Recommendation

Use local game-specific perception, timestamped object tracking, and a receding-horizon controller as the first real-phone baseline. Compare small learned feature policies after control and perception are measured. This is a project-specific engineering choice, not a claim that one algorithm is universally best.

1. Stream directly from a version-pinned scrcpy server, decode H.264 in memory, and expose one latest-frame slot. Send bounded repeated DOWN/MOVE/UP drag strokes over its persistent control socket; actual held-touch tests plateaued.
2. Start with manually selected sprite templates where they are reliable. Replace unreliable animated/overlapping categories with a custom detector trained on manually reviewed game frames.
3. Estimate motion with constant-velocity Kalman filters and gated assignment. Expand collision radii for uncertainty and expire lost tracks.
4. Evaluate 81 two-stage trajectories, including continuous swept-circle checks between sampled times. Collision penalties dominate collectible rewards. Include an optional attack-lane term because Layla automatically shoots to the right.
5. Compare PPO and DQN on compact object features in a separate simulator. Collect human action demonstrations if simulation transfer is weak. Promote a policy only after held-out performance and real-game evaluation support it.

## Current documentation and model choice

The current official Ultralytics model catalogue recommends YOLO26 and YOLO11 for stable workloads. YOLO26 offers an end-to-end detection path and CPU-oriented improvements; those published benchmarks are not measurements on this laptop. YOLO26n and YOLO11n are the relevant starting candidates, rather than large models. Test 320, 416, and 640 inference sizes on the same real frames; small objects may lose accuracy at 320. The benchmark records full predict-call latency after warmup, not just the network forward pass. See the [official model catalogue](https://docs.ultralytics.com/models/), [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/), and [YOLO26/YOLO11 comparison](https://docs.ultralytics.com/compare/yolo26-vs-yolo11/).

On this Intel machine, OpenVINO CPU and integrated-GPU execution are candidates. Its runtime enumerating a GPU does not establish that it is faster. Exported models must be timed, and accuracy must be checked after conversion or quantization. See [Ultralytics OpenVINO integration](https://docs.ultralytics.com/integrations/openvino/). COCO weights provide transferable image features but contain no verified Starfall-specific class mapping. Do not map arbitrary COCO labels to meteors or claim game accuracy from timing tests.

Ultralytics distributes its software/models under AGPL-3.0 and Enterprise licensing. Review the official terms before redistribution or commercial integration; no paid license or service has been purchased. [Official licensing guidance](https://docs.ultralytics.com/help/contributing/).

## Capture and control

The actual current release is scrcpy 5.0.1, published 8 October 2026. Its portable Windows bundle was downloaded from the official release and checked against the GitHub SHA256 digest. scrcpy's protocol is internal and version-dependent: the implementation therefore pins the client and server to 5.0.1. A newer bundle must not be silently substituted.

The documentation still describes parts of the protocol as 4.0, so the implementation also checked the release-tagged C serialization tests, demuxer, and Java options. Version 5 uses 12-byte session packets for video dimensions and 12-byte media headers containing timestamp flags and payload length. Touch messages are 32 bytes. These details are covered by tests. [Official release](https://github.com/Genymobile/scrcpy/releases/tag/v5.0.1), [developer documentation](https://github.com/Genymobile/scrcpy/blob/v5.0.1/doc/develop.md), [control serialization tests](https://github.com/Genymobile/scrcpy/blob/v5.0.1/app/tests/test_control_msg_serialize.c).

Repeated ADB screenshot/process launches introduce overhead and do not provide the continuous joystick contact needed here. ADB remains responsible for authorization, forwarding, and server startup. Live video and touches use persistent sockets. The controller uses one pointer, releases on cancellation, and has a watchdog for stalls. Hard process termination, USB removal, or device failure cannot guarantee a delivered release; reconnect stops input until perception recovers. [Android ADB documentation](https://developer.android.com/tools/adb).

## Tracking alternatives

Kalman filtering plus gated Hungarian association is a small, transparent baseline. It needs benchmarking on crossings and missed detections. Template matching has no learned invariance to animation, scale, glow, or overlapping sprites; its current real-screen screening is optimistic because the templates and frames share a session. Optical flow is useful for short gaps and joystick calibration, but camera/background scrolling and spell effects can produce false motion. [OpenCV template-matching documentation](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html).

ByteTrack's second association stage can recover low-confidence boxes, which may help partial occlusion. Its published strong results concern tracking benchmarks rather than these fast game sprites, and its quoted throughput is on a V100. It is a later candidate after detector confidence is meaningful; no ByteTrack performance is claimed here. [Original ByteTrack paper](https://arxiv.org/abs/2110.06864).

## Learning alternatives

| Method | Why consider it | Why it is not the initial real-phone controller |
|---|---|---|
| Predictive control | Uses measurable movement and hazard trajectories; no policy-training attempts needed | Incorrect detections/dynamics can still defeat prediction; approximate circles need calibration |
| PPO with a small MLP | Stable, well supported, supports the nine discrete movement actions | On-policy data cannot be freely reused; real-phone training is slow |
| DQN | Discrete action space and replay allow experience reuse | Sensitive to observation/reward quality; training budget and tuning still matter |
| Human imitation / DAgger | Uses the user's successful behavior and corrective examples | Needs synchronized state-action demonstrations and recovery-state coverage |
| SAC / TD-MPC2 | Competitive candidates if joystick magnitude becomes continuous or measured dynamics need learning | Added complexity is not justified before the nine-action baseline is tested |
| DreamerV3 / Dreamer4 | Learn world models and train by imagination; useful when interaction is expensive | High compute/data demands, uncertain modeling of game effects; published results do not establish laptop or Starfall suitability |

The original [PPO paper](https://arxiv.org/abs/1707.06347) and [SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) motivate reproducible baselines and separate evaluation. [DAgger](https://arxiv.org/abs/1011.0686) addresses the state-distribution shift that naive behavioral cloning encounters. [TD-MPC2](https://arxiv.org/abs/2310.16828) optimizes trajectories inside a learned latent model for continuous control. [DreamerV3](https://danijar.com/project/dreamerv3/) and the newer [Dreamer4](https://danijar.com/project/dreamer4/) are promising research alternatives, not pretrained policies for this game. Choosing them merely because they are newer would overlook this machine's hardware and the available game data.

Randomize uncertain speed, response, hazard trajectories, and appearance only over ranges supported by measurement as evidence grows. The existing sandbox uses explicitly synthetic defaults. [Domain randomization research](https://arxiv.org/abs/1703.06907) supports investigating variability during transfer, but does not guarantee transfer to MLBB.

## What the real phone established

The in-game rules and inspection screenshots establish a joystick-controlled player (Layla), automatic attacks to the right, points for destroying meteors, a first Magma Wall clear condition, an endless phase, visible hearts, and several power-up orb types. The Starshot Split description states that it adds shots above and below the main attack. Three visually different meteor sprites were captured and provisionally named orange, blue, and purple meteor for annotation; these are appearance labels, not confirmed distinct physics classes.

An inspection attempt reached a first-clear/continue overlay displaying 3,660 points and 01:28. This was observation/setup footage and may include human input; it is NOT an autonomous agent benchmark. The long-term 100,000-point objective is unverified.

The dataset now contains more than 300 real frames from several observation and gameplay sessions. A 31-frame player-only subset has visually reviewed approximate boxes, and a small detector candidate was fine-tuned and evaluated on separate attempts. Broader independent labels are still needed for reliable multi-class perception and false-positive control. No synthetic images are represented as real training examples. Score, active effects, hitboxes and upgrade synergies are not yet reliably extracted by the live agent; unknown values have explicit masks or nulls. Heart scanning was removed from the active workflow at the user's request. The human handles low-health exits and reopens Starfall; a visual page loop starts subsequent attempts. Human resets are truncations rather than inferred deaths.

Real movement learning uses a compact action-conditioned linear response model. It fits displacement from isolated, timestamped drag strokes, rejects uncertain/overlapping/edge-clipped observations, and evaluates on the latest held-out attempt. Changes require sample coverage and a bootstrap error-improvement gate consistent with the existing calibration. This is supervised system identification used by MPC, not score-based RL, and reduced prediction error does not establish improved game score.

The user's subsequent combat clarification and actual recorded frames establish an orange bottom-right ultimate with a charge counter and a map-height Magma Wall with a visible glowing core. The ultimate fires on one tap. The live planner now prioritizes core alignment, scores shooting lanes and time-sensitive orb collection within collision-free paths, and treats the wall as a rectangle. Color/shape proposals plus reviewed icon references identify additional orb appearances. Counter readings gate ultimate taps; the orange glow alone is insufficient because it remains visible at zero charges. This remains a prototype pending a new connected-phone combat trial. The current synthetic sandbox has not yet reproduced these complete mechanics, so additional offline training alone does not validate their transfer.

## Evaluation protocol

Measure every stage and report median/p95/p99. Receive-to-dispatch time excludes phone encoding/USB delay. Actual observable motion delay comes from controlled visual calibration. Run synthetic baseline comparisons on paired held-out seeds and keep synthetic scores separate from MLBB points. Report score, survival, collision rate, and collections together. Compare candidate policies against MPC; a bootstrap score improvement with no worse collision rate is a simulation gate only. Real deployment requires additional paired gameplay trials with the same detector, controls, loadout, and rules. Never replace a live policy merely because training loss decreased.

Real-frame detector evaluation requires manual boxes on independent sessions: precision, recall, per-class AP, false-positive/false-negative review, confusion/overlap slices, and inference latency. The dataset validator rejects unreviewed images and invalid boxes. Entire sessions remain in one split, and cross-session near-duplicates merge before splitting. A minimum of three independent groups is enforced.
