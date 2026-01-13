# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of academic papers and resources on **Physical AI** — focusing on Vision-Language-Action (VLA) models, embodied intelligence, and robotic foundation models.

> **Physical AI** refers to AI systems that interact with and manipulate the physical world through robotic embodiments, combining perception, reasoning, and action in real-world environments.

---

## Table of Contents

- [Foundations](#foundations)
  - [Vision-Language Backbones](#vision-language-backbones)
  - [Visual Representations](#visual-representations)
- [VLA Architectures](#vla-architectures)
  - [End-to-End VLAs](#end-to-end-vlas)
  - [Modular VLAs](#modular-vlas)
  - [Compact & Efficient VLAs](#compact--efficient-vlas)
- [Action Representation](#action-representation)
  - [Discrete Tokenization](#discrete-tokenization)
  - [Continuous & Diffusion Policies](#continuous--diffusion-policies)
- [World Models](#world-models)
  - [JEPA & Latent Prediction](#jepa--latent-prediction)
  - [Generative World Models](#generative-world-models)
  - [Embodied World Models](#embodied-world-models)
- [Reasoning & Planning](#reasoning--planning)
  - [Chain-of-Thought & Deliberation](#chain-of-thought--deliberation)
  - [Error Detection & Recovery](#error-detection--recovery)
- [Learning Paradigms](#learning-paradigms)
  - [Imitation Learning](#imitation-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Reward Design](#reward-design)
- [Scaling & Generalization](#scaling--generalization)
  - [Scaling Laws](#scaling-laws)
  - [Cross-Embodiment Transfer](#cross-embodiment-transfer)
  - [Open-Vocabulary Generalization](#open-vocabulary-generalization)
- [Deployment](#deployment)
  - [Quantization & Compression](#quantization--compression)
  - [Real-Time Control](#real-time-control)
- [Safety & Alignment](#safety--alignment)
- [Lifelong Learning](#lifelong-learning)
- [Applications](#applications)
  - [Humanoid Robots](#humanoid-robots)
  - [Manipulation](#manipulation)
  - [Navigation](#navigation)
- [Resources](#resources)
  - [Datasets & Benchmarks](#datasets--benchmarks)
  - [Simulation Platforms](#simulation-platforms)
- [Surveys](#surveys)

---

## Foundations

### Vision-Language Backbones

> Core vision-language models that serve as pretrained backbones for Physical AI systems.

- **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision", *ICML 2021*. [[Paper](https://arxiv.org/abs/2103.00020)] [[Code](https://github.com/openai/CLIP)]
  - Foundational model aligning vision and language that underlies most VLA perception systems.

- **SigLIP**: "Sigmoid Loss for Language Image Pre-Training", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.15343)]

- **PaLI-X**: "PaLI-X: On Scaling up a Multilingual Vision and Language Model", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2305.18565)]
  - Vision-language backbone for RT-2.

- **LLaVA**: "Visual Instruction Tuning", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2304.08485)] [[Project](https://llava-vl.github.io/)]

- **Prismatic VLMs**: "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.07865)] [[Code](https://github.com/TRI-ML/prismatic-vlms)]
  - Systematic study of VLM design choices informing OpenVLA and other robotics VLMs.

### Visual Representations

> Self-supervised visual encoders and perception models used in robotics.

- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision", *arXiv, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.07193)] [[Code](https://github.com/facebookresearch/dinov2)]

- **SAM**: "Segment Anything", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2304.02643)] [[Project](https://segment-anything.com/)]

- **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2203.12601)] [[Code](https://github.com/facebookresearch/r3m)]
  - Pretrained on Ego4D human video, transfers effectively to robot manipulation.

- **MVP**: "Masked Visual Pre-training for Motor Control", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.06173)] [[Project](https://tetexiao.com/projects/mvp)]

- **Grounding DINO**: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", *ECCV 2024*. [[Paper](https://arxiv.org/abs/2303.05499)] [[Code](https://github.com/IDEA-Research/GroundingDINO)]

---

## VLA Architectures

### End-to-End VLAs

> Monolithic models that treat vision, language, and actions as unified tokens in a single architecture.

- **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *RSS 2023*. [[Paper](https://arxiv.org/abs/2212.06817)] [[Project](https://robotics-transformer1.github.io/)] [[Code](https://github.com/google-research/robotics_transformer)]
  - Pioneer proving large-scale multi-task data could train a single transformer for diverse manipulations.

- **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Project](https://robotics-transformer2.github.io/)]
  - Established the VLA paradigm by co-fine-tuning VLMs on robotic data.

- **OpenVLA**: "OpenVLA: An Open-Source Vision-Language-Action Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.09246)] [[Project](https://openvla.github.io/)] [[Code](https://github.com/openvla/openvla)]
  - Open-source 7B model that outperformed the 55B RT-2-X, democratizing VLA research.

- **PaLM-E**: "PaLM-E: An Embodied Multimodal Language Model", *ICML 2023*. [[Paper](https://arxiv.org/abs/2303.03378)] [[Project](https://palm-e.github.io/)]
  - 562B parameter model demonstrating emergent multi-modal chain-of-thought reasoning.

- **VIMA**: "VIMA: General Robot Manipulation with Multimodal Prompts", *ICML 2023*. [[Paper](https://arxiv.org/abs/2210.03094)] [[Project](https://vima.cs.stanford.edu/)] [[Code](https://github.com/vimalabs/VIMA)]
  - Introduced multimodal prompting (text + images) for specifying manipulation tasks.

- **LEO**: "An Embodied Generalist Agent in 3D World", *ICML 2024*. [[Paper](https://arxiv.org/abs/2311.12871)] [[Project](https://embodied-generalist.github.io/)]

- **3D-VLA**: "3D-VLA: A 3D Vision-Language-Action Generative World Model", *ICML 2024*. [[Paper](https://arxiv.org/abs/2403.09631)] [[Project](https://vis-www.cs.umass.edu/3dvla)]

- **Gato**: "A Generalist Agent", *TMLR 2022*. [[Paper](https://arxiv.org/abs/2205.06175)] [[Blog](https://deepmind.google/discover/blog/a-generalist-agent/)]
  - Single transformer handling 604 distinct tasks across games, chat, and robotics.

- **RoboFlamingo**: "Vision-Language Foundation Models as Effective Robot Imitators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2311.01378)] [[Project](https://roboflamingo.github.io/)]

- **Magma**: "Magma: A Foundation Model for Multimodal AI Agents", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.13130)] [[Code](https://github.com/microsoft/Magma)]

- **RoboVLMs**: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.14058)] [[Project](https://robovlms.github.io/)]

### Modular VLAs

> Models that decouple cognition (VLM-based planning) from action (specialized motor modules).

- **CogACT**: "CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.19650)] [[Project](https://cogact.github.io/)]
  - Decouples high-level cognition from low-level action via Diffusion Action Transformer.

- **Gemini Robotics**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)] [[Blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)]
  - Introduces "Thinking Before Acting" with internal natural language reasoning.

- **Helix**: "Helix: A Vision-Language-Action Model for Generalist Humanoid Control", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]

- **SayCan**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2204.01691)] [[Project](https://say-can.github.io/)]
  - First to combine LLM semantic knowledge with learned affordance functions.

- **Code as Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sep 2022*. [[Paper](https://arxiv.org/abs/2209.07753)] [[Project](https://code-as-policies.github.io/)]
  - Seminal work showing LLMs can generate executable robot control code.

- **SayPlan**: "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Task Planning", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.06135)] [[Project](https://sayplan.github.io/)]

- **Inner Monologue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Project](https://innermonologue.github.io/)]
  - Pioneered closed-loop language feedback where robots verbalize observations.

- **Instruct2Act**: "Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.11176)] [[Code](https://github.com/OpenGVLab/Instruct2Act)]

- **TidyBot**: "TidyBot: Personalized Robot Assistance with Large Language Models", *IROS 2023*. [[Paper](https://arxiv.org/abs/2305.05658)] [[Project](https://tidybot.cs.princeton.edu/)]

### Compact & Efficient VLAs

> Lightweight VLA models optimized for fast inference and edge deployment.

- **TinyVLA**: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2409.12514)] [[Project](https://tiny-vla.github.io/)]

- **SmolVLA**: "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", *arXiv, Jun 2025*. [[Paper](https://huggingface.co/blog/smolvla)] [[Code](https://github.com/huggingface/lerobot)]
  - 450M parameters achieving comparable performance to 10x larger models.

- **OpenVLA-OFT**: "OpenVLA-OFT: Efficient Fine-Tuning for Open Vision-Language-Action Models", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]

- **RT-H**: "RT-H: Action Hierarchies Using Language", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.01823)] [[Project](https://rt-hierarchy.github.io/)]

- **LAPA**: "Latent Action Pretraining from Videos", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.11758)] [[Project](https://latentactionpretraining.github.io/)]
  - Pretrains action representations from unlabeled video.

---

## Action Representation

### Discrete Tokenization

> Models that convert continuous joint movements into discrete "action tokens".

- **FAST**: "FAST: Efficient Action Tokenization for Vision-Language-Action Models", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.09747)] [[Project](https://www.pi.website/research/fast)]
  - Uses frequency-space (DCT) tokenization to compress action sequences 7x.

- **GR-1**: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2312.13139)] [[Project](https://gr1-manipulation.github.io/)]

- **GR-2**: "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.06158)] [[Project](https://gr2-manipulation.github.io/)]

- **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", *RSS 2023*. [[Paper](https://arxiv.org/abs/2304.13705)] [[Project](https://tonyzhaozh.github.io/aloha/)] [[Code](https://github.com/tonyzhaozh/act)]
  - Introduced Action Chunking with Transformers for smooth bimanual manipulation.

- **Behavior Transformers**: "Behavior Transformers: Cloning k Modes with One Stone", *NeurIPS 2022*. [[Paper](https://arxiv.org/abs/2206.11251)] [[Code](https://github.com/notmahi/bet)]

### Continuous & Diffusion Policies

> Models that use diffusion or flow matching to generate continuous trajectories.

- **π₀ (pi-zero)**: "π₀: A Vision-Language-Action Flow Model for General Robot Control", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.24164)] [[Project](https://www.physicalintelligence.company/blog/pi0)]
  - Uses flow matching to generate high-frequency (50 Hz) continuous actions for dexterous tasks.

- **π₀.5**: "π₀.5: Scaling Robot Foundation Models", *arXiv, Apr 2025*. [[Paper](https://www.physicalintelligence.company/blog/pi0-5)]

- **Octo**: "Octo: An Open-Source Generalist Robot Policy", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.12213)] [[Project](https://octo-models.github.io/)] [[Code](https://github.com/octo-models/octo)]

- **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", *RSS 2023*. [[Paper](https://arxiv.org/abs/2303.04137)] [[Project](https://diffusion-policy.cs.columbia.edu/)] [[Code](https://github.com/real-stanford/diffusion_policy)]
  - Foundational work showing diffusion models excel at capturing multimodal action distributions.

- **RDT-1B**: "RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.07864)] [[Project](https://rdt-robotics.github.io/rdt-robotics/)]

- **DexVLA**: "DexVLA: Vision-Language Model with Plug-In Diffusion Expert", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.05855)] [[Project](https://dex-vla.github.io/)]

- **Diffusion-VLA**: "Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.03293)] [[Project](https://diffusion-vla.github.io/)]

- **3D Diffusion Policy**: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via 3D Representations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2403.03954)] [[Project](https://3d-diffusion-policy.github.io/)]

- **Moto**: "Moto: Latent Motion Token as the Bridging Language for Robot Manipulation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04445)] [[Project](https://chenyi99.github.io/moto/)]

- **Consistency Policy**: "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.07503)] [[Project](https://consistency-policy.github.io/)]
  - Distills diffusion policies into single-step models for 10x faster inference.

---

## World Models

### JEPA & Latent Prediction

> Joint-Embedding Predictive Architecture (JEPA) predicts future latent states rather than pixels.

- "A Path Towards Autonomous Machine Intelligence", *Meta AI, Jun 2022*. [[Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)]
  - LeCun's foundational vision describing the "world model in the middle" cognitive architecture.

- **I-JEPA**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", *CVPR 2023*. [[Paper](https://arxiv.org/abs/2301.08243)] [[Code](https://github.com/facebookresearch/ijepa)]
  - First practical JEPA implementation.

- **V-JEPA**: "Video Joint Embedding Predictive Architecture", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.03014)] [[Code](https://github.com/facebookresearch/jepa)]

- **MC-JEPA**: "MC-JEPA: Self-Supervised Learning of Motion and Content Features", *CVPR 2023*. [[Paper](https://arxiv.org/abs/2307.12698)]

- **LeJEPA**: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", *arXiv, Nov 2025*. [[Paper](https://arxiv.org/abs/2511.XXXXX)]

- **VL-JEPA**: "VL-JEPA: Vision-Language Joint Embedding Predictive Architecture", *arXiv, Dec 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]

- "Value-guided Action Planning with JEPA World Models", *arXiv, Jan 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]

### Generative World Models

> World models that generate pixels, video, or interactive environments.

- **World Models**: "World Models", *NeurIPS 2018*. [[Paper](https://arxiv.org/abs/1803.10122)] [[Project](https://worldmodels.github.io/)]
  - Seminal Ha & Schmidhuber work popularizing world models for RL.

- **DreamerV3**: "Mastering Diverse Domains through World Models", *arXiv, Jan 2023*. [[Paper](https://arxiv.org/abs/2301.04104)] [[Project](https://danijar.com/project/dreamerv3/)]
  - State-of-the-art world model RL agent mastering 150+ tasks.

- **Genie**: "Genie: Generative Interactive Environments", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.15391)] [[Project](https://sites.google.com/view/genie-2024)]
  - Learns interactive world models from unlabeled videos.

- **Genie 2**: "Genie 2: A Large-Scale Foundation World Model", *DeepMind, Dec 2024*. [[Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)]
  - Generates diverse, playable 3D worlds from single images.

- **Sora**: "Video Generation Models as World Simulators", *OpenAI, Feb 2024*. [[Blog](https://openai.com/research/video-generation-models-as-world-simulators)]

- **GAIA-1**: "GAIA-1: A Generative World Model for Autonomous Driving", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.17080)]

- **GameNGen**: "Diffusion Models Are Real-Time Game Engines", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.14837)]
  - Runs DOOM entirely on a neural network.

- **DIAMOND**: "Diffusion for World Modeling: Visual Details Matter in Atari", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2405.12399)] [[Code](https://github.com/eloialonso/diamond)]

- **3D Gaussian Splatting**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering", *SIGGRAPH 2023*. [[Paper](https://arxiv.org/abs/2308.04079)] [[Project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]

- "From Words to Worlds: Spatial Intelligence is AI's Next Frontier", *World Labs, 2025*. [[Blog](https://www.worldlabs.ai/)]
  - Fei-Fei Li's manifesto on generative, multimodal, actionable world models.

- **Marble**: "Marble: A Multimodal World Model", *World Labs, Nov 2025*. [[Project](https://www.worldlabs.ai/)]

- **RTFM**: "RTFM: A Real-Time Frame Model", *World Labs, Oct 2025*. [[Project](https://www.worldlabs.ai/)]

### Embodied World Models

> World models designed for robotic manipulation, navigation, and physical reasoning.

- **Structured World Models**: "Structured World Models from Human Videos", *RSS 2023*. [[Paper](https://arxiv.org/abs/2308.10901)] [[Project](https://human-world-model.github.io/)]

- **WHALE**: "WHALE: Towards Generalizable and Scalable World Models for Embodied Decision-making", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.05619)]

- "A Controllable Generative World Model for Robot Manipulation", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.XXXXX)]

- **Code World Model**: "Code World Model: Learning to Execute Code in World Simulation", *Meta AI, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.XXXXX)]

- **PhyGDPO**: "PhyGDPO: Physics-Aware Text-to-Video Generation via Direct Preference Optimization", *Meta AI, Jan 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]

- "The Essential Role of Causality in Foundation World Models for Embodied AI", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.06665)]

- **MineDreamer**: "MineDreamer: Learning to Follow Instructions via Chain-of-Imagination", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.12037)] [[Project](https://sites.google.com/view/minedreamer)]

- **Video Language Planning**: "Video Language Planning", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.10625)] [[Project](https://video-language-planning.github.io/)]
  - Combines video prediction with tree search for long-horizon planning.

- "Learning Universal Policies via Text-Guided Video Generation", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.00111)] [[Project](https://universal-policy.github.io/unipi/)]

- **SIMA**: "Scaling Instructable Agents Across Many Simulated Worlds", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2404.10179)] [[Blog](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)]

- **UniSim**: "UniSim: Learning Interactive Real-World Simulators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.06114)] [[Project](https://universal-simulator.github.io/unisim/)]

---

## Reasoning & Planning

### Chain-of-Thought & Deliberation

> Models implementing "thinking before acting" with explicit reasoning or value-guided search.

- **Hume**: "Hume: Introducing Deliberative Alignment in Embodied AI", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Uses a value-query head to evaluate candidate actions via repeat sampling.

- **Embodied-CoT**: "Robotic Control via Embodied Chain-of-Thought Reasoning", *arXiv, Jul 2024*. [[Paper](https://arxiv.org/abs/2407.08693)] [[Project](https://embodied-cot.github.io/)]

- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR 2023*. [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]
  - Interleaves reasoning traces with actions.

- **ReKep**: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.01652)] [[Project](https://rekep-robot.github.io/)]

- **TraceVLA**: "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.10345)] [[Project](https://tracevla.github.io/)]

- **LLM-State**: "LLM-State: Open World State Representation for Long-horizon Task Planning", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.17406)]

- **Statler**: "Statler: State-Maintaining Language Models for Embodied Reasoning", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.17840)] [[Project](https://statler-lm.github.io/)]

- **RoboReflect**: "RoboReflect: Reflective Reasoning for Robot Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

### Error Detection & Recovery

> Methods for detecting failures and correcting robot actions in real-time.

- **DoReMi**: "Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", *arXiv, Jul 2023*. [[Paper](https://arxiv.org/abs/2307.00329)] [[Project](https://sites.google.com/view/doremi-paper)]

- **CoPAL**: "Corrective Planning of Robot Actions with Large Language Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.07263)] [[Project](https://sites.google.com/view/copal-robot)]

- **Code-as-Monitor**: "Code-as-Monitor: Constraint-aware Visual Programming for Failure Detection", *CVPR 2025*. [[Paper](https://arxiv.org/abs/2412.04455)] [[Project](https://code-as-monitor.github.io/)]

- **AHA**: "AHA: A Vision-Language-Model for Detecting and Reasoning over Failures", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.XXXXX)]

- **PRED**: "Pre-emptive Action Revision by Environmental Feedback", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]

---

## Learning Paradigms

### Imitation Learning

> Behavioral cloning and learning from demonstrations.

- **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Project](https://cliport.github.io/)] [[Code](https://github.com/cliport/cliport)]
  - Combines CLIP semantic features with transporter networks.

- **Play-LMP**: "Learning Latent Plans from Play", *CoRL 2019*. [[Paper](https://arxiv.org/abs/1903.01973)] [[Project](https://learning-from-play.github.io/)]
  - Learns reusable skills from unstructured "play" data without task labels.

- **MimicPlay**: "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2302.12422)] [[Project](https://mimic-play.github.io/)]

- **RVT**: "RVT: Robotic View Transformer for 3D Object Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.14896)] [[Project](https://robotic-view-transformer.github.io/)] [[Code](https://github.com/NVlabs/RVT)]

- **RVT-2**: "RVT-2: Learning Precise Manipulation from Few Demonstrations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2406.08545)] [[Project](https://robotic-view-transformer-2.github.io/)]

- **DIAL**: "Robotic Skill Acquisition via Instruction Augmentation", *arXiv, Nov 2022*. [[Paper](https://arxiv.org/abs/2211.11736)] [[Project](https://instructionaugmentation.github.io/)]

- **Perceiver-Actor**: "A Multi-Task Transformer for Robotic Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2209.05451)] [[Project](https://peract.github.io/)] [[Code](https://github.com/peract/peract)]

- **BOSS**: "Bootstrap Your Own Skills: Learning to Solve New Tasks with LLM Guidance", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2310.10021)] [[Project](https://clvrai.github.io/boss/)]

### Reinforcement Learning

> RL-based methods for optimizing VLA policies.

- **CO-RFT**: "CO-RFT: Chunked Offline Reinforcement Learning Fine-Tuning for VLAs", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Two-stage offline RL achieving 57% improvement over imitation learning.

- **HICRA**: "HICRA: Hierarchy-Aware Credit Assignment for Reinforcement Learning in VLAs", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Focuses optimization on "planning tokens" rather than execution tokens.

- **FLaRe**: "FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale RL Fine-Tuning", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.16578)] [[Project](https://robot-flare.github.io/)]

- **Plan-Seq-Learn**: "Plan-Seq-Learn: Language Model Guided RL for Long Horizon Tasks", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2405.01534)] [[Project](https://mihdalal.github.io/planseqlearn/)]

- **GLAM**: "Grounding Large Language Models in Interactive Environments with Online RL", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.02662)] [[Code](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)]

- **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models", *ICML 2023*. [[Paper](https://arxiv.org/abs/2302.06692)]

- **RL4VLA**: "RL4VLA: What Can RL Bring to VLA Generalization?", *NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]

- **TPO**: "TPO: Trajectory-wise Preference Optimization for VLAs", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **ReinboT**: "ReinboT: Reinforcement Learning for Robotic Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

### Reward Design

> Automated reward function generation using language models.

- **Text2Reward**: "Text2Reward: Automated Dense Reward Function Generation", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.11489)] [[Project](https://text-to-reward.github.io/)]
  - LLMs generate dense reward code from task descriptions.

- **Language to Rewards**: "Language to Rewards for Robotic Skill Synthesis", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.08647)] [[Project](https://language-to-reward.github.io/)]

- **ExploRLLM**: "ExploRLLM: Guiding Exploration in Reinforcement Learning with LLMs", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.09583)]

---

## Scaling & Generalization

### Scaling Laws

> Mathematical relationships between model/data scale and robotic performance.

- "Neural Scaling Laws for Embodied AI", *arXiv, May 2024*. [[Paper](https://arxiv.org/abs/2405.14005)]

- "Data Scaling Laws in Imitation Learning for Robotic Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.18647)] [[Project](https://data-scaling-laws.github.io/)]

- **AutoRT**: "AutoRT: Embodied Foundation Models for Large Scale Orchestration", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2401.12963)] [[Project](https://auto-rt.github.io/)]
  - Demonstrates autonomous fleet data collection enabling continuous scaling.

- **SARA-RT**: "SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.01990)]

- "Scaling Robot Learning with Semantically Imagined Experience", *RSS 2023*. [[Paper](https://arxiv.org/abs/2302.11550)]

### Cross-Embodiment Transfer

> Single policies controlling diverse robot types.

- **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]
  - Largest cross-embodiment study with 22 robot types.

- **GENBOT-1K**: "Towards Embodiment Scaling Laws: Training on ~1000 Robot Bodies", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Training on ~1,000 robot bodies enables zero-shot transfer to unseen robots.

- **Crossformer**: "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2408.11812)] [[Project](https://crossformer-model.github.io/)]
  - Single policy controlling manipulators, legged robots, and drones.

- **HPT**: "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20537)] [[Project](https://liruiw.github.io/hpt/)]

- **MetaMorph**: "MetaMorph: Learning Universal Controllers with Transformers", *ICLR 2022*. [[Paper](https://arxiv.org/abs/2203.11931)] [[Project](https://metamorph-iclr.github.io/)]

- **RUMs**: "Robot Utility Models: General Policies for Zero-Shot Deployment", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.05865)] [[Project](https://robotutilitymodels.com/)]

- **URMA**: "Unified Robot Morphology Architecture", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **RoboAgent**: "RoboAgent: Generalization and Efficiency via Semantic Augmentations", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2309.01918)] [[Project](https://robopen.github.io/)]

### Open-Vocabulary Generalization

> Models that generalize to novel visual appearances and semantic concepts.

- **MOO**: "Open-World Object Manipulation using Pre-trained Vision-Language Models", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2303.00905)] [[Project](https://robot-moo.github.io/)]

- **VoxPoser**: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.05973)] [[Project](https://voxposer.github.io/)]
  - Generates 3D affordance and constraint maps from language for zero-shot manipulation.

- **RoboPoint**: "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.10721)] [[Project](https://robo-point.github.io/)]

- **CLIP-Fields**: "CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory", *RSS 2023*. [[Paper](https://arxiv.org/abs/2210.05663)] [[Project](https://mahis.life/clip-fields/)]

- **VLMaps**: "Visual Language Maps for Robot Navigation", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2210.05714)] [[Project](https://vlmaps.github.io/)]

- **NLMap**: "Open-vocabulary Queryable Scene Representations", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2209.09874)] [[Project](https://nlmap-saycan.github.io/)]

- **LERF**: "LERF: Language Embedded Radiance Fields", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.09553)] [[Project](https://www.lerf.io/)]

- **Any-point Trajectory**: "Any-point Trajectory Modeling for Policy Learning", *RSS 2024*. [[Paper](https://arxiv.org/abs/2401.00025)] [[Project](https://xingyu-lin.github.io/atm/)]

---

## Deployment

### Quantization & Compression

> Low-bit weight quantization for efficient edge deployment.

- **BitVLA**: "BitVLA: 1-bit Vision-Language-Action Models for Robotics", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - First 1-bit (ternary) VLA reducing memory to 29.8%.

- **DeeR-VLA**: "DeeR-VLA: Dynamic Inference of Multimodal LLMs for Efficient Robot Execution", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.02359)] [[Code](https://github.com/yueyang130/DeeR-VLA)]

- **QuaRT-VLA**: "Quantized Robotics Transformers for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **PDVLA**: "PDVLA: Parallel Decoding for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

### Real-Time Control

> Methods bridging high-latency AI inference and low-latency physical control.

- **A2C2**: "A2C2: Asynchronous Action Chunk Correction for Real-Time Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]
  - Lightweight head adjusting outdated action chunks based on latest observations.

- **RTC**: "Real-Time Chunking: Asynchronous Execution for Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

---

## Safety & Alignment

> Ethical constraints, safety frameworks, and human-robot alignment.

- **Robot Constitution**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)]
  - Introduces data-driven "Robot Constitution" with natural language rules for safe behavior.

- **ASIMOV**: "ASIMOV: A Safety Benchmark for Embodied AI", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]

- **RoboPAIR**: "Jailbreaking LLM-Controlled Robots", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.13691)] [[Project](https://robopair.org/)]
  - Demonstrates adversarial attacks on LLM-controlled robots.

- **RoboGuard**: "Safety Guardrails for LLM-Enabled Robots", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]

- "Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.10340)]

- "Robots Enact Malignant Stereotypes", *FAccT 2022*. [[Paper](https://arxiv.org/abs/2207.11569)] [[Project](https://sites.google.com/view/robots-enact-stereotypes)]
  - First study showing robots inherit harmful biases from vision-language pretraining.

- "LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.08824)]

- "Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]

---

## Lifelong Learning

> Agents that continuously learn and adapt without forgetting prior skills.

- **Voyager**: "VOYAGER: An Open-Ended Embodied Agent with Large Language Models", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.16291)] [[Project](https://voyager.minedojo.org/)] [[Code](https://github.com/MineDojo/Voyager)]
  - First LLM-powered agent in Minecraft autonomously building a skill library.

- **RoboGen**: "RoboGen: A Generative and Self-Guided Robotic Agent", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.01455)] [[Project](https://robogen-ai.github.io/)]

- **RoboCat**: "RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.11706)] [[Blog](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)]
  - Self-improves through cycles of self-generated data collection.

- **LOTUS**: "LOTUS: Continual Imitation Learning via Unsupervised Skill Discovery", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2311.02058)] [[Project](https://ut-austin-rpl.github.io/Lotus/)]

- **DEPS**: "Describe, Explain, Plan and Select: Interactive Planning with LLMs for Open-World Agents", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.01560)] [[Code](https://github.com/CraftJarvis/MC-Planner)]

- **JARVIS-1**: "JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal LLMs", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.05997)] [[Project](https://craftjarvis-jarvis1.github.io/)]

- **MP5**: "MP5: A Multi-modal Open-ended Embodied System via Active Perception", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2312.07472)] [[Project](https://craftjarvis.github.io/MP5/)]

- **SPRINT**: "SPRINT: Semantic Policy Pre-training via Language Instruction Relabeling", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.11886)] [[Project](https://clvrai.github.io/sprint/)]

---

## Applications

### Humanoid Robots

> Foundation models for humanoid robot control.

- **GR00T N1**: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.14734)] [[Project](https://developer.nvidia.com/isaac/gr00t)]
  - NVIDIA's dual-system architecture for humanoid control.

- **HumanPlus**: "HumanPlus: Humanoid Shadowing and Imitation from Humans", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.10454)] [[Project](https://humanoid-ai.github.io/)]

- **ExBody**: "Expressive Whole-Body Control for Humanoid Robots", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.16796)] [[Project](https://expressive-humanoid.github.io/)]

- **H2O**: "Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.04436)] [[Project](https://human2humanoid.com/)]

- **OmniH2O**: "OmniH2O: Universal Human-to-Humanoid Teleoperation and Learning", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.08858)] [[Project](https://omni.human2humanoid.com/)]

- "Learning Humanoid Locomotion with Transformers", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2303.03381)] [[Project](https://humanoid-locomotion.github.io/)]

### Manipulation

> Robot manipulation with foundation models.

- **Scaling Up Distilling Down**: "Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.14535)] [[Project](https://www.cs.columbia.edu/~huy/scalingup/)]

- **LLM3**: "LLM3: Large Language Model-based Task and Motion Planning with Failure Reasoning", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11552)]

- **ManipVQA**: "ManipVQA: Injecting Robotic Affordance into Multi-Modal LLMs", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11289)]

- **UniAff**: "UniAff: A Unified Representation of Affordances for Tool Usage and Articulation", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.20551)]

- **SKT**: "SKT: State-Aware Keypoint Trajectories for Robotic Garment Manipulation", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.18082)]

- **Manipulate-Anything**: "Manipulate-Anything: Automating Real-World Robots using VLMs", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.18915)] [[Project](https://robot-ma.github.io/)]

- **A3VLM**: "A3VLM: Actionable Articulation-Aware Vision Language Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.07549)]

- **LaN-Grasp**: "Language-Driven Grasp Detection", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2311.09876)]

- **Grasp Anything**: "Pave the Way to Grasp Anything: Transferring Foundation Models", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.05716)]

### Navigation

> Vision-language models for robot navigation.

- **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Project](https://sites.google.com/view/lmnav)]

- **NaVILA**: "NaVILA: Legged Robot Vision-Language-Action Model for Navigation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04453)] [[Project](https://navila-bot.github.io/)]

- **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2203.10421)]

- **L3MVN**: "L3MVN: Leveraging Large Language Models for Visual Target Navigation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2304.05501)]

- **NaVid**: "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.15852)] [[Project](https://pku-epic.github.io/NaVid/)]

- **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2309.15940)] [[Project](https://ovsg-l.github.io/)]

- **CANVAS**: "CANVAS: Commonsense-Aware Navigation System", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.01273)]

- **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV 2020*. [[Paper](https://arxiv.org/abs/2004.14973)]

- **ThinkBot**: "ThinkBot: Embodied Instruction Following with Thought Chain Reasoning", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.07062)]

---

## Resources

### Datasets & Benchmarks

- **Open X-Embodiment**: Largest open-source robot dataset with 1M+ trajectories from 22 embodiments. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]

- **DROID**: Large-scale in-the-wild manipulation dataset (76K trajectories, 564 scenes). [[Paper](https://arxiv.org/abs/2403.12945)] [[Project](https://droid-dataset.github.io/)]

- **BridgeData V2**: Multi-task dataset for few-shot transfer. [[Paper](https://arxiv.org/abs/2308.12952)] [[Project](https://rail-berkeley.github.io/bridgedata/)]

- **ARIO**: Standardized format unifying diverse robot datasets. [[Paper](https://arxiv.org/abs/2408.10899)] [[Project](https://imaei.github.io/project_pages/ario/)]

- **LIBERO**: Benchmark for lifelong robot learning (130 tasks). [[Paper](https://arxiv.org/abs/2306.03310)] [[Project](https://libero-project.github.io/)]

- **RoboMIND**: Multi-embodiment intelligence benchmark. [[Paper](https://arxiv.org/abs/2412.13877)] [[Project](https://x-humanoid-robomind.github.io/)]

- **VLABench**: Long-horizon reasoning benchmark. [[Paper](https://arxiv.org/abs/2412.18194)] [[Project](https://vlabench.github.io/)]

- **SIMPLER**: Sim-to-real evaluation framework. [[Paper](https://arxiv.org/abs/2405.05941)] [[Project](https://simpler-env.github.io/)]

- **RoboCasa**: Large-scale household task simulation. [[Paper](https://arxiv.org/abs/2407.10943)] [[Project](https://robocasa.ai/)]

- **CALVIN**: Long-horizon language-conditioned manipulation. [[Paper](https://arxiv.org/abs/2112.03227)] [[Project](http://calvin.cs.uni-freiburg.de/)]

- **RLBench**: 100 diverse manipulation tasks. [[Paper](https://arxiv.org/abs/1909.12271)] [[Project](https://sites.google.com/view/rlbench)]

- **ARNOLD**: Language-grounded task learning in realistic 3D. [[Paper](https://arxiv.org/abs/2304.04321)] [[Project](https://arnold-benchmark.github.io/)]

- **ALFRED**: Vision-language navigation and manipulation. [[Paper](https://arxiv.org/abs/1912.01734)] [[Project](https://askforalfred.com/)]

- **GenSim / GenSim2**: LLM-based procedural task generation. [[Paper](https://arxiv.org/abs/2310.01361)] [[Project](https://gen-sim.github.io/)]

- **MineDojo**: Minecraft-based platform with YouTube pretraining. [[Paper](https://arxiv.org/abs/2206.08853)] [[Project](https://minedojo.org/)]

### Simulation Platforms

- **ManiSkill3**: GPU-parallelized robotics simulation. [[Paper](https://arxiv.org/abs/2410.00425)] [[Project](https://www.maniskill.ai/)]

- **Genesis**: Differentiable physics engine. [[Project](https://genesis-embodied-ai.github.io/)]

- **Isaac Lab / Isaac Sim**: NVIDIA's robotics simulation. [[Project](https://developer.nvidia.com/isaac-sim)]

- **MuJoCo Playground**: Browser-based MuJoCo. [[Project](https://playground.mujoco.org/)]

- **OmniGibson**: High-fidelity home simulation. [[Paper](https://arxiv.org/abs/2311.01014)] [[Project](https://behavior.stanford.edu/omnigibson/)]

- **Habitat 2.0**: Navigation and rearrangement simulation. [[Paper](https://arxiv.org/abs/2106.14405)] [[Project](https://aihabitat.org/)]

- **BEHAVIOR-1K**: 1,000 everyday activities benchmark. [[Paper](https://arxiv.org/abs/2403.09227)] [[Project](https://behavior.stanford.edu/)]

- **iGibson**: Interactive environments with object state changes. [[Paper](https://arxiv.org/abs/2012.02924)] [[Project](https://svl.stanford.edu/igibson/)]

- **RoboSuite**: Modular manipulation framework. [[Paper](https://arxiv.org/abs/2009.12293)] [[Project](https://robosuite.ai/)]

- **PyBullet**: Lightweight physics engine for RL. [[Project](https://pybullet.org/)]

---

## Surveys

- "Foundation Models in Robotics: Applications, Challenges, and the Future", *IJRR 2024*. [[Paper](https://arxiv.org/abs/2312.07843)] [[GitHub](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]

- "Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.08782)]

- "Robot Learning in the Era of Foundation Models: A Survey", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.14379)]

- "Language-conditioned Learning for Robotic Manipulation: A Survey", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.10807)]

- "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.02816)]

- "Understanding World or Predicting Future? A Comprehensive Review of World Models", *arXiv, 2024*. [[Paper](https://arxiv.org/abs/2406.XXXXX)]

- "The Development of LLMs for Embodied Navigation", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.00530)]

- "A Survey of Embodied Learning for Object-Centric Robotic Manipulation", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.11537)]

---

## Contributing

We welcome contributions! Please submit a pull request to add relevant papers, correct errors, or improve organization.

### Guidelines

- Focus on **Physical AI** papers (robotics, embodied agents, VLAs)
- Each paper should appear in only one category
- Include proper citations with links to papers, projects, and code
- Verify all links are working

---

## Acknowledgments

This list draws inspiration from:
- [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics)
- [Awesome-Generalist-Agents](https://github.com/cheryyunl/awesome-generalist-agents)
- [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)
