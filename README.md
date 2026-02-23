# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of academic papers and resources on **Physical AI** — focusing on Vision-Language-Action (VLA) models, world models, embodied ai, and robotic foundation models.

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
  - [Discrete Diffusion VLAs](#discrete-diffusion-vlas)
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
- [Sim-to-Real Transfer](#sim-to-real-transfer)
- [Surveys](#surveys)
- [Resources](#resources)
  - [Datasets & Benchmarks](#datasets--benchmarks)
  - [Simulation Platforms](#simulation-platforms)
- [Companies & Projects](#companies--projects)
- [Related Works](#related-works)

---

## Foundations

### Vision-Language Backbones

> Core vision-language models that serve as pretrained backbones for Physical AI systems.

- **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision", *ICML 2021*. [[Paper](https://arxiv.org/abs/2103.00020)] [[Code](https://github.com/openai/CLIP)]
  - Foundational model aligning vision and language that underlies most VLA perception systems.

- **SigLIP**: "Sigmoid Loss for Language Image Pre-Training", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.15343)]

- **PaLI-X**: "PaLI-X: On Scaling up a Multilingual Vision and Language Model", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2305.18565)]

- **LLaVA**: "Visual Instruction Tuning", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2304.08485)] [[Project](https://llava-vl.github.io/)]

- **Prismatic VLMs**: "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.07865)] [[Code](https://github.com/TRI-ML/prismatic-vlms)]
  - Systematic study of VLM design choices informing OpenVLA and other robotics VLMs.

### Visual Representations

> Self-supervised visual encoders and perception models used in robotics.

- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision", *arXiv, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.07193)] [[Code](https://github.com/facebookresearch/dinov2)]

- **SAM**: "Segment Anything", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2304.02643)] [[Project](https://segment-anything.com/)]

- **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2203.12601)] [[Code](https://github.com/facebookresearch/r3m)]

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

- **π0.5**: "π0.5: A Vision-Language-Action Model with Open-World Generalization", *Physical Intelligence, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.16054)] [[Project](https://www.pi.website/blog/pi05)] [[Code](https://github.com/Physical-Intelligence/openpi)]

- **π0.6**: "π0.6: A VLA that Learns from Experience", *Physical Intelligence, 2025*. [[Blog](https://www.physicalintelligence.company/blog/pistar06)]

- **GR-3**: "GR-3 Technical Report", *ByteDance Seed, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.15493)] [[Project](https://seed.bytedance.com/GR3)]

- **UniVLA**: "UniVLA: Unified Vision-Language-Action Model", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.19850)] [[Code](https://github.com/baaivision/UniVLA)]
  - **TL;DR**: Models vision, language, and actions as single interleaved stream of discrete tokens (VQ image tokens + FAST/DCT action tokens) in 8.5B autoregressive VLA. Two training stages: (1) post-train VLM with text/images to predict future frames, (2) finetune with vision and action token prediction. Emphasis on post-training stage to align VLM representations with robot tasks. Strong results on CALVIN, LIBERO, and SimplerEnv-Bridge.

- **SpatialVLA**: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.15830)]

- **AgiBot World**: "AgiBot World Colosseo: Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems", *AgiBot, 2025*. [[Paper](https://agibot-world.com/blog/agibot_go1.pdf)] [[Project](https://agibot-world.com/blog/go1)]

- **EnerVerse**: "EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation", *AgiBot, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.01895)]

- **Genie Envisioner**: "Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation", *AgiBot, 2025*. [[Paper](https://arxiv.org/abs/2508.05635)] [[Project](https://genie-envisioner.github.io/)] [[Code](https://github.com/AgibotTech/Genie-Envisioner)]

- **GraspVLA**: "GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2505.03233)] [[Project](https://pku-epic.github.io/GraspVLA-web/)] [[Code](https://github.com/PKU-EPIC/GraspVLA)]

- **VLA-0**: "VLA-0: Building State-of-the-Art VLAs with Zero Modification", *NVIDIA, 2025*. [[Paper](https://arxiv.org/abs/2510.13054)] [[Project](https://vla0.github.io/)] [[Code](https://github.com/NVlabs/vla0)]

- **ThinkAct**: "ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning", *NVIDIA, NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2507.16815)] [[Project](https://jasper0314-huang.github.io/thinkact-vla/)]

- **OmniVLA**: "OmniVLA: Physically-Grounded Multimodal VLA with Unified Multi-Sensor Perception", *Microsoft Research, 2025*. [[Paper](https://arxiv.org/abs/2511.01210)]

- **V-JEPA 2**: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", *Meta, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.09985)] [[Project](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)] [[Code](https://github.com/facebookresearch/vjepa2)]

- **RoboBrain**: "RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete", *CVPR 2025*. [[Paper](https://arxiv.org/abs/2502.21257)] [[Project](https://superrobobrain.github.io/)]

- **DexGraspVLA**: "DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping", *PsiBot, 2025*. [[Paper](https://arxiv.org/abs/2502.20900)] [[Project](https://dexgraspvla.github.io/)]

- **Hi Robot**: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models", *Physical Intelligence, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.19417)] [[Project](https://www.pi.website/research/hirobot)]

- **Motus**: "Motus: A Unified Latent Action World Model", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2512.13030)] [[Project](https://motus-robotics.github.io/motus)] [[Code](https://github.com/thu-ml/Motus)]

- **GR-RL**: "GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation", *ByteDance Seed, Dec 2024*. [[Paper](https://arxiv.org/abs/2512.01801)] [[Project](https://seed.bytedance.com/gr_rl)]

- **StarVLA**: "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing", *arXiv, 2025*. [[Report](https://www.overleaf.com/read/qqtwrnprctkf#d5bdce)] [[Code](https://github.com/starVLA/starVLA)]

- **InternVLA-A1**: "InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation", *arXiv, Jan 2026*. [[Paper](https://arxiv.org/abs/2601.02456)] [[Code](https://github.com/InternRobotics/InternVLA-A1)]

- **InternVLA-M1**: "InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.13778)] [[Project](https://internrobotics.github.io/internvla-m1.github.io/)] [[Code](https://github.com/InternRobotics/InternVLA-M1)]

- **Cosmos Policy**: "Fine-Tuning Video Models for Visuomotor Control and Planning", *ICLR 2026 Submission*.
  - **TL;DR**: Finetunes NVIDIA Cosmos Video Foundation Model for action prediction. Core idea: inject additional modalities like future action chunks or value function estimations into latent token sequence. Good results on LIBERO with real-world comparisons against Pi0.5.

- **Disentangled Robot Learning**: "Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining", *ICLR 2026 Submission*.
  - **TL;DR**: Novel approach pretraining separate forward and inverse dynamics models. In second stage, combines them for coupled policy finetuning. Good results on CALVIN, decent on SIMPLER.

- **XR-1**: "Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations", *ICLR 2026 Submission*.
  - **TL;DR**: Introduces Unified Vision-Motion Codes (UVMC), a discrete latent representation jointly encoding visual dynamics and robotic motion using dual-branch VQ-VAE with shared codebook. Enables better co-pretraining from human and robot demonstrations. Tested against Groot-N.1.5 and Pi0 with good real-world results.

- **VLM4VLA**: "Revisiting Vision-Language-Models in Vision-Language-Action Models", *ICLR 2026 Submission*.
  - **TL;DR**: Comprehensive comparison of VLMs as backbone choice for VLAs. Finds downstream VLA performance has no correlation with VLM performance on standard benchmarks. Important finding for VLA architecture design, though limited to benchmark setups without real robot results.

- **FLOWER**: "FLOWER: Flow-based VLA for CALVIN Benchmark", *arXiv, 2025*.
  - State-of-the-art on CALVIN benchmarks using flow-based action generation.

- **EO-1**: "EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.21112)] [[Project](https://eo-robotics.ai/eo-1)] [[Code](https://github.com/eo-robotics/EO-1)]

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

- **HybridVLA**: "HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.10631)] [[Project](https://hybrid-vla.github.io/)]

- **CoT-VLA**: "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models", *CVPR 2025*. [[Paper](https://arxiv.org/abs/2503.22020)] [[Project](https://cot-vla.github.io/)]

- **OpenHelix**: "OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.03912)] [[Project](https://openhelix-robot.github.io/)]

- **OneTwoVLA**: "OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.11917)] [[Project](https://one-two-vla.github.io/)]

- **Hume**: "Hume: Introducing System-2 Thinking in Visual-Language-Action Model", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.21432)] [[Project](https://hume-vla.github.io/)]

- **RationalVLA**: "RationalVLA: A Rational Vision-Language-Action Model with Dual System", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.10826)] [[Project](https://irpn-eai.github.io/rational_vla/)]

- **Fast-in-Slow**: "Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.01953)] [[Project](https://fast-in-slow.github.io/)]

- **TriVLA**: "TriVLA: A Unified Triple-System-Based Unified Vision-Language-Action Model", *arXiv, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.01424)] [[Project](https://zhenyangliu.github.io/TriVLA/)]

- **DualVLA**: "DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action", *arXiv, Nov 2025*. [[Paper](https://arxiv.org/abs/2511.22134)] [[Project](https://costaliya.github.io/DualVLA/)]

### Compact & Efficient VLAs

> Lightweight VLA models optimized for fast inference and edge deployment.

- **TinyVLA**: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2409.12514)] [[Project](https://tiny-vla.github.io/)]

- **SmolVLA**: "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", *arXiv, Jun 2025*. [[Paper](https://huggingface.co/blog/smolvla)] [[Code](https://github.com/huggingface/lerobot)]
  - 450M parameters achieving comparable performance to 10x larger models.

- **OpenVLA-OFT**: "OpenVLA-OFT: Efficient Fine-Tuning for Open Vision-Language-Action Models", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]

- **RT-H**: "RT-H: Action Hierarchies Using Language", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.01823)] [[Project](https://rt-hierarchy.github.io/)]

- **LAPA**: "Latent Action Pretraining from Videos", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.11758)] [[Project](https://latentactionpretraining.github.io/)]

- **BitVLA**: "BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.07530)] [[Code](https://github.com/ustcwhy/BitVLA)]

- **MoLe-VLA**: "MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20384)] [[Project](https://sites.google.com/view/mole-vla)]

- **VLA-Cache**: "VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.02175)]

- **NORA**: "NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.19854)] [[Project](https://declare-lab.github.io/nora)]

- **NORA-1.5**: "NORA-1.5: A Vision-Language-Action Model Trained using World Model and Action-based Preference Rewards", *arXiv, Nov 2025*. [[Paper](https://arxiv.org/abs/2511.14659)] [[Project](https://declare-lab.github.io/nora-1.5)] [[Code](https://github.com/declare-lab/nora-1.5)]

- **CEED-VLA**: "CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.13725)] [[Project](https://irpn-eai.github.io/CEED-VLA/)]

- **Running VLAs at Real-time Speed**: *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.26742)]

- **Cross-Platform Scaling of VLAs**: "Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.11480)]

- **VLA-Adapter**: "VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.09372)] [[Project](https://vla-adapter.github.io/)]

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

- **FASTer**: "FASTer: Toward Powerful and Efficient Autoregressive VLAs with Learnable Action Tokenizer and Block-Wise Decoding", *ICLR 2026 Submission*.
  - **TL;DR**: Novel discrete action tokenizer combining Residual Vector Quantization (RVQ) with frequency L1 loss (DCT) and time domain L1 loss. Patchifies action tokens along temporal and grouped action dimension axes (e.g., base motion, arm joints). Higher compression ratio than FAST with strong results on SIMPLER and LIBERO.

- **OmniSAT**: "OmniSAT: Compact Action Token, Faster Autoregression for Vision-Language-Action Models", *ICLR 2026 Submission*.
  - **TL;DR**: VLA tokenizer using B-Splines for compact action representation. Two-stage encoding: (1) aligns different action chunk lengths into normalized fixed-length representation, (2) B-Spline encoder for compact representation, then VQ-VAE for discrete tokens. Improves upon both FAST and BEAST across LIBERO and SIMPLER.

- **VQ-VLA**: "VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers", *ICCV 2025*. [[Paper](https://arxiv.org/abs/2507.01016)] [[Project](https://xiaoxiao0406.github.io/vqvla.github.io)]

### Discrete Diffusion VLAs

> Models using discrete diffusion for parallel action token generation instead of autoregressive decoding.

- **Discrete Diffusion VLA**: "Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.20072)]
  - **TL;DR**: Takes OpenVLA and applies discrete diffusion for fast action chunk-based generation of discrete action tokens. Proposes adaptive decoding for inference with strong results on LIBERO + SIMPLER.

- **dVLA**: "Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought", *ICLR 2026 Submission*.
  - **TL;DR**: Discrete diffusion VLA using co-generation for future frames, text, and actions leveraging fast parallel sampling over AR models. Essentially ECoT + discrete diffusion done well, with good results on LIBERO + real world experiments.

- **DIVA**: "Discrete Diffusion Vision-Language-Action Models for Parallelized Action Generation", *ICLR 2026 Submission*.
  - **TL;DR**: Discrete diffusion VLA focusing on how to substitute tokens during inference for better performance through optimized token replacement strategies.

- **Unified Diffusion VLA**: "Unified Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process", *ICLR 2026 Submission*.
  - **TL;DR**: Generates future frames and discrete actions together with block-wise causal masking. Good results on CALVIN, LIBERO, and SIMPLER.

- **LLaDA-VLA**: "Vision Language Diffusion Action Models", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2506.14932)] [[Project](https://wenyuqing.github.io/llada-vla/)]

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

- **Dita**: "Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2410.15959v4)] [[Project](https://robodita.github.io/)]

- **Real-Time Chunking**: "Real-Time Execution of Action Chunking Flow Policies", *Physical Intelligence, NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2506.07339)] [[Project](https://www.pi.website/research/real_time_chunking)]

- **ManiFlow**: "ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2509.01819)] [[Project](https://maniflow-policy.github.io/)] [[Code](https://github.com/geyan21/ManiFlow_Policy)]

- **Unified Video Action Model**: "Unified Video Action Model", *RSS 2025*. [[Paper](https://arxiv.org/abs/2503.00200)] [[Project](https://unified-video-action-model.github.io/)]

- **Streaming Flow Policy**: "Streaming Flow Policy: Simplifying diffusion/flow-matching policies", *CoRL 2025 Oral*. [[Paper](https://arxiv.org/abs/2505.21851)] [[Project](https://siddancha.github.io/streaming-flow-policy/)]

- **FlowPolicy**: "FlowPolicy: Enabling Fast and Robust 3D Flow-Based Policy", *AAAI 2025*. [[Paper](https://arxiv.org/abs/2412.04987)] [[Code](https://github.com/zql-kk/FlowPolicy)]

- **MoDE**: "Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers", *ICLR 2025*. [[Paper](https://arxiv.org/abs/2412.12953)] [[Project](https://mbreuss.github.io/MoDE_Diffusion_Policy/)]

- **Reactive Diffusion Policy**: "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning", *RSS 2025*. [[Paper](https://arxiv.org/abs/2503.02881)] [[Project](https://reactive-diffusion-policy.github.io/)]

- **VITA**: "VITA: Vision-To-Action Flow Matching Policy", *arXiv, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.13231)] [[Project](https://ucd-dare.github.io/VITA/)]

- **Chain-of-Action**: "Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation", *ByteDance Seed, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.09990)] [[Project](https://chain-of-action.github.io/)]

- **Hierarchical Diffusion Policy**: "Hierarchical Diffusion Policy: Manipulation Trajectory Generation Via Contact Guidance", *TRO 2025*. [[Paper](https://arxiv.org/abs/2411.12982)] [[Code](https://github.com/dexin-wang/Hierarchical-Diffusion-Policy)]

- **Adapt3R**: "Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.04877)] [[Project](https://www.pair.toronto.edu/Adapt3R/)]

- **3D FlowMatch Actor**: "3D FlowMatch Actor: Unified 3D Policy for Single and Dual-Arm Manipulation", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.11002)] [[Project](https://3d-flowmatch-actor.github.io/)] [[Code](https://github.com/nickgkan/3d_flowmatch_actor)]

- **CLAM**: "CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.04999)] [[Project](https://clamrobot.github.io/)]

- **H3DP**: "H3DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.07819)] [[Project](https://lyy-iiis.github.io/h3dp/)]

- **UniSkill**: "UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.08787)] [[Project](https://kimhanjung.github.io/UniSkill/)]

- **Latent Action Diffusion**: "Latent Action Diffusion for Cross-Embodiment Manipulation", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.14608)] [[Project](https://mimicrobotics.github.io/lad/)]

- **Dex1B**: "Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation", *RSS 2025*. [[Paper](https://arxiv.org/abs/2506.17198)] [[Project](https://jianglongye.com/dex1b/)]

- **DemoDiffusion**: "DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.20668)] [[Project](https://demodiffusion.github.io/)]

- **One-Step Diffusion Policy**: "One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.21257)]

- **GauDP**: "GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies", *NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2511.00998)] [[Project](https://ziyeeee.github.io/gaudp.io/)]

- **DiWA**: "DiWA: Diffusion Policy Adaptation with World Models", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2508.03645)] [[Project](https://diwa.cs.uni-freiburg.de/)] [[Code](https://github.com/acl21/diwa)]

- **GPC**: "Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.01068)] [[Project](https://sagecao1125.github.io/GPC-Site/)] [[Code](https://github.com/SageCao1125/GPC)]
  - **TL;DR**: Composes flow/diffusion-based VLA policies at test time using convex optimization and test-time search to combine scores from multiple policies. Improves performance without additional training by leveraging energy-based formulation that allows summing model scores.

---

## World Models

### JEPA & Latent Prediction

> Joint-Embedding Predictive Architecture (JEPA) predicts future latent states rather than pixels.

- "A Path Towards Autonomous Machine Intelligence", *Meta AI, Jun 2022*. [[Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)]
  - LeCun's foundational vision describing the "world model in the middle" cognitive architecture.

- **I-JEPA**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", *CVPR 2023*. [[Paper](https://arxiv.org/abs/2301.08243)] [[Code](https://github.com/facebookresearch/ijepa)]

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

- "Learning Universal Policies via Text-Guided Video Generation", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.00111)] [[Project](https://universal-policy.github.io/unipi/)]

- **SIMA**: "Scaling Instructable Agents Across Many Simulated Worlds", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2404.10179)] [[Blog](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)]

- **UniSim**: "UniSim: Learning Interactive Real-World Simulators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.06114)] [[Project](https://universal-simulator.github.io/unisim/)]

---

## Reasoning & Planning

### Chain-of-Thought & Deliberation

> Models implementing "thinking before acting" with explicit reasoning or value-guided search.

- **Hume**: "Hume: Introducing Deliberative Alignment in Embodied AI", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **Embodied-CoT**: "Robotic Control via Embodied Chain-of-Thought Reasoning", *arXiv, Jul 2024*. [[Paper](https://arxiv.org/abs/2407.08693)] [[Project](https://embodied-cot.github.io/)]

- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR 2023*. [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]

- **ReKep**: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.01652)] [[Project](https://rekep-robot.github.io/)]

- **TraceVLA**: "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.10345)] [[Project](https://tracevla.github.io/)]

- **LLM-State**: "LLM-State: Open World State Representation for Long-horizon Task Planning", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.17406)]

- **Statler**: "Statler: State-Maintaining Language Models for Embodied Reasoning", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.17840)] [[Project](https://statler-lm.github.io/)]

- **RoboReflect**: "RoboReflect: Reflective Reasoning for Robot Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **Cosmos-Reason1**: "Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning", *NVIDIA, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.15558)] [[Code](https://github.com/nvidia-cosmos/cosmos-reason1)]

- **EmbodiedVSR**: "EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.11089)]

- **Reflective Planning**: "Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.16707)]

- **Embodied-Reasoner**: "Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.21696)] [[Project](https://embodied-reasoner.github.io/)]

- **Embodied-R**: "Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning via Reinforcement Learning", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.12680)] [[Project](https://embodiedcity.github.io/Embodied-R/)]

- **RoBridge**: "RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.01709)] [[Project](https://abliao.github.io/RoBridge/)]

- **Visual Embodied Brain**: "Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.00123)] [[Code](https://github.com/OpenGVLab/VeBrain)]

- **From Seeing to Doing**: "From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.08548)] [[Project](https://embodied-fsd.github.io/)]

- **Actions as Language**: "Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting", *ICLR 2026 Submission*.
  - **TL;DR**: Instead of directly fine-tuning VLMs with discrete action tokens (which causes catastrophic forgetting), relabels robot datasets with subtasks, actions as text, and intermediate motion-planning phrases like "move left". Bridges VLM domain gap without reducing VQA benchmark performance. Cheap LoRA finetuning achieves strong action prediction while maintaining VLM reasoning.

- **InstructVLA**: "Vision-Language-Action Instruction Tuning: From Understanding to Manipulation", *arXiv, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.17520)] [[Project](https://yangs03.github.io/InstructVLA_Home/)]
  - **TL;DR**: Two-stage Vision-Language-Action Instruction Tuning pipeline: (1) pretrain action expert and latent action interface, (2) instruction-tune MoE-adapted VLM to switch between textual reasoning and latent action generation. Decouples multimodal reasoning from action generation to avoid catastrophic forgetting. Introduces instruction-based SIMPLER benchmark.

- **Hybrid ECoT Training**: "Hybrid Training for Vision-Language-Action Models", *ICLR 2026 Submission*.
  - **TL;DR**: Decomposes ECoT pretraining into think/act/follow subtasks that maintain performance benefits while enabling fast inference. Shows co-training with ECoT objectives results in better representations for action prediction.

- **HAMLET**: "Switch Your Vision-Language-Action Model into a History-Aware Policy", *ICLR 2026 Submission*.
  - **TL;DR**: Plug-and-play memory module with moment tokens to capture temporal information from prior timesteps. Memory module aggregates tokens over time for history-conditioned prediction, addressing the limitation that most VLAs only encode current images.

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

- **Play-LMP**: "Learning Latent Plans from Play", *CoRL 2019*. [[Paper](https://arxiv.org/abs/1903.01973)] [[Project](https://learning-from-play.github.io/)]
  - Learns reusable skills from unstructured "play" data without task labels.

- **MimicPlay**: "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2302.12422)] [[Project](https://mimic-play.github.io/)]

- **RVT**: "RVT: Robotic View Transformer for 3D Object Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.14896)] [[Project](https://robotic-view-transformer.github.io/)] [[Code](https://github.com/NVlabs/RVT)]

- **RVT-2**: "RVT-2: Learning Precise Manipulation from Few Demonstrations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2406.08545)] [[Project](https://robotic-view-transformer-2.github.io/)]

- **DIAL**: "Robotic Skill Acquisition via Instruction Augmentation", *arXiv, Nov 2022*. [[Paper](https://arxiv.org/abs/2211.11736)] [[Project](https://instructionaugmentation.github.io/)]

- **Perceiver-Actor**: "A Multi-Task Transformer for Robotic Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2209.05451)] [[Project](https://peract.github.io/)] [[Code](https://github.com/peract/peract)]

- **BOSS**: "Bootstrap Your Own Skills: Learning to Solve New Tasks with LLM Guidance", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2310.10021)] [[Project](https://clvrai.github.io/boss/)]

- **Phantom**: "Phantom: Training Robots Without Robots Using Only Human Videos", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.00779)] [[Project](https://phantom-human-videos.github.io/)]

- **ZeroMimic**: "ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.23877)] [[Project](https://zeromimic.github.io/)]

- **Human2Robot**: "Human2Robot: Learning Robot Actions from Paired Human-Robot Videos", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.16587)]

- **One-Shot Dual-Arm Imitation Learning**: *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.06831)] [[Project](https://www.robot-learning.uk/one-shot-dual-arm)]

- **DataMIL**: "DataMIL: Selecting Data for Robot Imitation Learning with Datamodels", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.09603)] [[Project](https://robin-lab.cs.utexas.edu/datamodels4imitation/)]

- **In-Context Imitation Learning**: "In-Context Imitation Learning via Next-Token Prediction", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.15980)] [[Project](https://icrt.dev/)]

- **RILe**: "RILe: Reinforced Imitation Learning", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2406.08472)]

- **CLASS**: "CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2508.01600)] [[Project](https://class-robot.github.io/)] [[Code](https://github.com/sean1295/CLASS)]

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

- **VLA-RL**: "VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable RL", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.18719)] [[Code](https://github.com/GuanxingLu/vlarl)]

- **SimpleVLA-RL**: "SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.09674)] [[Code](https://github.com/PRIME-RL/SimpleVLA-RL)]

- **ConRFT**: "ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy", *RSS 2025*. [[Paper](https://arxiv.org/abs/2502.05450)] [[Project](https://cccedric.github.io/conrft/)]

- **VLA-RFT**: "VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.00406)] [[Project](https://vla-rft.github.io/)] [[Code](https://github.com/OpenHelix-Team/VLA-RFT)]

- **RLinf-VLA**: "RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.06710)] [[Code](https://github.com/RLinf/RLinf)]

- **RoboMonkey**: "RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2506.17811)] [[Project](https://robomonkey-vla.github.io/)] [[Code](https://github.com/robomonkey-vla/RoboMonkey)]

- **Embodied-R1**: "Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.13998)] [[Project](https://embodied-r1.github.io/)] [[Code](https://github.com/pickxiguapi/Embodied-R1)]
  - **TL;DR**: Pointing VLM for embodied reasoning trained via two-stage Reinforced Fine-Tuning (RFT) curriculum on Embodied-Points-200K dataset. Uses embodiment-agnostic intermediates: REG (point to referred object), RRG (point to relation-defined place), OFG (point to functional part like handle), VTG (output point sequence as visual trace). Strong embodied benchmark performance with good SIMPLER generalization as planner.

- **VLA-Reasoner**: "VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.22643)]

- **Residual RL for VLAs**: "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL", *ICLR 2026 Submission*.
  - **TL;DR**: Residual RL method collecting data with frozen VLA and small residual policy. Residual interventions provide high-quality data with recovery behavior. Finally finetunes VLA using SFT. Achieves 99% on LIBERO.

- **STA-PPO/TPO**: "Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models", *ICLR 2026 Submission*.
  - **TL;DR**: Breaks robot tasks into semantic stages (Reach→Grasp→Transport→Place) and assigns rewards to each stage instead of whole trajectory. Uses STA-TPO for offline preference learning and STA-PPO for online RL, both at stage level. Achieves 98% on Bridge SIMPLER.

- **Verifier-free Test-Time Sampling**: "Verifier-free Test-Time Sampling for Vision Language Action Models", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.05681)]

### Reward Design

> Automated reward function generation using language models.

- **Text2Reward**: "Text2Reward: Automated Dense Reward Function Generation", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.11489)] [[Project](https://text-to-reward.github.io/)]

- **Language to Rewards**: "Language to Rewards for Robotic Skill Synthesis", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.08647)] [[Project](https://language-to-reward.github.io/)]

- **ExploRLLM**: "ExploRLLM: Guiding Exploration in Reinforcement Learning with LLMs", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.09583)]

---

## Scaling & Generalization

### Scaling Laws

> Mathematical relationships between model/data scale and robotic performance.

- "Neural Scaling Laws for Embodied AI", *arXiv, May 2024*. [[Paper](https://arxiv.org/abs/2405.14005)]

- "Data Scaling Laws in Imitation Learning for Robotic Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.18647)] [[Project](https://data-scaling-laws.github.io/)]

- **AutoRT**: "AutoRT: Embodied Foundation Models for Large Scale Orchestration", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2401.12963)] [[Project](https://auto-rt.github.io/)]

- **SARA-RT**: "SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.01990)]

- "Scaling Robot Learning with Semantically Imagined Experience", *RSS 2023*. [[Paper](https://arxiv.org/abs/2302.11550)]

### Cross-Embodiment Transfer

> Single policies controlling diverse robot types.

- **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]

- **GENBOT-1K**: "Towards Embodiment Scaling Laws: Training on ~1000 Robot Bodies", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Training on ~1,000 robot bodies enables zero-shot transfer to unseen robots.

- **Crossformer**: "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2408.11812)] [[Project](https://crossformer-model.github.io/)]
  - Single policy controlling manipulators, legged robots, and drones.

- **HPT**: "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20537)] [[Project](https://liruiw.github.io/hpt/)]

- **MetaMorph**: "MetaMorph: Learning Universal Controllers with Transformers", *ICLR 2022*. [[Paper](https://arxiv.org/abs/2203.11931)] [[Project](https://metamorph-iclr.github.io/)]

- **RUMs**: "Robot Utility Models: General Policies for Zero-Shot Deployment", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.05865)] [[Project](https://robotutilitymodels.com/)]

- **URMA**: "Unified Robot Morphology Architecture", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **RoboAgent**: "RoboAgent: Generalization and Efficiency via Semantic Augmentations", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2309.01918)] [[Project](https://robopen.github.io/)]

- **X-VLA**: "Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.10274)] [[Project](https://thu-air-dream.github.io/X-VLA/)] [[Code](https://github.com/2toinf/X-VLA)]
  - **TL;DR**: Tackles cross-action-space learning using soft-prompting tokens for different datasets. Soft-prompt tokens are learnable readout-tokens for the VLA. Strong results on LIBERO, CALVIN, SIMPLER, RoboTwin, and VLABench. Includes insightful scaling analysis ablating pretraining design decisions.

- **HiMoE-VLA**: "Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies", *arXiv, Dec 2025*. [[Paper](https://arxiv.org/abs/2512.05693)] [[Code](https://github.com/ZhiyingDu/HiMoE-VLA)]
  - **TL;DR**: Substitutes Pi-style action expert with Hierarchical Mixture-of-Experts Transformer for better embodiment adaptation. Interleaves standard blocks with Action-Space MoEs and Heterogeneity Balancing MoEs to handle different action spaces. Improves upon Pi0 across experiments.

- **D2E**: "Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.05684)] [[Project](https://worv-ai.github.io/d2e/)] [[Code](https://github.com/worv-ai/D2E)]

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

- **DeeR-VLA**: "DeeR-VLA: Dynamic Inference of Multimodal LLMs for Efficient Robot Execution", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.02359)] [[Code](https://github.com/yueyang130/DeeR-VLA)]

- **QuaRT-VLA**: "Quantized Robotics Transformers for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **PDVLA**: "PDVLA: Parallel Decoding for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

- **HyperVLA**: "Efficient Inference in Vision-Language-Action Models via Hypernetworks", *ICLR 2026 Submission*.
  - **TL;DR**: Uses hypernetworks to generate small task-specific policies conditioned on language instructions and initial images. Dramatically reduces inference cost by only activating the compact generated policy during execution instead of the full VLA model.

- **AutoQVLA**: "Not All Channels Are Equal in Vision-Language-Action Model's Quantization", *ICLR 2026 Submission*.
  - **TL;DR**: Analyzes quantization of OpenVLA and proposes improved quantization method that maintains performance with only 30% of the original VRAM requirements through channel-aware optimization.

- **SpecPrune-VLA**: "Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.05614)]

- **RLRC**: "Reinforcement Learning-based Recovery for Compressed Vision-Language-Action Models", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.17639)] [[Project](https://rlrc-vla.github.io/)]

### Real-Time Control

> Methods bridging high-latency AI inference and low-latency physical control.

- **A2C2**: "A2C2: Asynchronous Action Chunk Correction for Real-Time Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]

- **RTC**: "Real-Time Chunking: Asynchronous Execution for Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]

---

## Safety & Alignment

> Ethical constraints, safety frameworks, and human-robot alignment.

- **Robot Constitution**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)]
  - Introduces data-driven "Robot Constitution" with natural language rules for safe behavior.

- **ASIMOV**: "ASIMOV: A Safety Benchmark for Embodied AI", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]

- **RoboPAIR**: "Jailbreaking LLM-Controlled Robots", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.13691)] [[Project](https://robopair.org/)]

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

- **ApexNav**: "ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation", *RA-L 2025*. [[Paper](https://arxiv.org/abs/2504.14478)] [[Project](https://robotics-star.com/ApexNav)]

- **LOVON**: "LOVON: Legged Open-Vocabulary Object Navigator", *arXiv, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.06747)] [[Project](https://daojiepeng.github.io/LOVON/)]

- **Multimodal Spatial Language Maps**: "Multimodal Spatial Language Maps for Robot Navigation and Manipulation", *IJRR 2025*. [[Paper](https://arxiv.org/abs/2506.06862)] [[Project](https://mslmaps.github.io/)]

- **Learned Perceptive Forward Dynamics Model**: "Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.19322)] [[Code](https://github.com/leggedrobotics/fdm)]

- **VL-Nav**: "VL-Nav: Real-time Vision-Language Navigation with Spatial Reasoning", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.00931)]

- **TRAVEL**: "TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.07306)]

- **VR-Robo**: "VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.01536)]

- **NavigateDiff**: "NavigateDiff: Visual Predictors are Zero-Shot Navigation Assistants", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.13894)]

- **MapNav**: "MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based VLN", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.13451)]

- **OpenFly**: "OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.18041)]

- **WMNav**: "WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.02247)] [[Project](https://b0b8k1ng.github.io/WMNav/)]

- **SmartWay**: "SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.10069)]

- **UniGoal**: "UniGoal: Towards Universal Zero-shot Goal-oriented Navigation", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.10630)] [[Project](https://bagh2178.github.io/UniGoal/)]

- **P3Nav**: "P3Nav: A Unified Framework for Embodied Navigation Integrating Perception, Planning, and Prediction", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.18525)]

- **ForesightNav**: "ForesightNav: Learning Scene Imagination for Efficient Exploration", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.16062)] [[Code](https://github.com/uzh-rpg/foresight-nav)]

- **CityNavAgent**: "CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.05622)] [[Code](https://github.com/VinceOuti/CityNavAgent)]

- **NavDP**: "NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.08712)]

- **OctoNav**: "OctoNav: Towards Generalist Embodied Navigation", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.09839)] [[Project](https://buaa-colalab.github.io/OctoNav/)]

- **BeliefMapNav**: "BeliefMapNav: 3D Voxel-Based Belief Map for Zero-Shot Object Navigation", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.06487)] [[Code](https://github.com/ZiboKNOW/BeliefMapNav)]

- **TopV-Nav**: "TopV-Nav: Unlocking the Top-View Spatial Reasoning Potential of MLLM for Zero-shot Object Navigation", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.16425)]

- **CorrectNav**: "CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.10416)] [[Project](https://correctnav.github.io/)]

- **GC-VLN**: "GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2509.10454)] [[Project](https://bagh2178.github.io/GC-VLN/)] [[Code](https://github.com/bagh2178/GC-VLN)]

- **NavFoM**: "Embodied Navigation Foundation Model", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.12129)] [[Project](https://pku-epic.github.io/NavFoM-Web/)]

- **Search-TTA**: "Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild", *CoRL 2025*. [[Paper](https://arxiv.org/abs/2505.11350)] [[Project](https://search-tta.github.io/)] [[Code](https://github.com/marmotlab/Search-TTA-VLN)]

- **JanusVLN**: "JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.22548)] [[Project](https://miv-xjtu.github.io/JanusVLN.github.io/)] [[Code](https://github.com/MIV-XJTU/JanusVLN)]

- **TrackVLA++**: "TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.07134)] [[Project](https://pku-epic.github.io/TrackVLA-plus-plus-Web/)]

- **InternNav**: "Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation", *arXiv, Dec 2025*. [[Paper](https://arxiv.org/abs/2512.08186)] [[Project](https://internrobotics.github.io/internvla-n1-dualvln.github.io/)] [[Code](https://github.com/InternRobotics/InternNav)]

- **OmniVLA-Nav**: "OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.19480)] [[Project](https://omnivla-nav.github.io/)] [[Code](https://github.com/NHirose/OmniVLA)]

---

## Sim-to-Real Transfer

> Methods for bridging simulation-trained policies to real-world deployment.

- **RE3SIM**: "RE3SIM: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.08645)]

- **Real-to-Sim-to-Real with VLM-Generated Rewards**: "A Real-to-Sim-to-Real Approach to Robotic Manipulation with VLM-Generated Iterative Keypoint Rewards", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.08643)]

- **Distributional Real2Sim2Real**: "A Distributional Treatment of Real2Sim2Real for Vision-Driven Deformable Linear Object Manipulation", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.18615)]

- **Sim-to-Real for Vision-Based Dexterous Manipulation on Humanoids**: "Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.20396)] [[Project](https://toruowo.github.io/recipe/)]

- **Impact of Static Friction on Sim2Real**: "Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.01255)]

- **Few-shot Sim2Real**: "Few-shot Sim2Real Based on High Fidelity Rendering with Force Feedback Teleop", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.01301)]

- **RSR Loop Framework**: "An Real-Sim-Real (RSR) Loop Framework for Generalizable Robotic Policy Transfer with Differentiable Simulation", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.10118)] [[Code](https://github.com/sunnyshi0310/RSR-MJX)]

- **Real2Render2Real**: "Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.09601)] [[Project](https://real2render2real.com/)]

---

## Surveys

> Comprehensive reviews and taxonomies of VLA, world models, and embodied AI research.

- **Foundation Models in Robotics**: "Foundation Models in Robotics: Applications, Challenges, and the Future", *IJRR 2025*. [[Paper](https://arxiv.org/abs/2312.07843)] [[GitHub](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
  - Surveys foundation models across perception, decision-making, and control in robotics.

- **Learning-based Dynamics Models**: "A Review of Learning-based Dynamics Models for Robotic Manipulation", *Science Robotics 2025*. [[Paper](https://albertboai.com/assets/pdf/2025_scirobotics.adt1497.pdf)]
  - Reviews learned dynamics models for predicting physical interactions in manipulation.

- **VLA Anatomy Survey**: "An Anatomy of VLA Models: From Modules to Milestones", *arXiv, Dec 2025*. [[Paper](https://arxiv.org/abs/2512.11362)] [[Project](https://suyuz1.github.io/Survery/)]
  - Structural blueprint of VLA modules, milestones, and five core challenges (representation, execution, generalization, safety, evaluation).

- **World Models for Embodied AI**: "A Comprehensive Survey on World Models for Embodied AI", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.16732)] [[GitHub](https://github.com/Li-Zn-H/AwesomeWorldModels)]
  - Taxonomizes world models for embodied agents across prediction, planning, and simulation.

- **VLA for Real-World Robotics**: "VLA Models for Robotics: Real-World Applications", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.07077)]
  - Full-stack review bridging VLA research to practical robotics deployment.

- **Pure VLA Survey**: "Pure VLA Models: A Comprehensive Survey", *arXiv, Sep 2025*. [[Paper](https://arxiv.org/abs/2509.19012)]
  - Taxonomy of VLA paradigms: autoregressive, diffusion, RL, and hybrid methods.

- **Large VLM-based VLA**: "Large VLM-based VLA for Robotic Manipulation", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.13073)] [[GitHub](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)]
  - Reviews VLAs built on large pretrained VLMs, comparing monolithic vs hierarchical designs.

- **Embodied AI Decision-Making**: "Large Model Empowered Embodied AI: Decision-Making", *arXiv, Aug 2025*. [[Paper](https://arxiv.org/abs/2508.10399)]
  - Surveys how large models enable embodied decision-making and planning.

- **Foundation Model Driven Robotics**: "Foundation Model Driven Robotics", *arXiv, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.10087)]
  - Overview of foundation models transforming perception, planning, and control in robotics.

- **Action Tokenization Survey**: "VLA Survey: An Action Tokenization Perspective", *PKU-PsiBot, Jul 2025*. [[Paper](https://arxiv.org/abs/2507.01925)]
  - Analyzes VLA design through action token formats (language, code, affordance, trajectory, latent).

- **VLA for Autonomous Driving**: "VLA Models for Autonomous Driving", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.24044)] [[GitHub](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)]
  - Surveys VLA applications in end-to-end autonomous driving systems.

- **VLA Post-Training**: "VLA Post-Training and Human Motor Learning", *arXiv, Jun 2025*. [[Paper](https://arxiv.org/abs/2506.20966)] [[GitHub](https://github.com/AoqunJin/Awesome-VLA-Post-Training)]
  - Reviews post-training methods for VLAs including RL finetuning and human feedback.

- **Deep RL for Robotics**: "Deep RL for Robotics: Real-World Successes", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.03539)]
  - Surveys successful real-world deployments of deep RL in robotics.

- **Diffusion Policy Survey**: "Survey on Diffusion Policy for Robotic Manipulation", *TechRxiv 2025*. [[Paper](https://doi.org/10.36227/techrxiv.174378343.39356214/v1)] [[GitHub](https://github.com/HITSZ-Robotics/DiffusionPolicy-Robotics)]
  - Reviews diffusion-based policies for robotic manipulation tasks.

- **Industrial Robotics Survey**: "Embodied Intelligent Industrial Robotics", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.09305)] [[GitHub](https://github.com/jackeyzengl/Embodied_Intelligent_Industrial_Robotics_Paper_List)]
  - Surveys embodied AI for industrial automation and manufacturing.

- **Neural Brain Framework**: "Neural Brain: Framework for Embodied Agents", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.07634)] [[GitHub](https://github.com/CNJianLiu/Neural-Brain-for-Embodied-Agents)]
  - Proposes unified framework viewing embodied agents through neural architecture lens.

- **VLA Concepts Survey**: "VLA Models: Concepts, Progress, Applications", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.04769)]
  - Introductory survey covering VLA fundamentals and application domains.

- **Navigation with Simulators**: "Robotic Navigation with Physics Simulators", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.01458)]
  - Reviews sim-to-real transfer methods for robotic navigation.

- **Multimodal Navigation**: "Multimodal Perception for Goal-oriented Navigation", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.15643)]
  - Surveys multimodal perception fusion for robot navigation tasks.

- **Diffusion for Manipulation**: "Diffusion Models for Robotic Manipulation", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.08438)]
  - Reviews diffusion model applications in manipulation policy learning.

- **Dexterous Manipulation Survey**: "Dexterous Manipulation through Imitation Learning", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.03515)]
  - Surveys imitation learning methods for dexterous robot hands.

- **Multimodal Fusion for Robotics**: "Multimodal Fusion and VLMs for Robot Vision", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.02477)] [[GitHub](https://github.com/Xiaofeng-Han-Res/MF-RV)]
  - Reviews multimodal fusion techniques for robot perception systems.

- **SE(3)-Equivariant Learning**: "SE(3)-Equivariant Robot Learning: Tutorial Survey", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.09829)]
  - Tutorial on incorporating geometric symmetries into robot learning.

- **Generative AI for Manipulation**: "Generative AI in Robotic Manipulation", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.03464)] [[GitHub](https://github.com/GAI4Manipulation/AwesomeGAIManipulation)]
  - Surveys generative models (diffusion, LLMs, VLMs) for manipulation tasks.

- **VLA Survey 2025**: "Survey on Vision-Language-Action Models", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.06851)]
  - AI-generated survey demonstrating automated literature review for VLAs.

- **Embodied Multimodal Models**: "Exploring Embodied Multimodal Large Models", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.15336)]
  - Surveys multimodal LLMs adapted for embodied reasoning and control.

- **General-Purpose Robots Survey**: "General-Purpose Robots via Foundation Models", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.08782)]
  - Early survey on using foundation models for general-purpose robotics.

- **Robot Learning Foundation Models**: "Robot Learning in the Era of Foundation Models", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.14379)]
  - Surveys how foundation models are reshaping robot learning paradigms.

- **Language-conditioned Manipulation**: "Language-conditioned Learning for Manipulation", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.10807)]
  - Reviews language-guided policy learning for manipulation.

- **LLMs for Navigation**: "LLMs for Embodied Navigation", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.00530)]
  - Surveys LLM applications in robot navigation and exploration.

- **Object-Centric Manipulation**: "Embodied Learning for Object-Centric Manipulation", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.11537)]
  - Reviews object-centric representations for manipulation learning.

- **VLA for Embodied AI**: "A Survey on VLA Models for Embodied AI", *arXiv, May 2024*. [[Paper](https://arxiv.org/abs/2405.14093)]
  - Taxonomizes VLAs by component (vision, language, action) and control policy type.

- **Cyber-Physical Alignment**: "Aligning Cyber Space with Physical World", *arXiv, Jul 2024*. [[Paper](https://arxiv.org/abs/2407.06886)]
  - Surveys bridging digital AI models with physical robot execution.

- **VLN Survey**: "Vision-Language Navigation: Survey and Taxonomy", *arXiv, 2024*. [[Paper](https://arxiv.org/abs/2108.11544)]
  - Comprehensive taxonomy of vision-language navigation methods and benchmarks.

- **State of VLA at ICLR 2026**: "State of VLA Research at ICLR 2026", *Blog, Oct 2025*. [[Blog](https://mbreuss.github.io/blog_post_iclr_26_vla.html)]
  - Analysis of VLA trends from ICLR 2026 submissions covering discrete diffusion, ECoT, and tokenizers.

---

## Resources

### Datasets & Benchmarks

| Dataset | Scale | Focus | Links |
|---------|-------|-------|-------|
| Open X-Embodiment | 1M+ trajectories, 22 robots | Cross-embodiment | [Paper](https://arxiv.org/abs/2310.08864) · [Project](https://robotics-transformer-x.github.io/) |
| DROID | 76K trajectories, 564 scenes | In-the-wild manipulation | [Paper](https://arxiv.org/abs/2403.12945) · [Project](https://droid-dataset.github.io/) |
| BridgeData V2 | Multi-task | Few-shot transfer | [Paper](https://arxiv.org/abs/2308.12952) · [Project](https://rail-berkeley.github.io/bridgedata/) |
| ARIO | Unified format | Dataset standardization | [Paper](https://arxiv.org/abs/2408.10899) · [Project](https://imaei.github.io/project_pages/ario/) |
| LIBERO | 130 tasks | Lifelong learning | [Paper](https://arxiv.org/abs/2306.03310) · [Project](https://libero-project.github.io/) |
| RoboMIND | Multi-embodiment | Intelligence benchmark | [Paper](https://arxiv.org/abs/2412.13877) · [Project](https://x-humanoid-robomind.github.io/) |
| VLABench | Long-horizon | Reasoning benchmark | [Paper](https://arxiv.org/abs/2412.18194) · [Project](https://vlabench.github.io/) |
| SIMPLER | Sim-to-real | Policy evaluation | [Paper](https://arxiv.org/abs/2405.05941) · [Project](https://simpler-env.github.io/) |
| RoboCasa | Large-scale | Household tasks | [Paper](https://arxiv.org/abs/2407.10943) · [Project](https://robocasa.ai/) |
| CALVIN | Long-horizon | Language-conditioned | [Paper](https://arxiv.org/abs/2112.03227) · [Project](http://calvin.cs.uni-freiburg.de/) |
| RLBench | 100 tasks | Manipulation benchmark | [Paper](https://arxiv.org/abs/1909.12271) · [Project](https://sites.google.com/view/rlbench) |
| ARNOLD | Realistic 3D | Language-grounded | [Paper](https://arxiv.org/abs/2304.04321) · [Project](https://arnold-benchmark.github.io/) |
| ALFRED | VLN + manipulation | Instruction following | [Paper](https://arxiv.org/abs/1912.01734) · [Project](https://askforalfred.com/) |
| GenSim / GenSim2 | Procedural | Task generation | [Paper](https://arxiv.org/abs/2310.01361) · [Project](https://gen-sim.github.io/) |
| MineDojo | Minecraft | Open-world learning | [Paper](https://arxiv.org/abs/2206.08853) · [Project](https://minedojo.org/) |
| RoboTwin 2.0 | Bimanual manipulation | Domain randomization | [Paper](https://arxiv.org/abs/2506.18088) · [Project](https://robotwin-platform.github.io/) |
| RoboArena | Distributed evaluation | Real-world benchmark | [Paper](https://arxiv.org/abs/2506.18123) · [Project](https://robo-arena.github.io/) |
| RoboCerebra | Long-horizon | Manipulation evaluation | [Paper](https://arxiv.org/abs/2506.06677) · [Project](https://robocerebra.github.io/) |
| DivScene | Diverse scenes | Object navigation | [Paper](https://arxiv.org/abs/2410.02730) · [Project](https://zhaowei-wang-nlp.github.io/divscene-project-page/) |
| EWMBench | World model evaluation | Scene, motion, semantic | [Paper](https://arxiv.org/abs/2505.09694) · [Code](https://github.com/AgibotTech/EWMBench) |
| ManipBench | VLM evaluation | Low-level manipulation | [Paper](https://arxiv.org/abs/2505.09698) · [Project](https://manipbench.github.io/) |
| RoboTwin | CVPR 2025 | Dual-arm benchmark | [Paper](https://arxiv.org/abs/2409.02920) · [Code](https://github.com/TianxingChen/RoboTwin) |
| RoboVerse | Unified platform | Scalable robot learning | [Paper](https://roboverseorg.github.io/static/pdfs/paper_supp_20250405_1820.pdf) · [Code](https://github.com/RoboVerseOrg/RoboVerse) |
| AutoEval | Autonomous evaluation | Real-world manipulation | [Paper](https://arxiv.org/abs/2503.24278) · [Project](https://auto-eval.github.io/) |
| RoboFactory | Multi-robot collaboration | Compositional tasks | [Paper](https://arxiv.org/abs/2503.16408) · [Project](https://iranqin.github.io/robofactory/) |
| BOSS | Observation shift | Long-horizon tasks | [Paper](https://arxiv.org/abs/2502.15679) |
| OpenBench | Smart logistics | Semantic navigation | [Paper](https://arxiv.org/abs/2502.09238) |
| EmbSpatial-Bench | Spatial understanding | Embodied tasks | [Paper](https://arxiv.org/abs/2406.05756) |
| Diverse Behaviors Benchmark | Human demonstrations | Imitation learning | [Paper](https://arxiv.org/abs/2402.14606) |
| RoboArena | Zero-shot evaluation | Real-world generalization | [Project](https://robo-arena.github.io/) |
| RobotArena ∞ | Real-to-sim | Automatic evaluation | ICLR 2026 Submission |
| RoboCasa365 | 365 tasks, 2k+ scenes | Kitchen manipulation | ICLR 2026 Submission |
| WorldGym | World model evaluation | Policy evaluation | ICLR 2026 Submission |
| ManipulationNet | Zero-shot benchmark | Fair comparison | [Project](https://manipulation-net.github.io/) |

### Simulation Platforms

| Platform | Focus | Links |
|----------|-------|-------|
| ManiSkill3 | GPU-parallelized robotics | [Paper](https://arxiv.org/abs/2410.00425) · [Project](https://www.maniskill.ai/) |
| Genesis | Differentiable physics | [Project](https://genesis-embodied-ai.github.io/) |
| Isaac Lab / Isaac Sim | NVIDIA robotics simulation | [Project](https://developer.nvidia.com/isaac-sim) |
| MuJoCo Playground | Browser-based MuJoCo | [Paper](https://arxiv.org/abs/2502.08844) · [Project](https://playground.mujoco.org/) |
| OmniGibson | High-fidelity home simulation | [Paper](https://arxiv.org/abs/2311.01014) · [Project](https://behavior.stanford.edu/omnigibson/) |
| Habitat 2.0 | Navigation & rearrangement | [Paper](https://arxiv.org/abs/2106.14405) · [Project](https://aihabitat.org/) |
| BEHAVIOR-1K | 1,000 everyday activities | [Paper](https://arxiv.org/abs/2403.09227) · [Project](https://behavior.stanford.edu/) |
| iGibson | Interactive environments | [Paper](https://arxiv.org/abs/2012.02924) · [Project](https://svl.stanford.edu/igibson/) |
| RoboSuite | Modular manipulation | [Paper](https://arxiv.org/abs/2009.12293) · [Project](https://robosuite.ai/) |
| PyBullet | Lightweight physics for RL | [Project](https://pybullet.org/) |
| DexGarmentLab | Garment manipulation | [Paper](https://arxiv.org/abs/2505.11032) · [Project](https://wayrise.github.io/DexGarmentLab/) |
| MuBlE | Task planning benchmark | [Paper](https://arxiv.org/abs/2503.02834) · [Code](https://github.com/michaal94/MuBlE) |
| LocoMuJoCo | Locomotion benchmark | [Docs](https://loco-mujoco.readthedocs.io/) · [Code](https://github.com/robfiras/loco-mujoco) |
| BEHAVIOR Robot Suite | Whole-body manipulation | [Paper](https://arxiv.org/abs/2503.05652) · [Project](https://behavior-robot-suite.github.io/) |

---

## Companies & Projects

### Companies

| Company | Focus | Key Products | Links |
|---------|-------|--------------|-------|
| Physical Intelligence (π) | General-purpose robot foundation models | π₀, π₀.5, π₀.6, FAST | [Web](https://www.pi.website/) · [Blog](https://www.pi.website/) |
| Google DeepMind | Robotics research | RT-1/2, Gemini Robotics, Genie, PaLM-E | [Web](https://deepmind.google/) · [Blog](https://deepmind.google/discover/blog/) |
| OpenAI | AI research | CLIP, GPT-4V, Sora | [Web](https://openai.com/) · [Blog](https://openai.com/blog/) |
| Meta AI (FAIR) | JEPA, embodied AI | I-JEPA, V-JEPA, R3M, DINOv2, SAM | [Web](https://ai.meta.com/) · [Blog](https://ai.meta.com/blog/) |
| World Labs | Spatial intelligence & world models | Marble, RTFM | [Web](https://www.worldlabs.ai/) |
| NVIDIA | Simulation & foundation models | GR00T, Isaac Sim, Cosmos | [Web](https://developer.nvidia.com/isaac) · [Blog](https://developer.nvidia.com/blog/) |
| Microsoft Research | Multimodal agents | Magma | [Web](https://www.microsoft.com/en-us/research/) · [Blog](https://www.microsoft.com/en-us/research/blog/) |
| Hugging Face | Open-source VLAs | LeRobot, SmolVLA | [Web](https://huggingface.co/) · [Blog](https://huggingface.co/blog) |
| ByteDance | Vision-language-action models | GR-1, GR-2, CogACT | [Web](https://www.bytedance.com/) |
| Shanghai AI Lab | Embodied AI research | LEO, InternVL | [Web](https://www.shlab.org.cn/) |
| Covariant | Industrial robotics AI | RFM-1 | [Web](https://covariant.ai/) · [Blog](https://covariant.ai/insights/) |
| Skild AI | General-purpose robot brain | Skild Brain | [Web](https://www.skild.ai/) |
| RLWRLD | Industrial robotics foundation models | RFM | [Web](https://www.rlwrld.ai/) · [Blog](https://www.rlwrld.ai/blog) |
| Intrinsic (Alphabet) | Industrial robotics software | Flowstate | [Web](https://intrinsic.ai/) · [Blog](https://intrinsic.ai/blog) |
| Wayve | Embodied AI for driving | GAIA-1, LINGO | [Web](https://wayve.ai/) · [Blog](https://wayve.ai/thinking/) |
| Cortex AI | Real-world data for embodied AI | Egocentric + robot datasets | [Web](https://cortexrobot.ai/) |
| Verne Robotics | Mobile manipulation | Nemo | [Web](https://www.vernerobotics.com/) |
| Figure AI | Humanoid robots | Figure 01, Figure 02 | [Web](https://www.figure.ai/) |
| 1X Technologies | Humanoid robots | NEO, EVE | [Web](https://www.1x.tech/) · [Blog](https://www.1x.tech/discover) |
| Boston Dynamics | Advanced robotics | Atlas, Spot, Stretch | [Web](https://bostondynamics.com/) · [Blog](https://bostondynamics.com/blog/) |
| Tesla | Humanoid robots | Optimus | [Web](https://www.tesla.com/optimus) |
| Agility Robotics | Bipedal robots | Digit | [Web](https://agilityrobotics.com/) · [Blog](https://agilityrobotics.com/blog) |
| Unitree | Quadruped & humanoid robots | H1, G1, Go2 | [Web](https://www.unitree.com/) |
| Sanctuary AI | Humanoid robots | Phoenix | [Web](https://sanctuary.ai/) · [Blog](https://sanctuary.ai/resources/) |
| Apptronik | Humanoid robots | Apollo | [Web](https://apptronik.com/) |
| Fourier Intelligence | Humanoid & rehab robots | GR-1, GR-2 | [Web](https://www.fftai.com/) |
| Hello Robot | Mobile manipulation | Stretch | [Web](https://hello-robot.com/) · [Blog](https://hello-robot.com/blog) |
| Franka Robotics | Research robot arms | Panda | [Web](https://franka.de/) |
| Universal Robots | Collaborative robot arms | UR3, UR5, UR10 | [Web](https://www.universal-robots.com/) · [Blog](https://www.universal-robots.com/blog/) |
| UFACTORY | Affordable robot arms | xArm | [Web](https://www.ufactory.cc/) |
| Trossen Robotics | Research platforms | ViperX, WidowX, ALOHA | [Web](https://www.trossenrobotics.com/) |
| Flexiv | Adaptive robot arms | Rizon | [Web](https://www.flexiv.com/) |

### Research Labs & Initiatives

| Organization | Notable Contributions | Links |
|--------------|----------------------|-------|
| Stanford IRIS Lab | Diffusion Policy, MimicPlay | [Web](https://irislab.stanford.edu/) |
| Stanford SVL | BEHAVIOR, OmniGibson, VoxPoser | [Web](https://svl.stanford.edu/) |
| Stanford ILIAD | ACT, ALOHA, Mobile ALOHA | [Web](https://iliad.stanford.edu/) |
| Berkeley RAIL | Octo, BridgeData, R3M | [Web](https://rail.eecs.berkeley.edu/) |
| Berkeley BAIR | RT-X contributions, RoboAgent | [Web](https://bair.berkeley.edu/) · [Blog](https://bair.berkeley.edu/blog/) |
| CMU Robotics Institute | HomeRobot, OK-Robot | [Web](https://www.ri.cmu.edu/) |
| MIT CSAIL | LLM planning, manipulation | [Web](https://www.csail.mit.edu/) |
| NYU CILVR | OPEN TEACH, DynaMo, World Models | [Web](https://wp.nyu.edu/cilvr/) |
| Princeton REAL Lab | Manipulation research | [Web](https://robo-princeton.github.io/) |
| Columbia Robotics | Diffusion Policy, CLIPort | [Web](https://robotics.columbia.edu/) |
| Georgia Tech RIPL | LLM-Robotics survey | [Web](https://sites.gatech.edu/ripl/) |
| UW RSE Lab | CLIPort, VLAs | [Web](https://rse-lab.cs.washington.edu/) |
| Toyota Research Institute | Prismatic VLMs, OpenVLA | [Web](https://www.tri.global/) · [Blog](https://www.tri.global/news) |
| Tsinghua MARS Lab | LEO, CogACT | [Web](https://group.iiis.tsinghua.edu.cn/~marslab/) |
| Peking University | NaVid, various VLAs | [Web](https://www.pku.edu.cn/) |
| Open X-Embodiment | OXE dataset, RT-X | [Web](https://robotics-transformer-x.github.io/) |
| DROID Collaboration | DROID dataset | [Web](https://droid-dataset.github.io/) |

---

## Related Works

> Other awesome lists and resources for robotics and embodied AI.

- **Awesome World Models**: [[GitHub](https://github.com/knightnemo/Awesome-World-Models)]
- **Awesome-VLA-Robotics**: [[GitHub](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics)]
- **Awesome-VLA-RL**: [[GitHub](https://github.com/XiaoWei-i/Awesome-VLA-RL)]
- **Awesome-Robotics-Foundation-Models**: [[GitHub](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
- **Awesome-Generalist-Agents**: [[GitHub](https://github.com/cheryyunl/awesome-generalist-agents)]
- **Awesome-LLM-Robotics**: [[GitHub](https://github.com/GT-RIPL/Awesome-LLM-Robotics)]
- **Awesome World Models for Robotics**: [[GitHub](https://github.com/leofan90/Awesome-World-Models)]
- **Awesome-VLA-Post-Training**: [[GitHub](https://github.com/AoqunJin/Awesome-VLA-Post-Training)]
- **Awesome-BFM-Papers**: [[GitHub](https://github.com/yuanmingqi/awesome-bfm-papers)]
- **Awesome Embodied VLA/VA/VLN**: [[GitHub](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **Awesome VLA Study**: [[GitHub](https://github.com/MilkClouds/awesome-vla-study)]

---

## Citation

If you find this repository useful, please consider citing this list:

```bibtex
@misc{awesome-physical-ai,
    title = {Awesome Physical AI},
    author = {Keon Kim},
    journal = {GitHub repository},
    url = {https://github.com/keon/awesome-physical-ai},
    year = {2026},
}
```

---

## Contributing

We welcome contributions! Please submit a pull request to add relevant papers, correct errors, or improve organization.

### Guidelines

- Focus on **Physical AI** papers (robotics, embodied agents, world models, VLAs)
- Each paper should appear in only one category
- Include proper citations with links to papers, projects, and code
- Verify all links are working
