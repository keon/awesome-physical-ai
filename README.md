# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of academic papers and resources on **Physical AI** — focusing on Vision-Language-Action (VLA) models, embodied intelligence, and robotic foundation models.

> **Physical AI** refers to AI systems that interact with and manipulate the physical world through robotic embodiments, combining perception, reasoning, and action in real-world environments.

---

## Table of Contents

- [Surveys](#surveys)
- [Part I: Foundations](#part-i-foundations)
  - [Vision-Language Backbones](#vision-language-backbones)
  - [Visual Representations](#visual-representations)
- [Part II: VLA Architectures](#part-ii-vla-architectures)
  - [End-to-End VLAs](#end-to-end-vlas)
  - [Modular VLAs](#modular-vlas)
  - [Compact & Efficient VLAs](#compact--efficient-vlas)
- [Part III: Action Representation](#part-iii-action-representation)
  - [Discrete Tokenization](#discrete-tokenization)
  - [Continuous & Diffusion Policies](#continuous--diffusion-policies)
- [Part IV: World Models](#part-iv-world-models)
  - [JEPA & Latent Prediction](#jepa--latent-prediction)
  - [Generative World Models](#generative-world-models)
  - [Embodied World Models](#embodied-world-models)
- [Part V: Reasoning & Planning](#part-v-reasoning--planning)
  - [Chain-of-Thought & Deliberation](#chain-of-thought--deliberation)
  - [Error Detection & Recovery](#error-detection--recovery)
- [Part VI: Learning Paradigms](#part-vi-learning-paradigms)
  - [Imitation Learning](#imitation-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Reward Design](#reward-design)
- [Part VII: Scaling & Generalization](#part-vii-scaling--generalization)
  - [Scaling Laws](#scaling-laws)
  - [Cross-Embodiment Transfer](#cross-embodiment-transfer)
  - [Open-Vocabulary Generalization](#open-vocabulary-generalization)
- [Part VIII: Deployment](#part-viii-deployment)
  - [Quantization & Compression](#quantization--compression)
  - [Real-Time Control](#real-time-control)
- [Part IX: Safety & Alignment](#part-ix-safety--alignment)
- [Part X: Lifelong Learning](#part-x-lifelong-learning)
- [Part XI: Applications](#part-xi-applications)
  - [Humanoid Robots](#humanoid-robots)
  - [Manipulation](#manipulation)
  - [Navigation](#navigation)
- [Part XII: Resources](#part-xii-resources)
  - [Datasets & Benchmarks](#datasets--benchmarks)
  - [Simulation Platforms](#simulation-platforms)

---

## Surveys

- "Foundation Models in Robotics: Applications, Challenges, and the Future", *IJRR 2024*. [[Paper](https://arxiv.org/abs/2312.07843)] [[GitHub](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
  - Comprehensive survey covering 200+ papers on foundation models in robotics.

- "Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.08782)]
  - Meta-analysis synthesizing trends across perception, planning, and control.

- "Robot Learning in the Era of Foundation Models: A Survey", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.14379)]
  - Covers the paradigm shift from task-specific to foundation model-based robot learning.

- "Language-conditioned Learning for Robotic Manipulation: A Survey", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.10807)]
  - Focuses on how natural language enables more generalizable manipulation policies.

- "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.02816)]
  - Up-to-date survey covering the latest VLA architectures and deployment challenges.

- "Understanding World or Predicting Future? A Comprehensive Review of World Models", *arXiv, 2024*. [[Paper](https://arxiv.org/abs/2406.XXXXX)]
  - Systematic review categorizing differing definitions of world models across research groups.

- "The Development of LLMs for Embodied Navigation", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.00530)]
  - Tracks the evolution of LLM-based navigation from prompting to end-to-end VLAs.

---

## Part I: Foundations

### Vision-Language Backbones

> Core vision-language models that serve as pretrained backbones for Physical AI systems.

- **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision", *ICML 2021*. [[Paper](https://arxiv.org/abs/2103.00020)] [[Code](https://github.com/openai/CLIP)]
  - Foundational model aligning vision and language that underlies most VLA perception systems.

- **SigLIP**: "Sigmoid Loss for Language Image Pre-Training", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.15343)]
  - Improved CLIP training with sigmoid loss, used in many recent VLA visual encoders.

- **PaLI-X**: "PaLI-X: On Scaling up a Multilingual Vision and Language Model", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2305.18565)]
  - Serves as the vision-language backbone for RT-2, providing strong visual grounding.

- **LLaVA**: "Visual Instruction Tuning", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2304.08485)] [[Project](https://llava-vl.github.io/)]
  - Open-source VLM that serves as backbone for several open-source VLA models.

- **Prismatic VLMs**: "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.07865)] [[Code](https://github.com/TRI-ML/prismatic-vlms)]
  - Systematic study of VLM design choices informing OpenVLA and other robotics VLMs.

### Visual Representations

> Self-supervised visual encoders and perception models used in robotics.

- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision", *arXiv, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.07193)] [[Code](https://github.com/facebookresearch/dinov2)]
  - Self-supervised visual features highly effective for robotic manipulation and navigation.

- **SAM**: "Segment Anything", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2304.02643)] [[Project](https://segment-anything.com/)]
  - Universal segmentation model used for object extraction in robotic perception pipelines.

- **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2203.12601)] [[Code](https://github.com/facebookresearch/r3m)]
  - Pretrained visual encoder on Ego4D human video that transfers to robot manipulation.

- **MVP**: "Masked Visual Pre-training for Motor Control", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.06173)] [[Project](https://tetexiao.com/projects/mvp)]
  - Shows masked autoencoder pretraining creates visual representations effective for control.

- **Grounding DINO**: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", *ECCV 2024*. [[Paper](https://arxiv.org/abs/2303.05499)] [[Code](https://github.com/IDEA-Research/GroundingDINO)]
  - Open-vocabulary object detector used as perception backbone in many VLA systems.

---

## Part II: VLA Architectures

### End-to-End VLAs

> Monolithic models that treat vision, language, and actions as unified tokens in a single architecture.

- **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *RSS 2023*. [[Paper](https://arxiv.org/abs/2212.06817)] [[Project](https://robotics-transformer1.github.io/)] [[Code](https://github.com/google-research/robotics_transformer)]
  - Pioneer proving large-scale multi-task demonstration data could train a single transformer for diverse manipulations.

- **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Project](https://robotics-transformer2.github.io/)]
  - Established the VLA paradigm by co-fine-tuning VLMs on robotic data, enabling semantic generalization.

- **OpenVLA**: "OpenVLA: An Open-Source Vision-Language-Action Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.09246)] [[Project](https://openvla.github.io/)] [[Code](https://github.com/openvla/openvla)]
  - Open-source 7B model that outperformed the 55B RT-2-X, democratizing VLA research.

- **PaLM-E**: "PaLM-E: An Embodied Multimodal Language Model", *ICML 2023*. [[Paper](https://arxiv.org/abs/2303.03378)] [[Project](https://palm-e.github.io/)]
  - 562B parameter model demonstrating emergent multi-modal chain-of-thought reasoning for robotics.

- **VIMA**: "VIMA: General Robot Manipulation with Multimodal Prompts", *ICML 2023*. [[Paper](https://arxiv.org/abs/2210.03094)] [[Project](https://vima.cs.stanford.edu/)] [[Code](https://github.com/vimalabs/VIMA)]
  - Introduced multimodal prompting (text + images) for specifying manipulation tasks.

- **LEO**: "An Embodied Generalist Agent in 3D World", *ICML 2024*. [[Paper](https://arxiv.org/abs/2311.12871)] [[Project](https://embodied-generalist.github.io/)]
  - Unifies 3D perception, language understanding, and embodied action in a single model.

- **3D-VLA**: "3D-VLA: A 3D Vision-Language-Action Generative World Model", *ICML 2024*. [[Paper](https://arxiv.org/abs/2403.09631)] [[Project](https://vis-www.cs.umass.edu/3dvla)]
  - Integrates 3D scene understanding with VLA for better spatial reasoning.

- **Gato**: "A Generalist Agent", *TMLR 2022*. [[Paper](https://arxiv.org/abs/2205.06175)] [[Blog](https://deepmind.google/discover/blog/a-generalist-agent/)]
  - Single transformer handling 604 distinct tasks across games, chat, and robotics.

- **RoboFlamingo**: "Vision-Language Foundation Models as Effective Robot Imitators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2311.01378)] [[Project](https://roboflamingo.github.io/)]
  - Adapts OpenFlamingo VLM with a policy head to capture sequential history.

- **Magma**: "Magma: A Foundation Model for Multimodal AI Agents", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.13130)] [[Code](https://github.com/microsoft/Magma)]
  - Microsoft's unified foundation model bridging virtual (GUI) and physical (robot) agent control.

- **RoboVLMs**: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.14058)] [[Project](https://robovlms.github.io/)]
  - Systematic study identifying key design choices for building effective VLAs.

### Modular VLAs

> Models that decouple cognition (VLM-based planning) from action (specialized motor modules).

- **CogACT**: "CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.19650)] [[Project](https://cogact.github.io/)]
  - Decouples high-level cognition (VLM backbone) from low-level action (Diffusion Action Transformer).

- **Gemini Robotics**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)] [[Blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)]
  - Introduces "Thinking Before Acting" with internal natural language reasoning for complex tasks.

- **Helix**: "Helix: A Vision-Language-Action Model for Generalist Humanoid Control", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]
  - Specialized VLA tailored for full upper-body control of humanoid robots.

- **SayCan**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2204.01691)] [[Project](https://say-can.github.io/)]
  - First to combine LLM semantic knowledge with learned affordance functions.

- **Code as Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sep 2022*. [[Paper](https://arxiv.org/abs/2209.07753)] [[Project](https://code-as-policies.github.io/)]
  - Seminal work showing LLMs can generate executable robot control code directly.

- **SayPlan**: "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Task Planning", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.06135)] [[Project](https://sayplan.github.io/)]
  - Grounds LLMs using 3D scene graphs for scalable, long-horizon task planning.

- **Inner Monologue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Project](https://innermonologue.github.io/)]
  - Pioneered closed-loop language feedback where robots verbalize observations for planning.

- **Instruct2Act**: "Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.11176)] [[Code](https://github.com/OpenGVLab/Instruct2Act)]
  - Uses LLMs to compose foundation model APIs into executable robot programs.

- **TidyBot**: "TidyBot: Personalized Robot Assistance with Large Language Models", *IROS 2023*. [[Paper](https://arxiv.org/abs/2305.05658)] [[Project](https://tidybot.cs.princeton.edu/)]
  - Demonstrates LLM-based personalization for household tasks from few examples.

### Compact & Efficient VLAs

> Lightweight VLA models optimized for fast inference and edge deployment.

- **TinyVLA**: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2409.12514)] [[Project](https://tiny-vla.github.io/)]
  - Compact 1.3B model designed for fast inference and efficient training.

- **SmolVLA**: "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", *arXiv, Jun 2025*. [[Paper](https://huggingface.co/blog/smolvla)] [[Code](https://github.com/huggingface/lerobot)]
  - 450M parameters achieving comparable performance to 10x larger models on consumer GPUs.

- **OpenVLA-OFT**: "OpenVLA-OFT: Efficient Fine-Tuning for Open Vision-Language-Action Models", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Introduces orthogonal fine-tuning to adapt OpenVLA to new tasks with minimal forgetting.

- **RT-H**: "RT-H: Action Hierarchies Using Language", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.01823)] [[Project](https://rt-hierarchy.github.io/)]
  - Uses language-conditioned action hierarchies to reduce action space and speed inference.

- **LAPA**: "Latent Action Pretraining from Videos", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.11758)] [[Project](https://latentactionpretraining.github.io/)]
  - Pretrains action representations from unlabeled video, reducing dependency on robot demos.

---

## Part III: Action Representation

### Discrete Tokenization

> Models that convert continuous joint movements into discrete "action tokens" similar to words.

- **FAST**: "FAST: Efficient Action Tokenization for Vision-Language-Action Models", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.09747)] [[Project](https://www.pi.website/research/fast)]
  - Uses frequency-space (DCT) tokenization to compress action sequences 7x while preserving control.

- **GR-1**: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2312.13139)] [[Project](https://gr1-manipulation.github.io/)]
  - Leverages video prediction pretraining to learn action-conditioned visual dynamics.

- **GR-2**: "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.06158)] [[Project](https://gr2-manipulation.github.io/)]
  - Scales video pretraining to web-scale data for improved manipulation generalization.

- **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", *RSS 2023*. [[Paper](https://arxiv.org/abs/2304.13705)] [[Project](https://tonyzhaozh.github.io/aloha/)] [[Code](https://github.com/tonyzhaozh/act)]
  - Introduced Action Chunking with Transformers for smooth bimanual manipulation.

- **Behavior Transformers**: "Behavior Transformers: Cloning k Modes with One Stone", *NeurIPS 2022*. [[Paper](https://arxiv.org/abs/2206.11251)] [[Code](https://github.com/notmahi/bet)]
  - Handles multimodal action distributions by clustering actions into discrete modes.

### Continuous & Diffusion Policies

> Models that use diffusion or flow matching to generate continuous trajectories.

- **π₀ (pi-zero)**: "π₀: A Vision-Language-Action Flow Model for General Robot Control", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.24164)] [[Project](https://www.physicalintelligence.company/blog/pi0)]
  - Uses flow matching to generate high-frequency (50 Hz) continuous actions for dexterous tasks.

- **π₀.5**: "π₀.5: Scaling Robot Foundation Models", *arXiv, Apr 2025*. [[Paper](https://www.physicalintelligence.company/blog/pi0-5)]
  - Updated flow model designed for open-world generalization with improved zero-shot transfer.

- **Octo**: "Octo: An Open-Source Generalist Robot Policy", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.12213)] [[Project](https://octo-models.github.io/)] [[Code](https://github.com/octo-models/octo)]
  - Lightweight open-source policy using diffusion heads for smoother trajectory generation.

- **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", *RSS 2023*. [[Paper](https://arxiv.org/abs/2303.04137)] [[Project](https://diffusion-policy.cs.columbia.edu/)] [[Code](https://github.com/real-stanford/diffusion_policy)]
  - Foundational work showing diffusion models excel at capturing multimodal action distributions.

- **RDT-1B**: "RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.07864)] [[Project](https://rdt-robotics.github.io/rdt-robotics/)]
  - 1B-parameter diffusion model specifically designed for coordinated bimanual manipulation.

- **DexVLA**: "DexVLA: Vision-Language Model with Plug-In Diffusion Expert", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.05855)] [[Project](https://dex-vla.github.io/)]
  - Combines VLM reasoning with a pluggable diffusion expert for dexterous tasks.

- **Diffusion-VLA**: "Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.03293)] [[Project](https://diffusion-vla.github.io/)]
  - Bridges reasoning and action via a Reasoning Injection Module (FiLM).

- **3D Diffusion Policy**: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via 3D Representations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2403.03954)] [[Project](https://3d-diffusion-policy.github.io/)]
  - Integrates sparse 3D point cloud representations with diffusion for spatial generalization.

- **Moto**: "Moto: Latent Motion Token as the Bridging Language for Robot Manipulation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04445)] [[Project](https://chenyi99.github.io/moto/)]
  - Uses latent motion tokens bridging high-level plans and low-level control.

- **Consistency Policy**: "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.07503)] [[Project](https://consistency-policy.github.io/)]
  - Distills diffusion policies into single-step models for 10x faster inference.

---

## Part IV: World Models

### JEPA & Latent Prediction

> Yann LeCun's Joint-Embedding Predictive Architecture (JEPA) predicts future latent states rather than pixels.

- "A Path Towards Autonomous Machine Intelligence", *Meta AI, Jun 2022*. [[Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)]
  - LeCun's foundational vision describing the "world model in the middle" cognitive architecture.

- **I-JEPA**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", *CVPR 2023*. [[Paper](https://arxiv.org/abs/2301.08243)] [[Code](https://github.com/facebookresearch/ijepa)]
  - First practical JEPA implementation, predicting latent patches without pixel reconstruction.

- **V-JEPA**: "Video Joint Embedding Predictive Architecture", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.03014)] [[Code](https://github.com/facebookresearch/jepa)]
  - Extends I-JEPA to video, learning spatiotemporal representations for physical interactions.

- **MC-JEPA**: "MC-JEPA: Self-Supervised Learning of Motion and Content Features", *CVPR 2023*. [[Paper](https://arxiv.org/abs/2307.12698)]
  - Disentangles motion and content representations in video for physical dynamics understanding.

- **LeJEPA**: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", *arXiv, Nov 2025*. [[Paper](https://arxiv.org/abs/2511.XXXXX)]
  - Formalized theory with SIGReg as optimal training objective for latent world models.

- **VL-JEPA**: "VL-JEPA: Vision-Language Joint Embedding Predictive Architecture", *arXiv, Dec 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]
  - Non-generative alternative to multimodal LLMs via latent embedding prediction.

- "Value-guided Action Planning with JEPA World Models", *arXiv, Jan 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Shapes JEPA spaces so state distances approximate value functions for robotic planning.

### Generative World Models

> World models that generate pixels, video, or interactive environments.

- **World Models**: "World Models", *NeurIPS 2018*. [[Paper](https://arxiv.org/abs/1803.10122)] [[Project](https://worldmodels.github.io/)]
  - Seminal Ha & Schmidhuber work popularizing world models for RL with VAE + RNN.

- **DreamerV3**: "Mastering Diverse Domains through World Models", *arXiv, Jan 2023*. [[Paper](https://arxiv.org/abs/2301.04104)] [[Project](https://danijar.com/project/dreamerv3/)]
  - State-of-the-art world model RL agent mastering 150+ tasks across games and robotics.

- **Genie**: "Genie: Generative Interactive Environments", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.15391)] [[Project](https://sites.google.com/view/genie-2024)]
  - Learns interactive world models from unlabeled videos, generating playable 2D environments.

- **Genie 2**: "Genie 2: A Large-Scale Foundation World Model", *DeepMind, Dec 2024*. [[Blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)]
  - Generates diverse, playable 3D worlds from single images for training embodied agents.

- **Sora**: "Video Generation Models as World Simulators", *OpenAI, Feb 2024*. [[Blog](https://openai.com/research/video-generation-models-as-world-simulators)]
  - Argues that scaling video generation creates emergent world simulation capabilities.

- **GAIA-1**: "GAIA-1: A Generative World Model for Autonomous Driving", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.17080)]
  - 9B parameter world model generating realistic driving scenarios for simulation.

- **GameNGen**: "Diffusion Models Are Real-Time Game Engines", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.14837)]
  - Runs DOOM entirely on a neural network as a real-time interactive simulator.

- **DIAMOND**: "Diffusion for World Modeling: Visual Details Matter in Atari", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2405.12399)] [[Code](https://github.com/eloialonso/diamond)]
  - Diffusion-based world model achieving SOTA on Atari by preserving visual details.

- **3D Gaussian Splatting**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering", *SIGGRAPH 2023*. [[Paper](https://arxiv.org/abs/2308.04079)] [[Project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]
  - Foundational technique enabling fast novel view synthesis for spatial world models.

- "From Words to Worlds: Spatial Intelligence is AI's Next Frontier", *World Labs, 2025*. [[Blog](https://www.worldlabs.ai/)]
  - Fei-Fei Li's manifesto defining world models as generative, multimodal, actionable systems.

- **Marble**: "Marble: A Multimodal World Model", *World Labs, Nov 2025*. [[Project](https://www.worldlabs.ai/)]
  - Generates persistent, interactive 3D environments from text or 2D images.

- **RTFM**: "RTFM: A Real-Time Frame Model", *World Labs, Oct 2025*. [[Project](https://www.worldlabs.ai/)]
  - Generative model producing video in real-time as users interact with environments.

### Embodied World Models

> World models designed for robotic manipulation, navigation, and physical reasoning.

- **Structured World Models**: "Structured World Models from Human Videos", *RSS 2023*. [[Paper](https://arxiv.org/abs/2308.10901)] [[Project](https://human-world-model.github.io/)]
  - Learns structured world models from human activity videos for robot transfer.

- **WHALE**: "WHALE: Towards Generalizable and Scalable World Models for Embodied Decision-making", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.05619)]
  - Scalable world model architecture combining language and visual observations.

- "A Controllable Generative World Model for Robot Manipulation", *arXiv, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.XXXXX)]
  - Action-conditioned world models with multi-view prediction and temporal coherence.

- **Code World Model**: "Code World Model: Learning to Execute Code in World Simulation", *Meta AI, Oct 2025*. [[Paper](https://arxiv.org/abs/2510.XXXXX)]
  - 32B model treating code as a dynamic system, simulating program execution states.

- **PhyGDPO**: "PhyGDPO: Physics-Aware Text-to-Video Generation via Direct Preference Optimization", *Meta AI, Jan 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Uses DPO to ensure video generation remains physically consistent.

- "The Essential Role of Causality in Foundation World Models for Embodied AI", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.06665)]
  - Argues causal reasoning is essential for world models supporting robust physical AI.

- **MineDreamer**: "MineDreamer: Learning to Follow Instructions via Chain-of-Imagination", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.12037)] [[Project](https://sites.google.com/view/minedreamer)]
  - Uses world model imagination to generate subgoal images for instruction following.

- **Video Language Planning**: "Video Language Planning", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.10625)] [[Project](https://video-language-planning.github.io/)]
  - Combines video prediction with tree search for long-horizon robotic planning.

- "Learning Universal Policies via Text-Guided Video Generation", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.00111)] [[Project](https://universal-policy.github.io/unipi/)]
  - Uses video generation as planning, treating video models as universal policy representations.

- **SIMA**: "Scaling Instructable Agents Across Many Simulated Worlds", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2404.10179)] [[Blog](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)]
  - Generalist agent following language instructions across diverse 3D virtual worlds.

- **UniSim**: "UniSim: Learning Interactive Real-World Simulators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.06114)] [[Project](https://universal-simulator.github.io/unisim/)]
  - Learns universal simulators from video that can simulate any scene with realistic physics.

---

## Part V: Reasoning & Planning

### Chain-of-Thought & Deliberation

> Models implementing "thinking before acting" with explicit reasoning or value-guided search.

- **Hume**: "Hume: Introducing Deliberative Alignment in Embodied AI", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Uses a value-query head to evaluate candidate actions via repeat sampling, enabling failure recovery.

- **Embodied-CoT**: "Robotic Control via Embodied Chain-of-Thought Reasoning", *arXiv, Jul 2024*. [[Paper](https://arxiv.org/abs/2407.08693)] [[Project](https://embodied-cot.github.io/)]
  - Generates explicit reasoning chains before action prediction for interpretability.

- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR 2023*. [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]
  - Interleaves reasoning traces with actions, establishing "thinking before acting" foundation.

- **ReKep**: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.01652)] [[Project](https://rekep-robot.github.io/)]
  - Uses VLMs to generate relational keypoint constraints guiding motion planning.

- **TraceVLA**: "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.10345)] [[Project](https://tracevla.github.io/)]
  - Overlays visual traces on images to enhance spatial-temporal reasoning.

- **LLM-State**: "LLM-State: Open World State Representation for Long-horizon Task Planning", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.17406)]
  - Maintains explicit world state representations that LLMs query for long-horizon planning.

- **Statler**: "Statler: State-Maintaining Language Models for Embodied Reasoning", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.17840)] [[Project](https://statler-lm.github.io/)]
  - Explicitly maintains world state in natural language for multi-step reasoning.

- **RoboReflect**: "RoboReflect: Reflective Reasoning for Robot Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Enables robots to adjust strategies based on unsuccessful attempts through self-reflection.

### Error Detection & Recovery

> Methods for detecting failures and correcting robot actions in real-time.

- **DoReMi**: "Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", *arXiv, Jul 2023*. [[Paper](https://arxiv.org/abs/2307.00329)] [[Project](https://sites.google.com/view/doremi-paper)]
  - Detects when execution deviates from expectations and triggers replanning automatically.

- **CoPAL**: "Corrective Planning of Robot Actions with Large Language Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.07263)] [[Project](https://sites.google.com/view/copal-robot)]
  - Uses LLM feedback to correct plans online when environmental changes occur.

- **Code-as-Monitor**: "Code-as-Monitor: Constraint-aware Visual Programming for Failure Detection", *CVPR 2025*. [[Paper](https://arxiv.org/abs/2412.04455)] [[Project](https://code-as-monitor.github.io/)]
  - Generates constraint-checking code that monitors execution proactively.

- **AHA**: "AHA: A Vision-Language-Model for Detecting and Reasoning over Failures", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.XXXXX)]
  - VLM specialized for failure detection and root cause analysis in manipulation.

- **PRED**: "Pre-emptive Action Revision by Environmental Feedback", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]
  - Preemptively revises actions before failure based on early feedback signals.

---

## Part VI: Learning Paradigms

### Imitation Learning

> Behavioral cloning and learning from demonstrations.

- **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Project](https://cliport.github.io/)] [[Code](https://github.com/cliport/cliport)]
  - Combines CLIP semantic features with transporter networks for language-conditioned manipulation.

- **Play-LMP**: "Learning Latent Plans from Play", *CoRL 2019*. [[Paper](https://arxiv.org/abs/1903.01973)] [[Project](https://learning-from-play.github.io/)]
  - Learns reusable skills from unstructured "play" data without task labels.

- **MimicPlay**: "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2302.12422)] [[Project](https://mimic-play.github.io/)]
  - Learns long-horizon manipulation from human video without teleoperation.

- **RVT**: "RVT: Robotic View Transformer for 3D Object Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.14896)] [[Project](https://robotic-view-transformer.github.io/)] [[Code](https://github.com/NVlabs/RVT)]
  - Multi-view transformer achieving SOTA manipulation with efficient virtual view rendering.

- **RVT-2**: "RVT-2: Learning Precise Manipulation from Few Demonstrations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2406.08545)] [[Project](https://robotic-view-transformer-2.github.io/)]
  - Learning precise manipulation from just 5 demonstrations.

- **DIAL**: "Robotic Skill Acquisition via Instruction Augmentation", *arXiv, Nov 2022*. [[Paper](https://arxiv.org/abs/2211.11736)] [[Project](https://instructionaugmentation.github.io/)]
  - Uses VLMs to relabel demonstrations with diverse instructions.

- **Perceiver-Actor**: "A Multi-Task Transformer for Robotic Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2209.05451)] [[Project](https://peract.github.io/)] [[Code](https://github.com/peract/peract)]
  - Efficient attention over 3D voxel grids for real-time multi-task manipulation.

- **BOSS**: "Bootstrap Your Own Skills: Learning to Solve New Tasks with LLM Guidance", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2310.10021)] [[Project](https://clvrai.github.io/boss/)]
  - Uses LLM to decompose tasks and bootstrap skill learning from minimal demos.

### Reinforcement Learning

> RL-based methods for optimizing VLA policies.

- **CO-RFT**: "CO-RFT: Chunked Offline Reinforcement Learning Fine-Tuning for VLAs", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Two-stage offline RL achieving 57% improvement over imitation learning with 30-60 samples.

- **HICRA**: "HICRA: Hierarchy-Aware Credit Assignment for Reinforcement Learning in VLAs", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Focuses optimization on "planning tokens" rather than execution tokens.

- **FLaRe**: "FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale RL Fine-Tuning", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.16578)] [[Project](https://robot-flare.github.io/)]
  - Large-scale RL fine-tuning producing adaptive policies that recover from disturbances.

- **Plan-Seq-Learn**: "Plan-Seq-Learn: Language Model Guided RL for Long Horizon Tasks", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2405.01534)] [[Project](https://mihdalal.github.io/planseqlearn/)]
  - Uses LLM plans to guide RL exploration for 10+ sequential subtasks.

- **GLAM**: "Grounding Large Language Models in Interactive Environments with Online RL", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.02662)] [[Code](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)]
  - Grounds LLM knowledge through online RL interaction in embodied environments.

- **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models", *ICML 2023*. [[Paper](https://arxiv.org/abs/2302.06692)]
  - Uses LLM-generated exploration bonuses to accelerate RL pretraining.

- **RL4VLA**: "RL4VLA: What Can RL Bring to VLA Generalization?", *NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]
  - Comprehensive study on when and how RL fine-tuning improves VLA generalization.

- **TPO**: "TPO: Trajectory-wise Preference Optimization for VLAs", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Fine-tunes VLAs based on trajectory preferences rather than simple imitation.

- **ReinboT**: "ReinboT: Reinforcement Learning for Robotic Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Integrates RL returns maximization to enhance manipulation and few-shot learning.

### Reward Design

> Automated reward function generation using language models.

- **Text2Reward**: "Text2Reward: Automated Dense Reward Function Generation", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.11489)] [[Project](https://text-to-reward.github.io/)]
  - LLMs generate dense reward code from task descriptions.

- **Language to Rewards**: "Language to Rewards for Robotic Skill Synthesis", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.08647)] [[Project](https://language-to-reward.github.io/)]
  - Translates natural language task descriptions into reward functions.

- **ExploRLLM**: "ExploRLLM: Guiding Exploration in Reinforcement Learning with LLMs", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.09583)]
  - LLM proposes exploration objectives to guide RL toward meaningful regions.

---

## Part VII: Scaling & Generalization

### Scaling Laws

> Mathematical relationships between model/data scale and robotic performance.

- "Neural Scaling Laws for Embodied AI", *arXiv, May 2024*. [[Paper](https://arxiv.org/abs/2405.14005)]
  - Meta-analysis proving robotics performance follows power-law relationships with scale.

- "Data Scaling Laws in Imitation Learning for Robotic Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.18647)] [[Project](https://data-scaling-laws.github.io/)]
  - Quantifies how manipulation success scales logarithmically with dataset size.

- **AutoRT**: "AutoRT: Embodied Foundation Models for Large Scale Orchestration", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2401.12963)] [[Project](https://auto-rt.github.io/)]
  - Demonstrates autonomous fleet data collection enabling continuous scaling.

- **SARA-RT**: "SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.01990)]
  - Attention mechanisms that scale efficiently to larger model sizes.

- "Scaling Robot Learning with Semantically Imagined Experience", *RSS 2023*. [[Paper](https://arxiv.org/abs/2302.11550)]
  - Uses generative models to synthesize diverse training scenarios.

### Cross-Embodiment Transfer

> Single policies controlling diverse robot types (humanoids, quadrupeds, manipulators).

- **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]
  - Largest cross-embodiment study with 22 robot types proving diverse training helps all.

- **GENBOT-1K**: "Towards Embodiment Scaling Laws: Training on ~1000 Robot Bodies", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Demonstrates training on ~1,000 robot bodies enables zero-shot transfer to unseen robots.

- **Crossformer**: "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2408.11812)] [[Project](https://crossformer-model.github.io/)]
  - Single policy controlling manipulators, legged robots, and drones.

- **HPT**: "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20537)] [[Project](https://liruiw.github.io/hpt/)]
  - Scales to multiple embodiments by treating inputs as heterogeneous token streams.

- **MetaMorph**: "MetaMorph: Learning Universal Controllers with Transformers", *ICLR 2022*. [[Paper](https://arxiv.org/abs/2203.11931)] [[Project](https://metamorph-iclr.github.io/)]
  - Single transformer controlling robots with different numbers of limbs.

- **RUMs**: "Robot Utility Models: General Policies for Zero-Shot Deployment", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.05865)] [[Project](https://robotutilitymodels.com/)]
  - Utility-maximizing policies that transfer zero-shot to new robots and environments.

- **URMA**: "Unified Robot Morphology Architecture", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Handles varying observation and action spaces across arbitrary morphologies.

- **RoboAgent**: "RoboAgent: Generalization and Efficiency via Semantic Augmentations", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2309.01918)] [[Project](https://robopen.github.io/)]
  - Uses semantic data augmentation to improve generalization without more real data.

### Open-Vocabulary Generalization

> Models that generalize to novel visual appearances and semantic concepts.

- **MOO**: "Open-World Object Manipulation using Pre-trained Vision-Language Models", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2303.00905)] [[Project](https://robot-moo.github.io/)]
  - Manipulates novel objects by leveraging CLIP embeddings for zero-shot understanding.

- **VoxPoser**: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.05973)] [[Project](https://voxposer.github.io/)]
  - Generates 3D affordance and constraint maps from language for zero-shot manipulation.

- **RoboPoint**: "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.10721)] [[Project](https://robo-point.github.io/)]
  - Predicts spatial affordances directly from VLM reasoning.

- **CLIP-Fields**: "CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory", *RSS 2023*. [[Paper](https://arxiv.org/abs/2210.05663)] [[Project](https://mahis.life/clip-fields/)]
  - Creates queryable 3D semantic maps for language-based object retrieval.

- **VLMaps**: "Visual Language Maps for Robot Navigation", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2210.05714)] [[Project](https://vlmaps.github.io/)]
  - Builds spatial maps grounded in natural language for zero-shot navigation.

- **NLMap**: "Open-vocabulary Queryable Scene Representations", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2209.09874)] [[Project](https://nlmap-saycan.github.io/)]
  - Creates open-vocabulary queryable scene representations for SayCan-style planning.

- **LERF**: "LERF: Language Embedded Radiance Fields", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.09553)] [[Project](https://www.lerf.io/)]
  - Embeds CLIP features into NeRF for 3D language-grounded scene understanding.

- **Any-point Trajectory**: "Any-point Trajectory Modeling for Policy Learning", *RSS 2024*. [[Paper](https://arxiv.org/abs/2401.00025)] [[Project](https://xingyu-lin.github.io/atm/)]
  - Learns policies by predicting any-point trajectories, enabling camera viewpoint transfer.

---

## Part VIII: Deployment

### Quantization & Compression

> Low-bit weight quantization for efficient edge deployment.

- **BitVLA**: "BitVLA: 1-bit Vision-Language-Action Models for Robotics", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - First 1-bit (ternary) VLA reducing memory to 29.8% for edge deployment.

- **DeeR-VLA**: "DeeR-VLA: Dynamic Inference of Multimodal LLMs for Efficient Robot Execution", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.02359)] [[Code](https://github.com/yueyang130/DeeR-VLA)]
  - Dynamically adjusts model depth based on task complexity for energy efficiency.

- **QuaRT-VLA**: "Quantized Robotics Transformers for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - 4-bit quantization maintaining 95%+ performance on embedded systems.

- **PDVLA**: "PDVLA: Parallel Decoding for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Treats autoregressive decoding as parallel fixed-point iterations for high control frequencies.

### Real-Time Control

> Methods bridging high-latency AI inference and low-latency physical control.

- **A2C2**: "A2C2: Asynchronous Action Chunk Correction for Real-Time Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]
  - Lightweight head adjusting outdated action chunks based on latest observations.

- **RTC**: "Real-Time Chunking: Asynchronous Execution for Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Overlaps prediction and execution phases to reduce effective robot latency.

---

## Part IX: Safety & Alignment

> Ethical constraints, safety frameworks, and human-robot alignment.

- **Robot Constitution**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)]
  - Introduces data-driven "Robot Constitution" with natural language rules for safe behavior.

- **ASIMOV**: "ASIMOV: A Safety Benchmark for Embodied AI", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Benchmark evaluating and improving semantic safety in physical AI.

- **RoboPAIR**: "Jailbreaking LLM-Controlled Robots", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.13691)] [[Project](https://robopair.org/)]
  - Demonstrates adversarial attacks on LLM-controlled robots, highlighting vulnerabilities.

- **RoboGuard**: "Safety Guardrails for LLM-Enabled Robots", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]
  - Runtime safety monitors preventing LLM-generated actions from causing harm.

- "Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.10340)]
  - Systematic analysis of safety risks when deploying language models for robot control.

- "Robots Enact Malignant Stereotypes", *FAccT 2022*. [[Paper](https://arxiv.org/abs/2207.11569)] [[Project](https://sites.google.com/view/robots-enact-stereotypes)]
  - First study showing robots inherit harmful biases from vision-language pretraining.

- "LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.08824)]
  - Comprehensive risk assessment of LLM-controlled robots in real-world scenarios.

- "Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Provides formal safety guarantees through reachability-based verification.

---

## Part X: Lifelong Learning

> Agents that continuously learn and adapt without forgetting prior skills.

- **Voyager**: "VOYAGER: An Open-Ended Embodied Agent with Large Language Models", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.16291)] [[Project](https://voyager.minedojo.org/)] [[Code](https://github.com/MineDojo/Voyager)]
  - First LLM-powered agent in Minecraft autonomously building a skill library of executable code.

- **RoboGen**: "RoboGen: A Generative and Self-Guided Robotic Agent", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.01455)] [[Project](https://robogen-ai.github.io/)]
  - Endlessly proposes and masters new manipulation skills through self-guided curriculum.

- **RoboCat**: "RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.11706)] [[Blog](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)]
  - Self-improves through cycles of self-generated data collection and training.

- **LOTUS**: "LOTUS: Continual Imitation Learning via Unsupervised Skill Discovery", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2311.02058)] [[Project](https://ut-austin-rpl.github.io/Lotus/)]
  - Discovers and accumulates manipulation skills without forgetting.

- **DEPS**: "Describe, Explain, Plan and Select: Interactive Planning with LLMs for Open-World Agents", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.01560)] [[Code](https://github.com/CraftJarvis/MC-Planner)]
  - Interactive LLM planning with self-explanation for open-world task completion.

- **JARVIS-1**: "JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal LLMs", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.05997)] [[Project](https://craftjarvis-jarvis1.github.io/)]
  - Maintains persistent memory enabling knowledge accumulation over extended play.

- **MP5**: "MP5: A Multi-modal Open-ended Embodied System via Active Perception", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2312.07472)] [[Project](https://craftjarvis.github.io/MP5/)]
  - Uses active perception for multi-modal embodied reasoning.

- **SPRINT**: "SPRINT: Semantic Policy Pre-training via Language Instruction Relabeling", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.11886)] [[Project](https://clvrai.github.io/sprint/)]
  - Relabels demonstrations with diverse instructions for broad language understanding.

---

## Part XI: Applications

### Humanoid Robots

> Foundation models for humanoid robot control.

- **GR00T N1**: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.14734)] [[Project](https://developer.nvidia.com/isaac/gr00t)]
  - NVIDIA's dual-system architecture trained on robot trajectories, human videos, and synthetic data.

- **HumanPlus**: "HumanPlus: Humanoid Shadowing and Imitation from Humans", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.10454)] [[Project](https://humanoid-ai.github.io/)]
  - Real-time human-to-humanoid motion retargeting from demonstration videos.

- **ExBody**: "Expressive Whole-Body Control for Humanoid Robots", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.16796)] [[Project](https://expressive-humanoid.github.io/)]
  - Enables humanoids to perform expressive, human-like whole-body movements.

- **H2O**: "Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.04436)] [[Project](https://human2humanoid.com/)]
  - Real-time teleoperation mapping human motion to humanoid whole-body control.

- **OmniH2O**: "OmniH2O: Universal Human-to-Humanoid Teleoperation and Learning", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.08858)] [[Project](https://omni.human2humanoid.com/)]
  - Universal teleoperation framework for diverse humanoid platforms.

- "Learning Humanoid Locomotion with Transformers", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2303.03381)] [[Project](https://humanoid-locomotion.github.io/)]
  - Transformer-based locomotion achieving robust bipedal walking.

### Manipulation

> Robot manipulation with foundation models.

- **Scaling Up Distilling Down**: "Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.14535)] [[Project](https://www.cs.columbia.edu/~huy/scalingup/)]
  - Uses internet-scale knowledge to acquire manipulation skills with minimal demos.

- **LLM3**: "LLM3: Large Language Model-based Task and Motion Planning with Failure Reasoning", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11552)]
  - LLM reasons about motion planning failures to generate alternatives.

- **ManipVQA**: "ManipVQA: Injecting Robotic Affordance into Multi-Modal LLMs", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11289)]
  - Grounds MLLMs with physical affordances for manipulation questions.

- **UniAff**: "UniAff: A Unified Representation of Affordances for Tool Usage and Articulation", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.20551)]
  - Unified affordance representation for tool use and articulated objects.

- **SKT**: "SKT: State-Aware Keypoint Trajectories for Robotic Garment Manipulation", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.18082)]
  - Keypoint-based approach for deformable object manipulation.

- **Manipulate-Anything**: "Manipulate-Anything: Automating Real-World Robots using VLMs", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.18915)] [[Project](https://robot-ma.github.io/)]
  - VLM-based system enabling manipulation of novel objects with minimal specification.

- **A3VLM**: "A3VLM: Actionable Articulation-Aware Vision Language Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.07549)]
  - Specialized VLM understanding articulated objects for manipulation planning.

- **LaN-Grasp**: "Language-Driven Grasp Detection", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2311.09876)]
  - Generates grasp poses conditioned on natural language descriptions.

- **Grasp Anything**: "Pave the Way to Grasp Anything: Transferring Foundation Models", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.05716)]
  - Transfers foundation model knowledge for universal pick-and-place.

### Navigation

> Vision-language models for robot navigation.

- **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Project](https://sites.google.com/view/lmnav)]
  - Combines LLM, VLM, and VNM for instruction-following navigation.

- **NaVILA**: "NaVILA: Legged Robot Vision-Language-Action Model for Navigation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04453)] [[Project](https://navila-bot.github.io/)]
  - Extends VLA paradigm to quadruped navigation.

- **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2203.10421)]
  - Zero-shot object navigation using CLIP for target localization.

- **L3MVN**: "L3MVN: Leveraging Large Language Models for Visual Target Navigation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2304.05501)]
  - LLM provides commonsense reasoning for efficient exploration.

- **NaVid**: "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.15852)] [[Project](https://pku-epic.github.io/NaVid/)]
  - Video-based VLM planning navigation steps from egocentric video.

- **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2309.15940)] [[Project](https://ovsg-l.github.io/)]
  - Builds open-vocabulary 3D scene graphs for grounding language queries.

- **CANVAS**: "CANVAS: Commonsense-Aware Navigation System", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.01273)]
  - Incorporates commonsense reasoning for intuitive human-robot interaction.

- **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV 2020*. [[Paper](https://arxiv.org/abs/2004.14973)]
  - First to pretrain navigation agents on web image-text pairs.

- **ThinkBot**: "ThinkBot: Embodied Instruction Following with Thought Chain Reasoning", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.07062)]
  - Generates explicit thought chains for complex multi-step instruction following.

---

## Part XII: Resources

### Datasets & Benchmarks

- **Open X-Embodiment**: Largest open-source robot dataset with 1M+ trajectories from 22 robot embodiments. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]

- **DROID**: Large-scale in-the-wild robot manipulation dataset (76K trajectories, 564 scenes). [[Paper](https://arxiv.org/abs/2403.12945)] [[Project](https://droid-dataset.github.io/)]

- **BridgeData V2**: Multi-task dataset enabling few-shot transfer to new objects/environments. [[Paper](https://arxiv.org/abs/2308.12952)] [[Project](https://rail-berkeley.github.io/bridgedata/)]

- **ARIO**: Standardized format unifying diverse robot datasets for pretraining. [[Paper](https://arxiv.org/abs/2408.10899)] [[Project](https://imaei.github.io/project_pages/ario/)]

- **LIBERO**: Benchmark for lifelong robot learning with 130 tasks. [[Paper](https://arxiv.org/abs/2306.03310)] [[Project](https://libero-project.github.io/)]

- **RoboMIND**: Multi-embodiment intelligence benchmark for manipulation. [[Paper](https://arxiv.org/abs/2412.13877)] [[Project](https://x-humanoid-robomind.github.io/)]

- **VLABench**: Long-horizon reasoning benchmark for language-conditioned manipulation. [[Paper](https://arxiv.org/abs/2412.18194)] [[Project](https://vlabench.github.io/)]

- **SIMPLER**: Sim-to-real evaluation framework for manipulation policies. [[Paper](https://arxiv.org/abs/2405.05941)] [[Project](https://simpler-env.github.io/)]

- **RoboCasa**: Large-scale simulation for everyday household tasks. [[Paper](https://arxiv.org/abs/2407.10943)] [[Project](https://robocasa.ai/)]

- **CALVIN**: Tests long-horizon language-conditioned manipulation. [[Paper](https://arxiv.org/abs/2112.03227)] [[Project](http://calvin.cs.uni-freiburg.de/)]

- **RLBench**: 100 diverse manipulation tasks for benchmarking. [[Paper](https://arxiv.org/abs/1909.12271)] [[Project](https://sites.google.com/view/rlbench)]

- **ARNOLD**: Language-grounded task learning in realistic 3D scenes. [[Paper](https://arxiv.org/abs/2304.04321)] [[Project](https://arnold-benchmark.github.io/)]

- **ALFRED**: Vision-language navigation and manipulation benchmark. [[Paper](https://arxiv.org/abs/1912.01734)] [[Project](https://askforalfred.com/)]

- **GenSim / GenSim2**: LLM-based procedural task generation for scalable training. [[Paper](https://arxiv.org/abs/2310.01361)] [[Project](https://gen-sim.github.io/)]

- **MineDojo**: Minecraft-based platform with YouTube video pretraining. [[Paper](https://arxiv.org/abs/2206.08853)] [[Project](https://minedojo.org/)]

### Simulation Platforms

- **ManiSkill3**: GPU-parallelized simulation for generalizable embodied AI. [[Paper](https://arxiv.org/abs/2410.00425)] [[Project](https://www.maniskill.ai/)]

- **Genesis**: Differentiable physics engine for diverse simulation modalities. [[Project](https://genesis-embodied-ai.github.io/)]

- **Isaac Lab / Isaac Sim**: NVIDIA's production-ready robotics simulation. [[Project](https://developer.nvidia.com/isaac-sim)]

- **MuJoCo Playground**: Browser-based MuJoCo for quick prototyping. [[Project](https://playground.mujoco.org/)]

- **OmniGibson**: High-fidelity home simulation on NVIDIA Omniverse. [[Paper](https://arxiv.org/abs/2311.01014)] [[Project](https://behavior.stanford.edu/omnigibson/)]

- **Habitat 2.0**: Efficient simulation for navigation and rearrangement. [[Paper](https://arxiv.org/abs/2106.14405)] [[Project](https://aihabitat.org/)]

- **BEHAVIOR-1K**: Benchmark with 1,000 everyday activities. [[Paper](https://arxiv.org/abs/2403.09227)] [[Project](https://behavior.stanford.edu/)]

- **iGibson**: Realistic interactive environments with object state changes. [[Paper](https://arxiv.org/abs/2012.02924)] [[Project](https://svl.stanford.edu/igibson/)]

- **RoboSuite**: Modular framework with standardized manipulation tasks. [[Paper](https://arxiv.org/abs/2009.12293)] [[Project](https://robosuite.ai/)]

- **PyBullet**: Lightweight physics engine for RL research. [[Project](https://pybullet.org/)]

---

## Contributing

We welcome contributions! Please submit a pull request to add relevant papers, correct errors, or improve organization.

### Guidelines

- Focus on **Physical AI** papers (robotics, embodied agents, VLAs)
- Each paper should appear in only one category (the most relevant)
- Include proper citations with links to papers, projects, and code
- Add a one-line description explaining why each paper is important
- Verify all links are working

---

## Acknowledgments

This list draws inspiration from:
- [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics)
- [Awesome-Generalist-Agents](https://github.com/cheryyunl/awesome-generalist-agents)
- [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)
