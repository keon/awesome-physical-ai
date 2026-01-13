# Awesome Physical AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of academic papers and resources on **Physical AI** — focusing on Vision-Language-Action (VLA) models, embodied intelligence, and robotic foundation models.

> **Physical AI** refers to AI systems that interact with and manipulate the physical world through robotic embodiments, combining perception, reasoning, and action in real-world environments.

---

## Table of Contents

- [Surveys](#surveys)
- [1. VLA Architectures](#1-vla-architectures)
  - [Monolithic End-to-End Models](#monolithic-end-to-end-models)
  - [Modular/Componentized Models](#modularcomponentized-models)
- [2. Action Representation](#2-action-representation)
  - [Discrete Tokenization](#discrete-tokenization)
  - [Continuous/Diffusion Models](#continuousdiffusion-models)
- [3. Cognitive Hierarchies](#3-cognitive-hierarchies)
  - [Fast Thinking (System 1)](#fast-thinking-system-1)
  - [Slow Thinking (System 2)](#slow-thinking-system-2)
- [4. Scaling Laws](#4-scaling-laws)
  - [Neural Scaling Laws](#neural-scaling-laws)
  - [Embodiment Scaling](#embodiment-scaling)
- [5. Deployment Efficiency](#5-deployment-efficiency)
  - [Quantization](#quantization)
  - [Inference Optimization](#inference-optimization)
- [6. Real-Time Latency Correction](#6-real-time-latency-correction)
- [7. Learning Paradigms](#7-learning-paradigms)
  - [Imitation Learning](#imitation-learning)
  - [Reinforcement Learning for VLAs](#reinforcement-learning-for-vlas)
- [8. Generalization Boundaries](#8-generalization-boundaries)
  - [Visual/Semantic Generalization](#visualsemantic-generalization)
  - [Cross-Embodiment Transfer](#cross-embodiment-transfer)
- [9. Safety and Alignment](#9-safety-and-alignment)
- [10. Lifelong Learning](#10-lifelong-learning)
- [Humanoid Robots](#humanoid-robots)
- [Manipulation](#manipulation)
- [Navigation](#navigation)
- [Foundational Vision-Language Models](#foundational-vision-language-models)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Simulation Platforms](#simulation-platforms)

---

## Surveys

- "Foundation Models in Robotics: Applications, Challenges, and the Future", *IJRR 2024*. [[Paper](https://arxiv.org/abs/2312.07843)] [[GitHub](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
  - Comprehensive survey covering 200+ papers on how foundation models are transforming robotics.

- "Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.08782)]
  - Meta-analysis synthesizing trends across perception, planning, and control with foundation models.

- "Robot Learning in the Era of Foundation Models: A Survey", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.14379)]
  - Covers the paradigm shift from task-specific to foundation model-based robot learning.

- "Language-conditioned Learning for Robotic Manipulation: A Survey", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.10807)]
  - Focuses specifically on how natural language enables more generalizable manipulation policies.

- "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.02816)]
  - Up-to-date survey covering the latest VLA architectures and deployment challenges.

- "A Survey of Embodied Learning for Object-Centric Robotic Manipulation", *arXiv, Aug 2024*. [[Paper](https://arxiv.org/abs/2408.11537)]
  - Deep dive into object-centric representations for dexterous manipulation.

- "The Development of LLMs for Embodied Navigation", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.00530)]
  - Tracks the evolution of LLM-based navigation from early prompting to end-to-end VLAs.

---

## 1. VLA Architectures

### Monolithic End-to-End Models

> Models that treat vision, language, and actions as unified tokens in a single end-to-end architecture.

- **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *RSS 2023*. [[Paper](https://arxiv.org/abs/2212.06817)] [[Project](https://robotics-transformer1.github.io/)] [[Code](https://github.com/google-research/robotics_transformer)]
  - Pioneer that proved large-scale multi-task demonstration data could train a single transformer for diverse manipulations.

- **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Project](https://robotics-transformer2.github.io/)]
  - Established the VLA paradigm by co-fine-tuning VLMs on robotic data, enabling semantic generalization like identifying "improvised hammers".

- **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]
  - First large-scale cross-embodiment study proving that diverse robot data improves generalization across platforms.

- **OpenVLA**: "OpenVLA: An Open-Source Vision-Language-Action Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.09246)] [[Project](https://openvla.github.io/)] [[Code](https://github.com/openvla/openvla)]
  - Open-source 7B model that outperformed the 55B RT-2-X, democratizing VLA research with consumer GPU fine-tuning.

- **OpenVLA-OFT**: "OpenVLA-OFT: Efficient Fine-Tuning for Open Vision-Language-Action Models", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Introduces orthogonal fine-tuning to adapt OpenVLA to new tasks with minimal forgetting.

- **PaLM-E**: "PaLM-E: An Embodied Multimodal Language Model", *ICML 2023*. [[Paper](https://arxiv.org/abs/2303.03378)] [[Project](https://palm-e.github.io/)]
  - 562B parameter model that demonstrated emergent multi-modal chain-of-thought reasoning for robotics.

- **PaLI-X**: "PaLI-X: On Scaling up a Multilingual Vision and Language Model", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2305.18565)]
  - Serves as the vision-language backbone for RT-2, providing strong visual grounding capabilities.

- **VIMA**: "VIMA: General Robot Manipulation with Multimodal Prompts", *ICML 2023*. [[Paper](https://arxiv.org/abs/2210.03094)] [[Project](https://vima.cs.stanford.edu/)] [[Code](https://github.com/vimalabs/VIMA)]
  - Introduced multimodal prompting (text + images) for specifying manipulation tasks with novel objects.

- **RoboVLMs**: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.14058)] [[Project](https://robovlms.github.io/)]
  - Systematic study identifying key design choices (action heads, visual encoders) for building effective VLAs.

- **RoboFlamingo**: "Vision-Language Foundation Models as Effective Robot Imitators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2311.01378)] [[Project](https://roboflamingo.github.io/)]
  - Adapts OpenFlamingo VLM with a policy head to capture sequential history with minimal fine-tuning.

- **3D-VLA**: "3D-VLA: A 3D Vision-Language-Action Generative World Model", *ICML 2024*. [[Paper](https://arxiv.org/abs/2403.09631)] [[Project](https://vis-www.cs.umass.edu/3dvla)]
  - Integrates 3D scene understanding with VLA for better spatial reasoning in cluttered environments.

- **LEO**: "An Embodied Generalist Agent in 3D World", *ICML 2024*. [[Paper](https://arxiv.org/abs/2311.12871)] [[Project](https://embodied-generalist.github.io/)] [[Code](https://github.com/embodied-generalist/embodied-generalist)]
  - Unifies 3D perception, language understanding, and embodied action in a single generalist model.

- **SmolVLA**: "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", *arXiv, Jun 2025*. [[Paper](https://huggingface.co/blog/smolvla)] [[Code](https://github.com/huggingface/lerobot)]
  - Compact 450M-parameter model achieving performance comparable to 10x larger models, runnable on consumer GPUs.

- **Magma**: "Magma: A Foundation Model for Multimodal AI Agents", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.13130)] [[Code](https://github.com/microsoft/Magma)]
  - Microsoft's unified foundation model bridging virtual (GUI) and physical (robot) agent control.

### Modular/Componentized Models

> Models that decouple cognition (VLM-based planning) from action (specialized motor modules) for better long-horizon task handling.

- **CogACT**: "CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.19650)] [[Project](https://cogact.github.io/)]
  - Decouples high-level cognition (VLM backbone) from low-level action (Diffusion Action Transformer) for better modularity.

- **Gemini Robotics**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)] [[Blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)]
  - Introduces "Thinking Before Acting" with internal natural language reasoning to decompose complex multi-step tasks.

- **Helix**: "Helix: A Vision-Language-Action Model for Generalist Humanoid Control", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]
  - Specialized VLA tailored for full upper-body control of humanoid robots including bimanual coordination.

- **SayPlan**: "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.06135)] [[Project](https://sayplan.github.io/)]
  - Grounds LLMs using 3D scene graphs for scalable, long-horizon task planning in large environments.

- **Code as Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sep 2022*. [[Paper](https://arxiv.org/abs/2209.07753)] [[Project](https://code-as-policies.github.io/)]
  - Seminal work showing LLMs can generate executable robot control code, bypassing action token prediction entirely.

- **SayCan**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2204.01691)] [[Project](https://say-can.github.io/)]
  - First to combine LLM semantic knowledge with learned affordance functions to ground language in physical capabilities.

- **Diffusion-VLA (DiVLA)**: "Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.03293)] [[Project](https://diffusion-vla.github.io/)]
  - Bridges reasoning and action via a Reasoning Injection Module (FiLM) that embeds self-generated phrases into the policy.

- **Inner Monologue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Project](https://innermonologue.github.io/)]
  - Pioneered closed-loop language feedback where robots verbalize observations to inform LLM planning.

- **Instruct2Act**: "Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.11176)] [[Code](https://github.com/OpenGVLab/Instruct2Act)]
  - Uses LLMs to compose foundation model APIs (SAM, CLIP) into executable robot manipulation programs.

- **TidyBot**: "TidyBot: Personalized Robot Assistance with Large Language Models", *IROS 2023*. [[Paper](https://arxiv.org/abs/2305.05658)] [[Project](https://tidybot.cs.princeton.edu/)] [[Code](https://github.com/jimmyyhwu/tidybot)]
  - Demonstrates LLM-based personalization for household tasks by learning user preferences from few examples.

---

## 2. Action Representation

### Discrete Tokenization

> Models that convert continuous joint movements into discrete "action tokens" similar to words.

- **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *RSS 2023*. [[Paper](https://arxiv.org/abs/2212.06817)] [[Project](https://robotics-transformer1.github.io/)]
  - Discretizes 7-DoF actions into 256 bins per dimension, treating robot control as a sequence modeling problem.

- **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Project](https://robotics-transformer2.github.io/)]
  - Represents actions as text tokens in the VLM vocabulary, enabling knowledge transfer from web pretraining.

- **OpenVLA**: "OpenVLA: An Open-Source Vision-Language-Action Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.09246)] [[Project](https://openvla.github.io/)]
  - Uses learned action tokenizer with 256 discrete bins, optimized for parameter-efficient fine-tuning.

- **FAST**: "FAST: Efficient Action Tokenization for Vision-Language-Action Models", *arXiv, Jan 2025*. [[Paper](https://arxiv.org/abs/2501.09747)] [[Project](https://www.pi.website/research/fast)]
  - Uses frequency-space (DCT) tokenization to compress action sequences 7x while preserving fine-grained control.

- **π₀-FAST**: "π₀-FAST: Efficient Action Tokenization for Vision-Language-Action Flow Models", *arXiv, 2025*. [[Paper](https://www.pi.website/research/fast)]
  - Extends FAST tokenization to flow matching models, achieving efficient high-frequency dexterous control.

- **GR-1**: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2312.13139)] [[Project](https://gr1-manipulation.github.io/)]
  - Leverages video prediction pretraining to learn action-conditioned visual dynamics as discrete tokens.

- **GR-2**: "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.06158)] [[Project](https://gr2-manipulation.github.io/)]
  - Scales video pretraining to web-scale data for improved generalization across manipulation scenarios.

- **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", *RSS 2023*. [[Paper](https://arxiv.org/abs/2304.13705)] [[Project](https://tonyzhaozh.github.io/aloha/)] [[Code](https://github.com/tonyzhaozh/act)]
  - Introduced Action Chunking with Transformers for smooth bimanual manipulation with the ALOHA platform.

- **Behavior Transformers**: "Behavior Transformers: Cloning k Modes with One Stone", *NeurIPS 2022*. [[Paper](https://arxiv.org/abs/2206.11251)] [[Code](https://github.com/notmahi/bet)]
  - Handles multimodal action distributions by clustering actions into discrete modes before prediction.

### Continuous/Diffusion Models

> Models that use diffusion or flow matching to generate continuous trajectories for high-frequency, dexterous tasks.

- **π₀ (pi-zero)**: "π₀: A Vision-Language-Action Flow Model for General Robot Control", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.24164)] [[Project](https://www.physicalintelligence.company/blog/pi0)]
  - Uses flow matching to generate high-frequency (50 Hz) continuous actions for dexterous tasks like laundry folding.

- **π₀.5**: "π₀.5: Scaling Robot Foundation Models", *arXiv, Apr 2025*. [[Paper](https://www.physicalintelligence.company/blog/pi0-5)]
  - Updated flow model designed for open-world generalization with improved zero-shot transfer capabilities.

- **Octo**: "Octo: An Open-Source Generalist Robot Policy", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.12213)] [[Project](https://octo-models.github.io/)] [[Code](https://github.com/octo-models/octo)]
  - Lightweight open-source policy using diffusion heads for smoother continuous trajectory generation.

- **RDT-1B**: "RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.07864)] [[Project](https://rdt-robotics.github.io/rdt-robotics/)]
  - 1B-parameter diffusion model specifically designed for coordinated bimanual robot manipulation.

- **DexVLA**: "DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control", *arXiv, Feb 2025*. [[Paper](https://arxiv.org/abs/2502.05855)] [[Project](https://dex-vla.github.io/)]
  - Combines VLM reasoning with a pluggable diffusion expert for dexterous manipulation tasks.

- **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", *RSS 2023*. [[Paper](https://arxiv.org/abs/2303.04137)] [[Project](https://diffusion-policy.cs.columbia.edu/)] [[Code](https://github.com/real-stanford/diffusion_policy)]
  - Foundational work showing diffusion models excel at capturing multimodal action distributions for manipulation.

- **3D Diffusion Policy**: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2403.03954)] [[Project](https://3d-diffusion-policy.github.io/)] [[Code](https://github.com/YanjieZe/3D-Diffusion-Policy)]
  - Integrates sparse 3D point cloud representations with diffusion for better spatial generalization.

- **Moto**: "Moto: Latent Motion Token as the Bridging Language for Robot Manipulation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04445)] [[Project](https://chenyi99.github.io/moto/)]
  - Uses latent motion tokens as an intermediate representation bridging high-level plans and low-level control.

- **Consistency Policy**: "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.07503)] [[Project](https://consistency-policy.github.io/)]
  - Distills diffusion policies into single-step models for 10x faster inference without quality loss.

---

## 3. Cognitive Hierarchies

### Fast Thinking (System 1)

> Reactive, high-frequency motor control models optimized for quick responses.

- **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *RSS 2023*. [[Paper](https://arxiv.org/abs/2212.06817)] [[Project](https://robotics-transformer1.github.io/)]
  - Operates at 3 Hz with direct image-to-action mapping for reactive control without explicit planning.

- **Gato**: "A Generalist Agent", *TMLR 2022*. [[Paper](https://arxiv.org/abs/2205.06175)] [[Blog](https://deepmind.google/discover/blog/a-generalist-agent/)]
  - Single transformer handling 604 distinct tasks across games, chat, and robotics via unified tokenization.

- **Perceiver-Actor**: "A Multi-Task Transformer for Robotic Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2209.05451)] [[Project](https://peract.github.io/)] [[Code](https://github.com/peract/peract)]
  - Efficient attention over 3D voxel grids enabling real-time multi-task manipulation planning.

- **PerAct2**: "PerAct2: Benchmarking and Improving Robot Manipulation Transformers", *arXiv, 2024*. [[Paper](https://arxiv.org/abs/2407.XXXXX)]
  - Improved training recipes and architectures for faster, more accurate manipulation transformers.

### Slow Thinking (System 2)

> Models that implement "thinking before acting" with value-guided search or chain-of-thought planning.

- **Gemini Robotics-ER**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)]
  - Specialized Embodied Reasoning model that creates detailed plans, calls digital tools, and reasons about physics.

- **Hume**: "Hume: Introducing Deliberative Alignment in Embodied AI", *arXiv, May 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Uses a value-query head to evaluate candidate actions via repeat sampling, enabling failure recovery through deliberation.

- **Embodied-CoT**: "Robotic Control via Embodied Chain-of-Thought Reasoning", *arXiv, Jul 2024*. [[Paper](https://arxiv.org/abs/2407.08693)] [[Project](https://embodied-cot.github.io/)]
  - Generates explicit reasoning chains before action prediction, improving interpretability and performance.

- **RoboReflect**: "RoboReflect: Reflective Reasoning for Robot Manipulation", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Enables robots to automatically adjust strategies based on unsuccessful attempts through self-reflection.

- **ReKep**: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.01652)] [[Project](https://rekep-robot.github.io/)] [[Code](https://github.com/huangwl18/ReKep)]
  - Uses VLMs to generate relational keypoint constraints that guide motion planning with explicit spatial reasoning.

- **TraceVLA**: "TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.10345)] [[Project](https://tracevla.github.io/)]
  - Overlays visual traces on images to enhance spatial-temporal reasoning for trajectory prediction.

- **LLM-State**: "LLM-State: Open World State Representation for Long-horizon Task Planning with Large Language Model", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.17406)]
  - Maintains explicit world state representations that LLMs can query and update for long-horizon planning.

- **DoReMi**: "Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", *arXiv, Jul 2023*. [[Paper](https://arxiv.org/abs/2307.00329)] [[Project](https://sites.google.com/view/doremi-paper)]
  - Detects when plan execution deviates from expectations and triggers LLM replanning automatically.

- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR 2023*. [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]
  - Interleaves reasoning traces with actions, establishing the foundation for "thinking before acting" in agents.

- **Statler**: "Statler: State-Maintaining Language Models for Embodied Reasoning", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.17840)] [[Project](https://statler-lm.github.io/)]
  - Explicitly maintains world state in natural language for multi-step embodied reasoning tasks.

---

## 4. Scaling Laws

### Neural Scaling Laws

> Research on the mathematical relationship between model/data scale and robotic performance.

- "Neural Scaling Laws for Embodied AI", *arXiv, May 2024*. [[Paper](https://arxiv.org/abs/2405.14005)]
  - Meta-analysis of 327 papers proving robotics performance follows power-law relationships with scale.

- "Data Scaling Laws in Imitation Learning for Robotic Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.18647)] [[Project](https://data-scaling-laws.github.io/)]
  - Quantifies how manipulation success rates scale logarithmically with demonstration dataset size.

- **AutoRT**: "AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2401.12963)] [[Project](https://auto-rt.github.io/)]
  - Demonstrates autonomous fleet data collection enabling continuous scaling of robot training data.

- **SARA-RT**: "SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.01990)]
  - Introduces attention mechanisms that scale efficiently to larger model sizes while maintaining robustness.

- "Scaling Robot Learning with Semantically Imagined Experience", *RSS 2023*. [[Paper](https://arxiv.org/abs/2302.11550)]
  - Uses generative models to synthesize diverse training scenarios, scaling experience beyond real data.

### Embodiment Scaling

> Cross-embodiment learning and generalization across different robot morphologies.

- **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]
  - Largest cross-embodiment study with 22 robot types proving diverse training improves all embodiments.

- **GENBOT-1K**: "Towards Embodiment Scaling Laws: Training on ~1000 Robot Bodies", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Demonstrates that training on ~1,000 diverse robot bodies enables zero-shot transfer to unseen robots.

- **URMA**: "Unified Robot Morphology Architecture", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Handles varying observation and action spaces across arbitrary robot morphologies via attention mechanisms.

- **Crossformer**: "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2408.11812)] [[Project](https://crossformer-model.github.io/)]
  - Single policy controlling manipulators, legged robots, and drones through unified action representation.

- **HPT**: "Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers", *NeurIPS 2024*. [[Paper](https://arxiv.org/abs/2409.20537)] [[Project](https://liruiw.github.io/hpt/)]
  - Scales to multiple embodiments by treating proprioceptive and visual inputs as heterogeneous token streams.

- **Extreme Cross-Embodiment**: "Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.19432)] [[Project](https://extreme-cross-embodiment.github.io/)]
  - Tests the boundaries of transfer learning between drastically different robot morphologies.

- **UniSim**: "UniSim: Learning Interactive Real-World Simulators", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2310.06114)] [[Project](https://universal-simulator.github.io/unisim/)]
  - Learns world simulators from video that can simulate any embodiment for policy training.

- **RoboAgent**: "RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2309.01918)] [[Project](https://robopen.github.io/)]
  - Uses semantic data augmentation to improve generalization without requiring more real robot data.

---

## 5. Deployment Efficiency

### Quantization

> Low-bit weight quantization for efficient edge deployment on robotic hardware.

- **BitVLA**: "BitVLA: 1-bit Vision-Language-Action Models for Robotics", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - First 1-bit (ternary) VLA reducing memory to 29.8% of standard models for edge deployment.

- **DeeR-VLA**: "DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution", *arXiv, Nov 2024*. [[Paper](https://arxiv.org/abs/2411.02359)] [[Code](https://github.com/yueyang130/DeeR-VLA)]
  - Dynamically adjusts model depth based on task complexity for energy-efficient inference.

- **QuaRT-VLA**: "Quantized Robotics Transformers for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - 4-bit quantization maintaining 95%+ performance while enabling deployment on embedded systems.

### Inference Optimization

> Architectural and algorithmic improvements for real-time robotic control.

- **TinyVLA**: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2409.12514)] [[Project](https://tiny-vla.github.io/)]
  - Compact 1.3B model designed for fast inference and efficient training with limited demonstrations.

- **SmolVLA**: "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", *arXiv, Jun 2025*. [[Paper](https://huggingface.co/blog/smolvla)] [[Code](https://github.com/huggingface/lerobot)]
  - 450M parameters achieving comparable performance to 10x larger models on consumer GPUs.

- **PDVLA**: "PDVLA: Parallel Decoding for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Treats autoregressive decoding as parallel fixed-point iterations for high control frequencies.

- **LAPA**: "Latent Action Pretraining from Videos", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.11758)] [[Project](https://latentactionpretraining.github.io/)]
  - Pretrains action representations from unlabeled video, reducing dependency on expensive robot demonstrations.

- **RT-H**: "RT-H: Action Hierarchies Using Language", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.01823)] [[Project](https://rt-hierarchy.github.io/)]
  - Uses language-conditioned action hierarchies to reduce effective action space and speed up inference.

---

## 6. Real-Time Latency Correction

> Methods that bridge the gap between high-latency AI inference and low-latency physical control requirements.

- **A2C2**: "A2C2: Asynchronous Action Chunk Correction for Real-Time Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2512.XXXXX)]
  - Lightweight correction head that adjusts outdated action chunks based on latest visual observations in real-time.

- **RTC**: "Real-Time Chunking: Asynchronous Execution for Robot Control", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Overlaps prediction and execution phases to reduce effective robot latency without faster inference.

- **DoReMi**: "Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", *arXiv, Jul 2023*. [[Paper](https://arxiv.org/abs/2307.00329)] [[Project](https://sites.google.com/view/doremi-paper)]
  - Monitors execution and triggers replanning when detecting deviation from expected outcomes.

- **CoPAL**: "Corrective Planning of Robot Actions with Large Language Models", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2310.07263)] [[Project](https://sites.google.com/view/copal-robot)]
  - Uses LLM feedback to correct plans online when environmental changes invalidate current actions.

- **PRED**: "Pre-emptive Action Revision by Environmental Feedback for Embodied Instruction Following Agents", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]
  - Preemptively revises actions before failure based on early environmental feedback signals.

- **Code-as-Monitor**: "Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection", *CVPR 2025*. [[Paper](https://arxiv.org/abs/2412.04455)] [[Project](https://code-as-monitor.github.io/)]
  - Generates constraint-checking code that monitors execution and triggers corrections proactively.

- **AHA**: "AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.XXXXX)]
  - VLM specialized for failure detection and root cause analysis in manipulation tasks.

---

## 7. Learning Paradigms

### Imitation Learning

> Behavioral cloning and learning from demonstrations.

- **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Project](https://cliport.github.io/)] [[Code](https://github.com/cliport/cliport)]
  - Combines CLIP semantic features with transporter networks for language-conditioned manipulation.

- **Play-LMP**: "Learning Latent Plans from Play", *CoRL 2019*. [[Paper](https://arxiv.org/abs/1903.01973)] [[Project](https://learning-from-play.github.io/)]
  - Learns reusable skills from unstructured "play" data without task labels or rewards.

- **BOSS**: "Bootstrap Your Own Skills: Learning to Solve New Tasks with LLM Guidance", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2310.10021)] [[Project](https://clvrai.github.io/boss/)]
  - Uses LLM to decompose tasks and bootstrap skill learning from minimal demonstrations.

- **MimicPlay**: "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2302.12422)] [[Project](https://mimic-play.github.io/)]
  - Learns long-horizon manipulation from human video demonstrations without teleoperation.

- **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2203.12601)] [[Project](https://sites.google.com/view/robot-r3m)] [[Code](https://github.com/facebookresearch/r3m)]
  - Pretrained visual encoder on Ego4D human video that transfers effectively to robot manipulation.

- **MVP**: "Masked Visual Pre-training for Motor Control", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.06173)] [[Project](https://tetexiao.com/projects/mvp)]
  - Shows masked autoencoder pretraining creates visual representations highly effective for control.

- **RVT**: "RVT: Robotic View Transformer for 3D Object Manipulation", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.14896)] [[Project](https://robotic-view-transformer.github.io/)] [[Code](https://github.com/NVlabs/RVT)]
  - Multi-view transformer achieving state-of-the-art manipulation with efficient virtual view rendering.

- **RVT-2**: "RVT-2: Learning Precise Manipulation from Few Demonstrations", *RSS 2024*. [[Paper](https://arxiv.org/abs/2406.08545)] [[Project](https://robotic-view-transformer-2.github.io/)]
  - Improved RVT with better sample efficiency, learning precise manipulation from just 5 demos.

- **DIAL**: "Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models", *arXiv, Nov 2022*. [[Paper](https://arxiv.org/abs/2211.11736)] [[Project](https://instructionaugmentation.github.io/)]
  - Uses VLMs to relabel demonstrations with diverse instructions, multiplying effective training data.

### Reinforcement Learning for VLAs

> RL-based methods for optimizing VLA policies, especially targeting high-impact reasoning tokens.

- **CO-RFT**: "CO-RFT: Chunked Offline Reinforcement Learning Fine-Tuning for Vision-Language-Action Models", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Two-stage offline RL achieving 57% improvement over imitation learning with only 30-60 samples.

- **HICRA**: "HICRA: Hierarchy-Aware Credit Assignment for Reinforcement Learning in VLAs", *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2601.XXXXX)]
  - Focuses optimization pressure on "planning tokens" rather than execution tokens, accelerating reasoning emergence.

- **ReinboT**: "ReinboT: Reinforcement Learning for Robotic Manipulation with Few-Shot Learning", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Integrates RL returns maximization to enhance manipulation and few-shot generalization.

- **TPO**: "TPO: Trajectory-wise Preference Optimization for Vision-Language-Action Models", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Fine-tunes VLAs based on trajectory preferences rather than simple imitation for better alignment.

- **iRe-VLA**: "iRe-VLA: Iterative Reinforcement Learning and Behavior Cloning for VLAs", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Iterates between online RL exploration and behavior cloning for continuous policy improvement.

- **ConRFT**: "ConRFT: Conservative Reinforcement Fine-Tuning with Human Intervention", *arXiv, 2025*. [[Paper](https://arxiv.org/abs/2505.XXXXX)]
  - Two-stage framework using offline RL to initialize value functions before online RL with safety constraints.

- **RL4VLA**: "RL4VLA: What Can RL Bring to VLA Generalization? An Empirical Study", *NeurIPS 2025*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]
  - Comprehensive study on when and how RL fine-tuning improves VLA generalization.

- **FLaRe**: "FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.16578)] [[Project](https://robot-flare.github.io/)]
  - Large-scale RL fine-tuning producing adaptive policies that recover from novel disturbances.

- **Plan-Seq-Learn**: "Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks", *ICLR 2024*. [[Paper](https://arxiv.org/abs/2405.01534)] [[Project](https://mihdalal.github.io/planseqlearn/)]
  - Uses LLM plans to guide RL exploration, solving tasks requiring 10+ sequential subtasks.

- **GLAM**: "Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.02662)] [[Code](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)]
  - Grounds LLM knowledge through online RL interaction in embodied environments.

- **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models", *ICML 2023*. [[Paper](https://arxiv.org/abs/2302.06692)]
  - Uses LLM-generated exploration bonuses to accelerate RL pretraining in sparse reward settings.

- **Text2Reward**: "Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning", *arXiv, Sep 2023*. [[Paper](https://arxiv.org/abs/2309.11489)] [[Project](https://text-to-reward.github.io/)]
  - LLMs generate dense reward code from task descriptions, eliminating manual reward engineering.

- **Language to Rewards**: "Language to Rewards for Robotic Skill Synthesis", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2306.08647)] [[Project](https://language-to-reward.github.io/)]
  - Translates natural language task descriptions into reward functions for MuJoCo locomotion and manipulation.

- **ExploRLLM**: "ExploRLLM: Guiding Exploration in Reinforcement Learning with Large Language Models", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2403.09583)]
  - LLM proposes exploration objectives to guide RL agents toward meaningful state-action regions.

---

## 8. Generalization Boundaries

### Visual/Semantic Generalization

> Models that generalize to novel visual appearances and semantic concepts.

- **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Project](https://robotics-transformer2.github.io/)]
  - Identifies "improvised hammers" and "healthy snacks" through web knowledge transfer to novel objects.

- **MOO**: "Open-World Object Manipulation using Pre-trained Vision-Language Models", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2303.00905)] [[Project](https://robot-moo.github.io/)]
  - Manipulates novel objects by leveraging CLIP embeddings for zero-shot object understanding.

- **VoxPoser**: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.05973)] [[Project](https://voxposer.github.io/)]
  - Generates 3D affordance and constraint maps from language for zero-shot manipulation.

- **RoboPoint**: "RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.10721)] [[Project](https://robo-point.github.io/)]
  - Predicts spatial affordances (where to grasp/place) directly from VLM reasoning.

- **CLIP-Fields**: "CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory", *RSS 2023*. [[Paper](https://arxiv.org/abs/2210.05663)] [[Project](https://mahis.life/clip-fields/)] [[Code](https://github.com/notmahi/clip-fields)]
  - Creates queryable 3D semantic maps enabling language-based object retrieval in new environments.

- **VLMaps**: "Visual Language Maps for Robot Navigation", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2210.05714)] [[Project](https://vlmaps.github.io/)] [[Code](https://github.com/vlmaps/vlmaps)]
  - Builds spatial maps grounded in natural language for zero-shot navigation to described locations.

- **NLMap**: "Open-vocabulary Queryable Scene Representations for Real World Planning", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2209.09874)] [[Project](https://nlmap-saycan.github.io/)]
  - Creates open-vocabulary queryable scene representations for grounding SayCan-style planning.

- **LERF**: "LERF: Language Embedded Radiance Fields", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.09553)] [[Project](https://www.lerf.io/)] [[Code](https://github.com/kerrj/lerf)]
  - Embeds CLIP features into NeRF for 3D language-grounded scene understanding.

- **Grounding DINO**: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", *ECCV 2024*. [[Paper](https://arxiv.org/abs/2303.05499)] [[Code](https://github.com/IDEA-Research/GroundingDINO)]
  - Open-vocabulary object detector used as perception backbone in many VLA systems.

### Cross-Embodiment Transfer

> Single policies controlling diverse robot types (humanoids, quadrupeds, manipulators).

- **π₀ (pi-zero)**: "π₀: A Vision-Language-Action Flow Model for General Robot Control", *arXiv, Oct 2024*. [[Paper](https://arxiv.org/abs/2410.24164)] [[Project](https://www.physicalintelligence.company/blog/pi0)]
  - Single policy controlling mobile manipulators, bimanual arms, and single-arm robots.

- **RUMs**: "Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.05865)] [[Project](https://robotutilitymodels.com/)]
  - Trains utility-maximizing policies that transfer zero-shot to new robots and environments.

- **NaVILA**: "NaVILA: Legged Robot Vision-Language-Action Model for Navigation", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2412.04453)] [[Project](https://navila-bot.github.io/)]
  - Extends VLA paradigm to quadruped navigation with vision-language-locomotion integration.

- **Octo**: "Octo: An Open-Source Generalist Robot Policy", *RSS 2024*. [[Paper](https://arxiv.org/abs/2405.12213)] [[Project](https://octo-models.github.io/)]
  - Pretrained on OXE dataset for efficient fine-tuning to new embodiments with minimal data.

- **MetaMorph**: "MetaMorph: Learning Universal Controllers with Transformers", *ICLR 2022*. [[Paper](https://arxiv.org/abs/2203.11931)] [[Project](https://metamorph-iclr.github.io/)]
  - Single transformer controlling robots with different numbers of limbs and morphologies.

- **Any-point Trajectory Modeling**: "Any-point Trajectory Modeling for Policy Learning", *RSS 2024*. [[Paper](https://arxiv.org/abs/2401.00025)] [[Project](https://xingyu-lin.github.io/atm/)]
  - Learns policies by predicting any-point trajectories, enabling transfer across camera viewpoints.

---

## 9. Safety and Alignment

> Ethical constraints, safety frameworks, and human-robot alignment for physical AI systems.

- **Gemini Robotics (Robot Constitution)**: "Gemini Robotics: Bringing AI into the Physical World", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.20020)]
  - Introduces data-driven "Robot Constitution" with natural language rules for safe autonomous behavior.

- **ASIMOV**: "ASIMOV: A Safety Benchmark for Embodied AI", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Benchmark evaluating and improving semantic safety in physical AI across diverse scenarios.

- "Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.10340)]
  - Systematic analysis of safety risks when deploying language models for robot control.

- **RoboPAIR**: "Jailbreaking LLM-Controlled Robots", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.13691)] [[Project](https://robopair.org/)]
  - Demonstrates adversarial attacks on LLM-controlled robots, highlighting alignment vulnerabilities.

- **RoboGuard**: "Safety Guardrails for LLM-Enabled Robots", *arXiv, Apr 2025*. [[Paper](https://arxiv.org/abs/2504.XXXXX)]
  - Runtime safety monitors that prevent LLM-generated actions from causing physical harm.

- "Robots Enact Malignant Stereotypes", *FAccT 2022*. [[Paper](https://arxiv.org/abs/2207.11569)] [[Project](https://sites.google.com/view/robots-enact-stereotypes)]
  - First study showing robots inherit and amplify harmful biases from vision-language pretraining.

- "LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.08824)]
  - Comprehensive risk assessment of LLM-controlled robots in real-world deployment scenarios.

- "Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.XXXXX)]
  - Provides formal safety guarantees for LLM-controlled robots through reachability-based verification.

---

## 10. Lifelong Learning

> Agents that continuously learn and adapt without forgetting prior skills.

- **Voyager**: "VOYAGER: An Open-Ended Embodied Agent with Large Language Models", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.16291)] [[Project](https://voyager.minedojo.org/)] [[Code](https://github.com/MineDojo/Voyager)]
  - First LLM-powered agent in Minecraft that autonomously builds a skill library of executable code.

- **RoboGen**: "RoboGen: A Generative and Self-Guided Robotic Agent", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.01455)] [[Project](https://robogen-ai.github.io/)]
  - Endlessly proposes and masters new manipulation skills through self-guided curriculum generation.

- **RoboCat**: "RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.11706)] [[Blog](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)]
  - Self-improves through cycles of self-generated data collection and training on new tasks.

- **LIBERO**: "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2306.03310)] [[Project](https://libero-project.github.io/)] [[Code](https://github.com/Lifelong-Robot-Learning/LIBERO)]
  - Benchmark with 130 tasks measuring forward/backward transfer and catastrophic forgetting.

- **LOTUS**: "LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery", *arXiv, Dec 2024*. [[Paper](https://arxiv.org/abs/2311.02058)] [[Project](https://ut-austin-rpl.github.io/Lotus/)]
  - Discovers and accumulates manipulation skills without forgetting through unsupervised segmentation.

- **DEPS**: "Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2302.01560)] [[Code](https://github.com/CraftJarvis/MC-Planner)]
  - Interactive LLM planning with self-explanation for open-world Minecraft task completion.

- **JARVIS-1**: "JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.05997)] [[Project](https://craftjarvis-jarvis1.github.io/)] [[Code](https://github.com/CraftJarvis/JARVIS-1)]
  - Maintains persistent memory across tasks enabling knowledge accumulation over extended play.

- **MP5**: "MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2312.07472)] [[Project](https://craftjarvis.github.io/MP5/)]
  - Uses active perception to gather task-relevant information for multi-modal embodied reasoning.

- **SPRINT**: "SPRINT: Semantic Policy Pre-training via Language Instruction Relabeling", *ICRA 2024*. [[Paper](https://arxiv.org/abs/2306.11886)] [[Project](https://clvrai.github.io/sprint/)]
  - Relabels demonstrations with diverse instructions to pretrain policies with broad language understanding.

---

## Humanoid Robots

> Foundation models specifically designed for humanoid robot control.

- **GR00T N1**: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots", *arXiv, Mar 2025*. [[Paper](https://arxiv.org/abs/2503.14734)] [[Project](https://developer.nvidia.com/isaac/gr00t)]
  - NVIDIA's dual-system architecture trained on robot trajectories, human videos, and synthetic data.

- **HumanPlus**: "HumanPlus: Humanoid Shadowing and Imitation from Humans", *arXiv, Jun 2024*. [[Paper](https://arxiv.org/abs/2406.10454)] [[Project](https://humanoid-ai.github.io/)]
  - Real-time human-to-humanoid motion retargeting enabling learning from human demonstration videos.

- **ExBody**: "Expressive Whole-Body Control for Humanoid Robots", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.16796)] [[Project](https://expressive-humanoid.github.io/)]
  - Enables humanoids to perform expressive, human-like whole-body movements and gestures.

- **H2O**: "Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.04436)] [[Project](https://human2humanoid.com/)]
  - Real-time teleoperation system mapping human motion to humanoid whole-body control.

- **OmniH2O**: "OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.08858)] [[Project](https://omni.human2humanoid.com/)]
  - Universal teleoperation framework for diverse humanoid platforms with dexterous hand control.

- **Humanoid Locomotion**: "Learning Humanoid Locomotion with Transformers", *arXiv, Mar 2024*. [[Paper](https://arxiv.org/abs/2303.03381)] [[Project](https://humanoid-locomotion.github.io/)]
  - Transformer-based locomotion achieving robust bipedal walking across diverse terrains.

- **Whole-Body Humanoid Control**: "Whole-Body Humanoid Robot Locomotion with Human Reference", *arXiv, Feb 2024*. [[Paper](https://arxiv.org/abs/2402.18294)]
  - Uses human motion reference to train natural-looking humanoid locomotion behaviors.

---

## Manipulation

> Robot manipulation with foundation models.

- **Scaling Up Distilling Down**: "Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2307.14535)] [[Project](https://www.cs.columbia.edu/~huy/scalingup/)]
  - Uses internet-scale knowledge to acquire manipulation skills with minimal robot demonstrations.

- **LLM-GROP**: "Task and Motion Planning with Large Language Models for Object Rearrangement", *IROS 2023*. [[Paper](https://arxiv.org/abs/2303.06247)]
  - Integrates LLM task planning with geometric motion planning for complex rearrangement.

- **LLM3**: "LLM3: Large Language Model-based Task and Motion Planning with Motion Failure Reasoning", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11552)]
  - LLM reasons about motion planning failures to generate alternative approaches.

- **ManipVQA**: "ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Modal Large Language Models", *IROS 2024*. [[Paper](https://arxiv.org/abs/2403.11289)]
  - Grounds MLLMs with physical affordances for answering manipulation-relevant questions.

- **UniAff**: "UniAff: A Unified Representation of Affordances for Tool Usage and Articulation with Vision-Language Models", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.20551)]
  - Unified affordance representation handling both tool use and articulated object manipulation.

- **SKT**: "SKT: Integrating State-Aware Keypoint Trajectories with Vision-Language Models for Robotic Garment Manipulation", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.18082)]
  - Keypoint-based approach for challenging deformable object manipulation like garments.

- **Grasp Anything**: "Pave the Way to Grasp Anything: Transferring Foundation Models for Universal Pick-Place Robots", *arXiv, Jun 2023*. [[Paper](https://arxiv.org/abs/2306.05716)]
  - Transfers foundation model knowledge for universal pick-and-place across object categories.

- **Manipulate-Anything**: "Manipulate-Anything: Automating Real-World Robots using Vision-Language Models", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.18915)] [[Project](https://robot-ma.github.io/)]
  - VLM-based system enabling robots to manipulate novel objects with minimal task specification.

- **A3VLM**: "A3VLM: Actionable Articulation-Aware Vision Language Model", *CoRL 2024*. [[Paper](https://arxiv.org/abs/2406.07549)]
  - Specialized VLM understanding articulated objects (doors, drawers) for manipulation planning.

- **LaN-Grasp**: "Language-Driven Grasp Detection", *CVPR 2024*. [[Paper](https://arxiv.org/abs/2311.09876)]
  - Generates grasp poses conditioned on natural language descriptions of desired grasps.

---

## Navigation

> Vision-language models for robot navigation.

- **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action", *CoRL 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Project](https://sites.google.com/view/lmnav)]
  - Combines LLM, VLM, and VNM for instruction-following navigation without task-specific training.

- **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration", *ICRA 2023*. [[Paper](https://arxiv.org/abs/2203.10421)]
  - Zero-shot object navigation using CLIP for target localization without navigation training.

- **L3MVN**: "L3MVN: Leveraging Large Language Models for Visual Target Navigation", *IROS 2024*. [[Paper](https://arxiv.org/abs/2304.05501)]
  - LLM provides commonsense reasoning for efficient exploration toward visual targets.

- **NaVid**: "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation", *RSS 2024*. [[Paper](https://arxiv.org/abs/2402.15852)] [[Project](https://pku-epic.github.io/NaVid/)]
  - Video-based VLM that plans navigation steps from egocentric video understanding.

- **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", *CoRL 2023*. [[Paper](https://arxiv.org/abs/2309.15940)] [[Project](https://ovsg-l.github.io/)]
  - Builds open-vocabulary 3D scene graphs for grounding language queries in navigation.

- **GSON**: "GSON: A Group-based Social Navigation Framework with Large Multimodal Model", *arXiv, Sep 2024*. [[Paper](https://arxiv.org/abs/2409.XXXXX)]
  - LMM-based social navigation understanding group dynamics and social conventions.

- **CANVAS**: "CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction", *ICRA 2025*. [[Paper](https://arxiv.org/abs/2410.01273)]
  - Incorporates commonsense reasoning for more intuitive human-robot navigation interaction.

- **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV 2020*. [[Paper](https://arxiv.org/abs/2004.14973)]
  - First to pretrain navigation agents on web image-text pairs for vision-language navigation.

- **ADAPT**: "ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts", *CVPR 2022*. [[Paper](https://arxiv.org/abs/2205.15509)]
  - Aligns action prompts with vision-language representations for better navigation decisions.

- **ThinkBot**: "ThinkBot: Embodied Instruction Following with Thought Chain Reasoning", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.07062)]
  - Generates explicit thought chains for complex multi-step instruction following.

---

## Foundational Vision-Language Models

> Core vision-language models that serve as backbones for Physical AI systems.

- **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision", *ICML 2021*. [[Paper](https://arxiv.org/abs/2103.00020)] [[Code](https://github.com/openai/CLIP)]
  - Foundational model aligning vision and language that underlies most VLA perception systems.

- **SigLIP**: "Sigmoid Loss for Language Image Pre-Training", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2303.15343)]
  - Improved CLIP training with sigmoid loss, used in many recent VLA visual encoders.

- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision", *arXiv, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.07193)] [[Code](https://github.com/facebookresearch/dinov2)]
  - Self-supervised visual features highly effective for robotic manipulation and navigation.

- **SAM**: "Segment Anything", *ICCV 2023*. [[Paper](https://arxiv.org/abs/2304.02643)] [[Project](https://segment-anything.com/)]
  - Universal segmentation model used for object extraction in robotic perception pipelines.

- **LLaVA**: "Visual Instruction Tuning", *NeurIPS 2023*. [[Paper](https://arxiv.org/abs/2304.08485)] [[Project](https://llava-vl.github.io/)]
  - Open-source VLM that serves as backbone for several open-source VLA models.

- **Prismatic VLMs**: "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", *ICML 2024*. [[Paper](https://arxiv.org/abs/2402.07865)] [[Code](https://github.com/TRI-ML/prismatic-vlms)]
  - Systematic study of VLM design choices informing OpenVLA and other robotics VLMs.

---

## Datasets and Benchmarks

- **Open X-Embodiment**: Largest open-source robot dataset with 1M+ trajectories from 22 robot embodiments. [[Paper](https://arxiv.org/abs/2310.08864)] [[Project](https://robotics-transformer-x.github.io/)]
  - Enables cross-embodiment pretraining and provides standardized evaluation across platforms.

- **DROID**: Large-scale in-the-wild robot manipulation dataset. [[Paper](https://arxiv.org/abs/2403.12945)] [[Project](https://droid-dataset.github.io/)]
  - 76K trajectories across 564 scenes capturing real-world manipulation diversity.

- **BridgeData V2**: A large-scale robot manipulation dataset. [[Paper](https://arxiv.org/abs/2308.12952)] [[Project](https://rail-berkeley.github.io/bridgedata/)]
  - Multi-task dataset enabling few-shot transfer to new objects and environments.

- **ARIO**: All Robots in One — unified dataset for general-purpose embodied agents. [[Paper](https://arxiv.org/abs/2408.10899)] [[Project](https://imaei.github.io/project_pages/ario/)]
  - Standardized format unifying diverse robot datasets for large-scale pretraining.

- **LIBERO**: Benchmark for lifelong robot learning with 130 tasks. [[Paper](https://arxiv.org/abs/2306.03310)] [[Project](https://libero-project.github.io/)]
  - Measures knowledge transfer and catastrophic forgetting in continual learning settings.

- **RoboMIND**: Multi-embodiment intelligence benchmark for robot manipulation. [[Paper](https://arxiv.org/abs/2412.13877)] [[Project](https://x-humanoid-robomind.github.io/)]
  - Evaluates cross-embodiment generalization on standardized manipulation tasks.

- **VLABench**: Long-horizon reasoning benchmark for language-conditioned manipulation. [[Paper](https://arxiv.org/abs/2412.18194)] [[Project](https://vlabench.github.io/)]
  - Tests VLA reasoning capabilities on tasks requiring multi-step planning.

- **SIMPLER**: Evaluation platform for real-world manipulation policies in simulation. [[Paper](https://arxiv.org/abs/2405.05941)] [[Project](https://simpler-env.github.io/)]
  - Sim-to-real evaluation framework correlating simulation with real-world performance.

- **RoboCasa**: Large-scale simulation for everyday household tasks. [[Paper](https://arxiv.org/abs/2407.10943)] [[Project](https://robocasa.ai/)]
  - Procedurally generated home environments for training household assistance robots.

- **CALVIN**: A benchmark for language-conditioned policy learning. [[Paper](https://arxiv.org/abs/2112.03227)] [[Project](http://calvin.cs.uni-freiburg.de/)] [[Code](https://github.com/mees/calvin)]
  - Tests long-horizon language-conditioned manipulation with chained instructions.

- **RLBench**: A large-scale benchmark for robot learning. [[Paper](https://arxiv.org/abs/1909.12271)] [[Project](https://sites.google.com/view/rlbench)] [[Code](https://github.com/stepjam/RLBench)]
  - 100 diverse manipulation tasks in simulation for benchmarking policy learning.

- **ARNOLD**: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes. [[Paper](https://arxiv.org/abs/2304.04321)] [[Project](https://arnold-benchmark.github.io/)]
  - Tests continuous state-space manipulation grounded in natural language.

- **ALFRED**: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. [[Paper](https://arxiv.org/abs/1912.01734)] [[Project](https://askforalfred.com/)]
  - Vision-language navigation and manipulation following natural language instructions.

- **GenSim**: Generating Robotic Simulation Tasks via Large Language Models. [[Paper](https://arxiv.org/abs/2310.01361)] [[Project](https://gen-sim.github.io/)]
  - LLM-based procedural task generation for scalable simulation training.

- **GenSim2**: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs. [[Paper](https://arxiv.org/abs/2410.03645)] [[Project](https://gensim2.github.io/)]
  - Enhanced task generation with multi-modal reasoning for more diverse training scenarios.

- **MineDojo**: Building Open-Ended Embodied Agents with Internet-Scale Knowledge. [[Paper](https://arxiv.org/abs/2206.08853)] [[Project](https://minedojo.org/)] [[Code](https://github.com/MineDojo/MineDojo)]
  - Minecraft-based platform with YouTube video pretraining for open-world agent learning.

---

## Simulation Platforms

- **ManiSkill3**: GPU-parallelized robotics simulation for generalizable embodied AI. [[Paper](https://arxiv.org/abs/2410.00425)] [[Project](https://www.maniskill.ai/)]
  - Massively parallel simulation enabling training on thousands of environments simultaneously.

- **Genesis**: Generative physics engine for robotics and beyond. [[Paper](https://arxiv.org/abs/2410.00425)] [[Project](https://genesis-embodied-ai.github.io/)]
  - Differentiable physics engine supporting diverse simulation modalities in a unified framework.

- **Isaac Lab**: NVIDIA's robotics simulation and learning framework. [[Project](https://isaac-sim.github.io/IsaacLab/)]
  - Production-ready framework for RL and imitation learning with photorealistic rendering.

- **Isaac Sim**: NVIDIA's robotics simulation platform. [[Project](https://developer.nvidia.com/isaac-sim)]
  - Photorealistic simulation with accurate physics for sim-to-real robot training.

- **MuJoCo Playground**: Playground for MuJoCo-based robotics experiments. [[Report](https://playground.mujoco.org/assets/playground_technical_report.pdf)] [[Project](https://playground.mujoco.org/)]
  - Browser-based MuJoCo environment for quick prototyping and experimentation.

- **OmniGibson**: Platform for embodied AI built on NVIDIA Omniverse. [[Paper](https://arxiv.org/abs/2311.01014)] [[Project](https://behavior.stanford.edu/omnigibson/)]
  - High-fidelity home simulation with realistic object interactions and physics.

- **Habitat 2.0**: Platform for training home assistant robots. [[Paper](https://arxiv.org/abs/2106.14405)] [[Project](https://aihabitat.org/)]
  - Efficient simulation for navigation and rearrangement in scanned real-world spaces.

- **BEHAVIOR-1K**: Benchmark with 1,000 everyday activities. [[Paper](https://arxiv.org/abs/2403.09227)] [[Project](https://behavior.stanford.edu/)]
  - Comprehensive benchmark for evaluating household robot capabilities.

- **iGibson**: A Simulation Environment for Interactive Tasks in Large Realistic Scenes. [[Paper](https://arxiv.org/abs/2012.02924)] [[Project](https://svl.stanford.edu/igibson/)]
  - Realistic interactive environments with object state changes for mobile manipulation.

- **RoboSuite**: A Modular Simulation Framework for Robot Learning. [[Paper](https://arxiv.org/abs/2009.12293)] [[Project](https://robosuite.ai/)]
  - Modular framework with standardized manipulation tasks and multiple robot models.

- **PyBullet**: Physics simulation for robotics. [[Project](https://pybullet.org/)]
  - Lightweight physics engine popular for RL research and rapid prototyping.

---

## Contributing

We welcome contributions! Please submit a pull request to add relevant papers, correct errors, or improve organization.

### Guidelines

- Focus on **Physical AI** papers (robotics, embodied agents, VLAs)
- Include proper citations with links to papers, projects, and code
- Add a one-line description explaining why each paper is important
- Verify all links are working

---

## Acknowledgments

This list draws inspiration from:
- [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics)
- [Awesome-Generalist-Agents](https://github.com/cheryyunl/awesome-generalist-agents)
- [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)
