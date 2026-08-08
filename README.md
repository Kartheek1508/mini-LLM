# mini-llm

# Building and Aligning a Small Language Model from Scratch

> *Understanding language models by building every component ourselves.*

---

## Why this project?

The recent progress in Large Language Models has made powerful AI systems more accessible than ever. However, most projects built around these models begin with an already-trained foundation and focus on fine-tuning or inference. While this approach is practical, it often hides the complexity behind how these models are actually created.

This project takes a different path.

Our goal is to build a modern GPT-style language model from the ground up—not because it is the easiest approach, but because it provides the deepest understanding of the entire development pipeline. Every major component, from tokenization and transformer architecture to training, instruction tuning, and alignment, is treated as something to be implemented, studied, and improved rather than simply used.

Rather than viewing an LLM as a black box, we want to understand the engineering decisions, mathematical foundations, and trade-offs that shape its behavior.

---

## More than just another language model

This repository is not intended to compete with frontier-scale models, nor is it an attempt to recreate them at a smaller scale. Instead, it is an exploration of how modern language models are built and how alignment techniques influence their capabilities.

By working with a model in the **100M–120M parameter range**, experimentation becomes practical while still preserving many of the architectural characteristics found in today's larger models. This allows us to iterate quickly, conduct controlled experiments, and better understand how individual design choices affect overall performance.

The project is therefore equal parts engineering exercise and research platform.

---

## Our philosophy

Modern AI research moves quickly, and it's easy to rely on high-level libraries that abstract away the difficult parts. While these abstractions are incredibly useful, they can also make it difficult to develop an intuition for what happens beneath the surface.

We believe that implementing these systems ourselves leads to a much deeper understanding than simply assembling existing components.

Throughout this project we aim to:

* Build before abstracting.
* Understand before optimizing.
* Experiment before drawing conclusions.
* Document not only what works, but also what doesn't.

The journey is just as valuable as the final model.

---

## A focus on alignment

Building a language model is only part of the story.

Modern language models are expected to be helpful, follow instructions, and produce responses that align with human preferences. These capabilities are not obtained during pretraining alone—they emerge through additional stages of fine-tuning and alignment.

One of the primary goals of this project is to investigate **Direct Preference Optimization (DPO)** as an alignment technique for smaller language models.

Rather than assuming alignment methods scale down effectively, we want to measure their impact directly and answer a simple question:

> **How much does preference alignment improve a language model in the 100M–120M parameter range?**

The resulting models will be evaluated through quantitative benchmarks as well as qualitative comparisons to understand where alignment helps—and where it falls short.

---

## What this repository represents

This repository is an open record of the project's development.

It contains experiments, implementation details, training infrastructure, successes, failures, and lessons learned while building a language model from scratch.

As development progresses, the repository will continue to evolve with improved implementations, training logs, evaluation results, and research findings.

The first stable release (v1.0) will include the complete training pipeline, aligned models, benchmarks, and comprehensive technical documentation.

---

## Current Status

🚧 **This project is actively under development.**

Many components are still being implemented, refined, and tested. Expect frequent changes, refactoring, and experimentation as we work toward the first stable release.

If you're interested in language models, alignment research, or understanding how modern LLMs are built from first principles, we hope you'll enjoy following the journey.
