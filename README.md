# LLM-Benchmark: Agentic Debate Framework for Marketing Sentiment Analysis

## Project Overview

This research project explores **best practices for using Large Language Models (LLMs) in marketing sentiment analysis** through a systematic comparison of different LLM methodologies. The project uses product reviews from **Sephora and Clinique** (scraped datasets) as the evaluation corpus, focusing on hate speech and offensive content detection in customer reviews.

### Research Scope

The codebase compares three primary LLM approaches for sentiment/classification tasks:

1. **Prompting Strategies** (e.g., zero-shot, few-shot-CoT)
2. **Fine-tuning Methods** (model adaptation)
3. **Agentic Configurations** (current focus - multi-agent debate systems)

### Current Focus: Agentic Configuration

The codebase currently emphasizes **multi-agent debate systems** where multiple agents argue different perspectives on text classification (e.g., hate speech vs. non-hate speech), with a judge agent generating the final classification. This approach aims to improve classification accuracy and reduce model sycophancy through structured debate.


## Project Status: Active Development

**Current State:**
- ✅ Working codebase with multi-agent debate architecture
- ✅ Core LLM integration abstracted in `LLMResponse`
- ⏳ **In Progress:** Prompt engineering and system optimization

**Current Challenges:**
- High computational cost: RTX required, ~2 hours runtime for 8 samples
- Excessive token usage and multiple internal passes per classification
- Need for extensive prompt modifications to improve performance
- Classification quality requires careful prompt tuning


## Key Concepts & Methodology

### Multi-Agent Debate Approach

The framework implements a structured debate process to improve classification:

1. **Meta Prompts:** Each agent receives a system prompt establishing their perspective
2. **Initial Arguments:** Each side presents opening arguments with reference reasoning
3. **Rebuttals:** Agents respond to opposing arguments with counter-reasoning
4. **Judge Resolution:** A neutral judge synthesizes the debate and outputs final classification

## Performance Notes

**Current Benchmark (Checkpoint W36):**
- **Hardware:** RTX required (CPU testing only)
- **Runtime:** ~2 hours for 8 samples
- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Max Tokens Per Response:** 4000
- **Rounds:** 2 debate rounds + 1 judge round

**Optimization Opportunities:**
- Token reduction through prompt compression
- Parallel processing of independent debates
- Smaller model variants or quantization
- Structured output formats (reduce parsing overhead)


**Last Updated:** W37, September 2025  
**Current Focus:** Prompt engineering and agentic configuration optimization  
**Status:** Active development - contributing to ongoing marketing sentiment analysis research
