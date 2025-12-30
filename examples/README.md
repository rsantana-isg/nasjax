# NASJAX Examples

This directory contains example scripts demonstrating how to use NASJAX for neural architecture search.

## Running Examples

All examples work **without installing the package** - they can be run directly from source:

```bash
# From the repository root
python examples/quickstart.py
python examples/minimal_example.py
python examples/evolution_example.py
```

## Example Descriptions

### 1. `quickstart.py` - Quick Start Guide
**Time:** ~1 minute

The fastest way to get started with NASJAX. Shows how to evolve a neural network architecture with a single function call.

**What it demonstrates:**
- One-liner evolution with `evolve_architecture()`
- Minimal configuration
- Quick results

**Use this when:** You want to see NASJAX in action immediately.

---

### 2. `minimal_example.py` - Foundation Components
**Time:** ~5 seconds

Demonstrates the core building blocks of NASJAX without evolution.

**What it demonstrates:**
- Creating MLP descriptors (genotypes)
- Building Equinox networks (phenotypes)
- Forward passes (single and batch)
- JAX transformations (jit, vmap, grad)
- Descriptor serialization
- PyTree operations

**Use this when:** You want to understand the underlying architecture representation.

---

### 3. `evolution_example.py` - Complete Evolution Pipeline
**Time:** ~2-3 minutes

Comprehensive example showing all phases of the evolutionary algorithm.

**What it demonstrates:**
- Data preparation
- Evolution configuration
- Running the full evolution loop
- Analyzing results
- Training and evaluating the best network
- Saving results

**Use this when:** You want to see a complete neural architecture search workflow.

---

## Next Steps

After running these examples:
1. Read the [ROADMAP.md](../ROADMAP.md) for project overview
2. Check [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) for current progress
3. Explore the source code in `nasjax/` directory
4. Run the test suite: `pytest tests/`
5. Create your own custom evolution configurations!
