# Model Architecture

The BirdNET Geomodel is a multi-task neural network that predicts species occurrence from raw location and time inputs.

## Design Philosophy

The model is designed with a key constraint: **at inference time, only latitude, longitude, and week number are needed**. No environmental data, no preprocessing — just three numbers.

To make this work, the model learns spatial and temporal patterns during training by jointly predicting:

1. **Species occurrence** (primary task) — which species are present at a location/time
2. **Environmental features** (auxiliary task) — what the environment looks like at a location

The auxiliary task acts as a regularizer, encouraging the model to learn meaningful spatial representations even when species labels are sparse.

## Architecture Overview

```mermaid
graph TD
    subgraph Input
        A[lat, lon, week]
    end

    subgraph CircularEncoding
        B["lat → sin/cos harmonics (2 × n)"]
        C["lon → sin/cos harmonics (2 × n)"]
        D["week → sin/cos harmonics (2 × n)"]
    end

    A --> B
    A --> C
    A --> D

    subgraph Encoder
        E["Concatenate lat + lon"]
        F[Linear Projection]
        G["Residual Block × N"]
        FiLM["FiLM: γ × block + β"]
        H[LayerNorm]
    end

    B --> E
    C --> E
    D --> FiLM
    E --> F --> G --> FiLM --> H

    subgraph Heads
        I["Species Head<br/>(multi-label classification)"]
        J["Environmental Head<br/>(regression, training only)"]
    end

    H --> I
    H --> J
```

## Components

### Circular Encoding

Raw coordinates and week numbers are poor inputs for neural networks — the model wouldn't know that longitude -180° and +180° are the same place, or that week 48 is adjacent to week 1.

**Circular encoding** solves this by mapping each value to sine/cosine pairs at multiple harmonics:

$$
\text{encode}(\theta) = [\sin(\theta), \cos(\theta), \sin(2\theta), \cos(2\theta), \ldots, \sin(n\theta), \cos(n\theta)]
$$

- **Latitude**: degrees → radians, then encoded with `coord_harmonics` harmonics (default 4 → 8 features)
- **Longitude**: same as latitude (8 features)
- **Week**: mapped to $[0, 2\pi)$ over 48 weeks, then encoded with `week_harmonics` harmonics (default 4 → 8 features)

Spatial input features: $2 \times 2 \times \text{coord\_harmonics}$ = 16 by default.  Week features (8) are used for FiLM conditioning rather than concatenated.

Year-round predictions (week 0) are computed at inference time as the **max** across all 48 weekly predictions — no special week-0 encoding is needed.

### Shared Encoder (`SpatioTemporalEncoder`)

The encoder maps spatial coordinates into a rich embedding, **modulated by temporal information** via FiLM (Feature-wise Linear Modulation):

1. **Spatial projection**: Concatenated lat+lon circular features → Linear to `embed_dim` (default 512)
2. **Residual blocks** — each block applies LayerNorm → GELU → Linear → LayerNorm → GELU → Dropout → Linear with a skip connection
3. **FiLM conditioning** — after each residual block, the week encoding generates per-block scale (γ) and shift (β) parameters: $x' = \gamma \cdot \text{block}(x) + \beta$.  This forces the model to actively modulate spatial representations based on the time of year.
4. **Final LayerNorm** for stable downstream processing

The pre-norm residual design ensures stable training and strong gradient flow even with many blocks.

### Species Prediction Head

A multi-label classification head that outputs one logit per species:

1. Residual blocks for further processing
2. **Low-rank bottleneck**: instead of a single large Linear(hidden → n_species), the head uses Linear(hidden → bottleneck) → GELU → Linear(bottleneck → n_species)

The bottleneck (default 128) dramatically reduces parameters when n_species is large (10K+) and learns a compact species-embedding space whose dimensions can be interpreted as latent ecological niches.

Output: raw logits (apply sigmoid for probabilities).

### Environmental Prediction Head

A regression head that predicts normalized environmental features (elevation, temperature, precipitation, etc.) from the shared embedding. Only used during training as an auxiliary objective.

## Model Sizes

| Size | Embed Dim | Encoder Blocks | Species Head | Bottleneck | Approx. Parameters |
|------|-----------|----------------|--------------|------------|-------------------|
| `small` | 256 | 3 | 256, 1 block | 64 | ~1.8M + species |
| `medium` | 512 | 4 | 512, 2 blocks | 128 | ~7.2M + species |
| `large` | 1024 | 6 | 1024, 3 blocks | 256 | ~36.5M + species |

The "+ species" part scales with the number of species in the vocabulary (bottleneck × n_species parameters).

## Encoding Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--coord_harmonics` | 4 | Higher values capture finer spatial patterns (more harmonics) |
| `--week_harmonics` | 4 | Higher values capture sharper weekly transitions |

!!! tip "Choosing harmonics"
    The default values (4 coordinate, 4 week) work well for global models. Higher harmonics add capacity for finer-grained patterns but increase input dimensionality and risk overfitting on small datasets.
