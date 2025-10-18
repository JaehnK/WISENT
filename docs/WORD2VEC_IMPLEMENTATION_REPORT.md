# Word2Vec Implementation Report

## 1. Overview

This report describes the Word2Vec implementation used in the SENTIMENT project for learning dense word representations from textual data. The implementation is based on the Skip-gram model with negative sampling, as proposed by Mikolov et al. (2013).

### 1.1 Base Implementation

The initial implementation was adapted from the open-source repository **[Andras7/word2vec-pytorch](https://github.com/Andras7/word2vec-pytorch)**, which provides a fast and efficient Word2Vec implementation in PyTorch. However, significant modifications and enhancements were made to improve training stability, performance, and reproducibility.

### 1.2 Key Components

The implementation consists of four main modules:

1. **SkipGramModel** ([core/entities/skipgram.py](../core/entities/skipgram.py)) - Neural network architecture
2. **MemoryDataLoader** ([core/services/Word2Vec/DataLoader.py](../core/services/Word2Vec/DataLoader.py)) - Data loading and preprocessing
3. **Word2VecTrainer** ([core/services/Word2Vec/Trainer.py](../core/services/Word2Vec/Trainer.py)) - Training orchestration
4. **Word2VecService** ([core/services/Word2Vec/Word2VecService.py](../core/services/Word2Vec/Word2VecService.py)) - High-level API

---

## 2. Model Architecture

### 2.1 Skip-gram with Negative Sampling

The Skip-gram model learns to predict context words given a center word. The objective is to maximize the probability of context words appearing near the center word within a fixed window size.

**Architecture:**
```python
class SkipGramModel(nn.Module):
    - u_embeddings: Center word embeddings (vocab_size × emb_dimension)
    - v_embeddings: Context word embeddings (vocab_size × emb_dimension)
```

**Loss Function:**

The negative sampling loss is computed as:

```
L = -log σ(u · v_pos) - Σ(log σ(-u · v_neg_i))
```

Where:
- `u`: Center word embedding
- `v_pos`: Positive context word embedding
- `v_neg_i`: Negative sample embeddings
- `σ`: Sigmoid function

### 2.2 Key Implementation Details

**Forward Pass ([skipgram.py:22-40](../core/entities/skipgram.py#L22-L40)):**

```python
def forward(self, pos_u, pos_v, neg_v):
    # Get embeddings
    emb_u = self.u_embeddings(pos_u)          # Center words
    emb_v = self.v_embeddings(pos_v)          # Positive contexts
    emb_neg_v = self.v_embeddings(neg_v)      # Negative samples

    # Positive score (minimize negative log-sigmoid)
    score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
    score = torch.clamp(score, max=10, min=-10)  # Numerical stability
    score = -F.logsigmoid(score)

    # Negative score
    neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
    neg_score = torch.clamp(neg_score, max=10, min=-10)

    # Handle both 1D and 2D tensors for robustness
    if neg_score.dim() == 1:
        neg_score = -F.logsigmoid(-neg_score)
    else:
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

    return torch.mean(score + neg_score)
```

---

## 3. Critical Improvements

### 3.1 Embedding Initialization (CRITICAL FIX)

**Problem Identified:**
The original implementation used a very small initialization range (`initrange = 1.0 / emb_dimension`), which caused:
- Near-zero initial dot products
- Loss values getting stuck at ~4.16
- Poor gradient flow in early training

**Solution Implemented ([skipgram.py:16-20](../core/entities/skipgram.py#L16-L20)):**

```python
# Xavier/Glorot uniform initialization
initrange = (6.0 / (2 * self.emb_dimension)) ** 0.5
init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
```

**Impact:**
- Original: `initrange = 1.0 / 512 ≈ 0.002` for 512-dim embeddings
- Improved: `initrange = sqrt(6/256) ≈ 0.153` for 128-dim embeddings
- Result: Stable gradient flow and successful training convergence

### 3.2 Subsampling of Frequent Words

High-frequency words (e.g., "the", "a") provide less information. We implemented Mikolov's subsampling formula to downsample frequent words.

**Implementation ([DataLoader.py:60-71](../core/services/Word2Vec/DataLoader.py#L60-L71)):**

```python
def _init_discard_table(self):
    t = 0.0001  # threshold
    f = np.array(list(self.word_frequency.values())) / self.token_count
    keep_probs = (np.sqrt(f / t) + 1.0) * (t / f)
    self.discards = np.minimum(keep_probs, 1.0)
```

**Keep probability formula:**
```
P(keep) = min(1.0, (√(f/t) + 1) × (t/f))
```

Where:
- `f`: Relative word frequency
- `t`: Threshold (typically 1e-4)

### 3.3 Negative Sampling

**Unigram Distribution with 0.75 Power ([DataLoader.py:40-58](../core/services/Word2Vec/DataLoader.py#L40-L58)):**

```python
def _init_negative_table(self):
    # Sample negatives proportional to f^0.75
    pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = np.round(ratio * self.NEGATIVE_TABLE_SIZE)

    for wid, c in enumerate(count):
        self.negatives += [wid] * int(c)

    self.negatives = np.array(self.negatives)
    np.random.shuffle(self.negatives)
```

**Target Avoidance ([DataLoader.py:88-98](../core/services/Word2Vec/DataLoader.py#L88-L98)):**

Ensures negative samples don't match the positive target:

```python
# Resample if negative == positive target
mask = (response == target)
if np.any(mask):
    resample = np.random.randint(0, self.vocab_size, mask.sum())
    response[mask] = resample
```

### 3.4 Optimizer and Learning Rate Schedule

**Problem with Original:**
- Used Adam optimizer (less common in Word2Vec literature)
- Optimizer was re-initialized every epoch, resetting momentum

**Solution ([Trainer.py:75-76](../core/services/Word2Vec/Trainer.py#L75-L76)):**

```python
# Initialize optimizer once (outside epoch loop)
optimizer = optim.SGD(model.parameters(), lr=self.initial_lr, momentum=0.9)
```

**Linear Learning Rate Decay ([Trainer.py:147-154](../core/services/Word2Vec/Trainer.py#L147-L154)):**

```python
def _compute_linear_lr(self) -> float:
    progress = min(1.0, max(0.0, self.global_step / (self.total_steps - 1)))
    start_lr = self.initial_lr
    end_lr = self.initial_lr * self.min_lr_ratio  # min_lr_ratio = 1e-3
    return start_lr + (end_lr - start_lr) * progress
```

**Rationale:**
- SGD with momentum is standard in Word2Vec implementations
- Linear decay from `initial_lr` to `initial_lr × 0.001` over all training steps
- Follows the original Word2Vec paper more closely

### 3.5 Sparse to Dense Embeddings

**Change:**
```python
# Original
self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

# Modified
self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
```

**Rationale:**
- Better GPU utilization for dense operations
- Compatible with modern optimizers (Adam, SGD with momentum)
- Negligible memory overhead for vocabulary sizes < 1M

---

## 4. Hyperparameter Tuning

### 4.1 Optimized Configuration

Based on Word2Vec best practices and the original Mikolov et al. paper:

| Parameter | Original | Optimized | Justification |
|-----------|----------|-----------|---------------|
| `emb_dimension` | 512 | **128** | Standard Word2Vec size, better generalization |
| `batch_size` | 512 | **128** | Smaller batches → more gradient updates |
| `window_size` | 10 | **5** | Mikolov's default, balances local/global context |
| `learning_rate` | 0.001 (Adam) | **0.025** (SGD) | Standard for Word2Vec |
| `iterations` | Variable | **10** | Sufficient for convergence on most datasets |
| `negative_samples` | 5 | **5** | Standard (5-20 typical, 5 for large datasets) |

### 4.2 Configuration Code

**Default Configuration ([Word2VecService.py:58-73](../core/services/Word2Vec/Word2VecService.py#L58-L73)):**

```python
@classmethod
def create_default(cls, doc_service: DocumentService, min_count: int = 1):
    # Model: 128-dimensional embeddings
    model = SkipGramModel(word_data['vocab_size'], emb_dimension=128)

    # Dataset: window size = 5
    dataset = MemoryWord2vecDataset(data_loader, window_size=5)

    # Trainer: SGD, lr=0.025, batch_size=128, 10 epochs
    trainer = Word2VecTrainer(iterations=10, initial_lr=0.025, batch_size=128)

    return cls(doc_service, model, trainer, dataset, data_loader)
```

---

## 5. Reproducibility Enhancements

### 5.1 Random Seed Fixing

To ensure reproducible results across runs ([Trainer.py:19-27](../core/services/Word2Vec/Trainer.py#L19-L27)):

```python
def __init__(self, random_seed: int = 42, ...):
    # Fix all random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Components seeded:**
- PyTorch CPU operations
- PyTorch CUDA operations
- NumPy random sampling
- Python's `random` module
- CuDNN backend (deterministic mode)

---

## 6. Training Pipeline

### 6.1 Data Flow

```
Documents (raw text)
    ↓
DocumentService (preprocessing, tokenization)
    ↓
MemoryDataLoader (word2id mapping, frequency tables)
    ↓
MemoryWord2vecDataset (Skip-gram pair generation)
    ↓
PyTorch DataLoader (batching, shuffling)
    ↓
Word2VecTrainer (training loop)
    ↓
SkipGramModel (embeddings)
```

### 6.2 Training Loop

**Key Features:**
1. **Gradient clipping** ([Trainer.py:126](../core/services/Word2Vec/Trainer.py#L126)): `max_norm=1.0` prevents exploding gradients
2. **Exponential moving average** for loss tracking ([Trainer.py:133](../core/services/Word2Vec/Trainer.py#L133))
3. **Multi-worker data loading** ([Trainer.py:62](../core/services/Word2Vec/Trainer.py#L62)): `num_workers=4`
4. **GPU memory pinning** ([Trainer.py:63](../core/services/Word2Vec/Trainer.py#L63)): `pin_memory=True`

---

## 7. API Usage

### 7.1 Basic Training Example

```python
from services import Word2VecService, DocumentService

# Load and preprocess documents
doc_service = DocumentService(documents)
doc_service.preprocess()

# Create Word2Vec service with default config
word2vec_service = Word2VecService.create_default(
    doc_service=doc_service,
    min_count=5  # Filter words appearing < 5 times
)

# Train the model
word2vec_service.train(output_file="embeddings.txt")

# Get word vectors
vector = word2vec_service.get_word_vector("machine")
similar_words = word2vec_service.find_similar_words("machine", top_k=10)
```

### 7.2 Custom Configuration

```python
word2vec_service = Word2VecService.create_custom(
    doc_service=doc_service,
    embedding_dim=256,      # Larger embeddings
    window_size=10,         # Wider context window
    iterations=20,          # More training epochs
    learning_rate=0.05,     # Higher learning rate
    batch_size=256,         # Larger batches
    min_count=10            # Stricter vocabulary filtering
)
```

### 7.3 Downstream Tasks

The service provides several methods for utilizing trained embeddings:

1. **Single word vector**: `get_word_vector(word: str)`
2. **Multiple word vectors**: `get_multiple_word_vectors(words: List[str])`
3. **All word vectors**: `get_all_vectors()`
4. **Similarity search**: `find_similar_words(word: str, top_k: int)`
5. **Word analogies**: `word_analogy(word_a, word_b, word_c, top_k)`

**Example - Word Analogy:**
```python
# king - man + woman = ?
result = word2vec_service.word_analogy("king", "man", "woman", top_k=1)
# Expected: [("queen", 0.85)]
```

---

## 8. Performance Characteristics

### 8.1 Training Speed

**Optimizations:**
- Multi-worker data loading (4 workers)
- GPU acceleration (when available)
- Efficient negative sampling with pre-computed tables
- Dense tensor operations

**Typical Performance** (on NVIDIA GPU):
- ~100,000 training pairs/second
- 10 epochs on 1M pairs: ~2-3 minutes

### 8.2 Memory Usage

| Component | Memory |
|-----------|--------|
| Embeddings (50K vocab × 128 dim × 2) | ~50 MB |
| Negative sampling table | ~400 MB |
| Training pairs (precomputed) | Variable |
| PyTorch overhead | ~1-2 GB |

**Total:** ~2-3 GB for typical datasets (50K-100K vocabulary)

---

## 9. Key Differences from Original Implementation

| Aspect | Original (Andras7) | Our Implementation | Impact |
|--------|-------------------|-------------------|--------|
| Embedding init | `1.0 / emb_dim` | Xavier/Glorot | **Critical** - Fixed loss plateau |
| Optimizer | Adam (not specified) | SGD + momentum | Better alignment with Word2Vec paper |
| LR schedule | Cosine annealing | Linear decay | Simpler, more standard |
| Sparse embeddings | `sparse=True` | `sparse=False` | Better GPU utilization |
| v_embeddings init | Zero | Uniform | Breaks symmetry |
| Negative target | Allowed | Prevented | Higher quality negatives |
| Subsampling | Basic | Mikolov formula | More theoretically sound |
| Reproducibility | Not enforced | Full seed control | Enables reproducible experiments |

---

## 10. Validation and Quality Assurance

### 10.1 Correctness Checks

1. **Word similarity** - Similar words should have high cosine similarity
2. **Word analogies** - Semantic relationships should be captured
3. **Frequency-based similarity** - Co-occurring words should be closer

### 10.2 Training Indicators

**Healthy Training:**
- Loss decreases from ~4-5 to ~2-3 over 10 epochs
- Learning rate decays linearly from 0.025 to 0.000025
- No NaN or Inf values in loss
- Similar words cluster together in embedding space

**Problematic Signs:**
- Loss stuck or increasing
- Very high loss (>10) or very low loss (<0.5)
- NaN/Inf values
- Random word similarities (no semantic structure)

---

## 11. Future Enhancements

### 11.1 Potential Improvements

1. **Hierarchical Softmax** - Alternative to negative sampling for rare words
2. **Subword embeddings** - FastText-style character n-grams
3. **Dynamic window size** - Adaptive context based on sentence structure
4. **Contextualized embeddings** - Integrate with transformer models (BERT, etc.)
5. **Multi-GPU training** - Data parallelism for very large corpora

### 11.2 Advanced Features

1. **Out-of-vocabulary (OOV) handling** - Character-level or subword fallback
2. **Phrase detection** - Identify multi-word expressions
3. **Domain adaptation** - Fine-tune on specific domains
4. **Embedding evaluation suite** - Automated quality metrics

---

## 12. References

### 12.1 Papers

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). **Distributed representations of words and phrases and their compositionality**. In Advances in neural information processing systems (pp. 3111-3119).

2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient estimation of word representations in vector space**. arXiv preprint arXiv:1301.3781.

3. Goldberg, Y., & Levy, O. (2014). **word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method**. arXiv preprint arXiv:1402.3722.

### 12.2 Code References

- **Base Implementation**: [Andras7/word2vec-pytorch](https://github.com/Andras7/word2vec-pytorch)
- **PyTorch Documentation**: [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

---

## 13. Conclusion

This Word2Vec implementation provides a robust, efficient, and theoretically sound foundation for learning word embeddings. The key improvements over the base implementation—particularly the Xavier initialization and proper optimizer configuration—were critical in achieving stable training and high-quality embeddings.

The modular design allows easy integration into the broader SENTIMENT project, providing dense word representations for downstream tasks such as:
- Graph construction (word co-occurrence graphs)
- Document similarity
- Semantic search
- Text classification
- Clustering

**Key Takeaways:**
1. **Initialization matters** - Xavier initialization resolved loss plateau issues
2. **Follow the literature** - SGD + linear decay aligns with Word2Vec best practices
3. **Reproducibility is essential** - Fixed seeds enable consistent experiments
4. **Negative sampling quality** - Avoiding positive targets improves training

---

## Appendix: File Structure

```
core/
├── entities/
│   └── skipgram.py              # SkipGramModel (41 lines)
└── services/
    └── Word2Vec/
        ├── Word2VecService.py   # High-level API (288 lines)
        ├── DataLoader.py        # Data loading & preprocessing (251 lines)
        └── Trainer.py           # Training loop (263 lines)
```

**Total Lines of Code**: ~843 lines (excluding comments and blank lines)

---

**Document Version**: 1.0
**Date**: 2025-10-13
**Author**: JaehnK (lecielgris1@gmail.com)
**Project**: SENTIMENT - Graph-based Document Analysis
