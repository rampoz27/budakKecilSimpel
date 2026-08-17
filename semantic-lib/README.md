# Semantic Matcher Library

Reusable semantic matching library dengan transformer embeddings + TF-IDF hybrid approach.

## Installation

```bash
npm install semantic-matcher-lib
```

## Quick Start

```javascript
import { createMatcher } from 'semantic-matcher-lib';

const qaData = [
  { question: 'apa itu AI', answer: 'AI adalah kecerdasan buatan...' },
  { question: 'apa itu machine learning', answer: 'Machine learning adalah...' }
];

// Create dan initialize matcher
const matcher = await createMatcher(qaData);

// Match question
const result = await matcher.match('bagaimana cara belajar AI');
console.log(result.answer);
console.log(`Confidence: ${result.confidence}`);
```

## API

### SemanticMatcher

```javascript
import { SemanticMatcher } from 'semantic-matcher-lib';

const matcher = new SemanticMatcher(qaData, {
  modelName: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
  embeddingWeight: 0.8,
  tfidfWeight: 0.2,
  threshold: 0.4
});

await matcher.initialize();
```

### Methods

#### `match(question, returnDetails = false)`
Find best answer untuk question.

```javascript
const result = await matcher.match('apa itu python');
// {
//   answer: 'Python adalah bahasa pemrograman...',
//   confidence: 0.87,
//   matched_question: 'apa itu python',
//   embedding_score: 0.92,
//   tfidf_score: 0.78,
//   source: 'matched'
// }
```

#### `findTopK(question, topK = 5)`
Find top-K similar questions.

```javascript
const topMatches = await matcher.findTopK('apa itu AI', 3);
// [
//   { question: 'apa itu AI', answer: '...', score: 0.95, ... },
//   { question: 'apa itu machine learning', answer: '...', score: 0.78, ... },
//   { question: 'apa itu deep learning', answer: '...', score: 0.72, ... }
// ]
```

#### `addQAPair(question, answer)`
Add new Q&A pair dynamically.

```javascript
await matcher.addQAPair('apa itu nodejs', 'Node.js adalah runtime JavaScript...');
```

#### `getStats()`
Get matcher statistics.

```javascript
const stats = matcher.getStats();
// {
//   initialized: true,
//   total_questions: 54,
//   model: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
//   weights: { embedding: 0.8, tfidf: 0.2 },
//   threshold: 0.4,
//   tfidf_vocab_size: 1245
// }
```

### EmbeddingExtractor

```javascript
import { EmbeddingExtractor } from 'semantic-matcher-lib/embeddings';

const extractor = new EmbeddingExtractor();
await extractor.initialize();

const embedding = await extractor.extractSingle('apa itu AI');
const embeddings = await extractor.extract(['apa itu AI', 'apa itu ML']);
```

### Similarity Functions

```javascript
import { 
  cosineSimilarity, 
  dotProduct, 
  SimilarityCalculator 
} from 'semantic-matcher-lib/similarity';

const similarity = cosineSimilarity(embedding1, embedding2);
const distance = euclideanDistance(embedding1, embedding2);

const topMatches = SimilarityCalculator.findMostSimilar(
  queryEmbedding, 
  documentEmbeddings, 
  { topK: 5 }
);
```

### TF-IDF Functions

```javascript
import { 
  buildTfidfModel, 
  transformQuery, 
  tfidfCosineSim 
} from 'semantic-matcher-lib/tfidf';

const model = buildTfidfModel(['apa itu AI', 'apa itu ML']);
const queryVector = transformQuery(model, 'AI adalah');
const similarity = tfidfCosineSim(queryVector, model.documentVectors[0]);
```

## Configuration

### Model Selection

```javascript
// Balanced (default) - 384 dims, fast
'Xenova/paraphrase-multilingual-MiniLM-L12-v2'

// Better accuracy - 384 dims, slightly slower
'Xenova/all-MiniLM-L12-v2'

// Best accuracy - 768 dims, slower
'Xenova/all-mpnet-base-v2'
```

### Weight Tuning

```javascript
const matcher = new SemanticMatcher(qaData, {
  embeddingWeight: 0.9,  // Lebih rely on semantic similarity
  tfidfWeight: 0.1       // Kurang rely on keyword matching
});
```

## Performance Tips

1. **Caching**: Simpan embeddings di database jika dataset besar
2. **Batch Processing**: Gunakan `extractBatch()` untuk multiple texts
3. **Lazy Loading**: Model diload saat `initialize()` pertama kali dipanggil
4. **Threshold Tuning**: Adjust `threshold` sesuai use case

## License

MIT
