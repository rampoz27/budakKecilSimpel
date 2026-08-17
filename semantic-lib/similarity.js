/**
 * similarity.js
 * Similarity calculation functions (dot product, cosine, etc)
 */

/**
 * Dot product antara dua vector
 */
export function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Cosine similarity (sudah normalized, cukup dot product)
 */
export function cosineSimilarity(a, b) {
  return dotProduct(a, b);
}

/**
 * Euclidean distance antara dua vector
 */
export function euclideanDistance(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let sumSquares = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSquares += diff * diff;
  }
  return Math.sqrt(sumSquares);
}

/**
 * Normalize vector (L2 normalization)
 */
export function normalizeVector(v) {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return norm === 0 ? v : v.map((x) => x / norm);
}

/**
 * Helper class untuk similarity calculations
 */
export class SimilarityCalculator {
  /**
   * Find most similar item dari array
   * @param {number[]} queryVector - Query embedding
   * @param {number[][]} vectors - Array of embeddings
   * @param {object} options - {metric: 'cosine'|'euclidean', topK: 1}
   * @returns {object[]} [{index, score, similarity}, ...]
   */
  static findMostSimilar(queryVector, vectors, options = {}) {
    const { metric = 'cosine', topK = 1 } = options;
    const similarityFn = metric === 'cosine' ? cosineSimilarity : (a, b) => -euclideanDistance(a, b);

    const scores = vectors.map((vec, idx) => ({
      index: idx,
      similarity: similarityFn(queryVector, vec),
    }));

    return scores
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  /**
   * Batch similarity antara query vectors dan document vectors
   */
  static batchSimilarity(queryVectors, documentVectors, metric = 'cosine') {
    const similarityFn = metric === 'cosine' ? cosineSimilarity : (a, b) => -euclideanDistance(a, b);

    return queryVectors.map(queryVec =>
      documentVectors.map(docVec => similarityFn(queryVec, docVec))
    );
  }

  /**
   * Find threshold-based matches
   */
  static findByThreshold(queryVector, vectors, threshold = 0.5) {
    const similarities = vectors.map((vec, idx) => ({
      index: idx,
      similarity: cosineSimilarity(queryVector, vec),
    }));

    return similarities.filter(item => item.similarity >= threshold);
  }
}
