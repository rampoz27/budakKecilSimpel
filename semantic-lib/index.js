/**
 * semantic-matcher-lib
 * Main export untuk semantic matching dengan transformer models
 */

export { SemanticMatcher } from './matcher.js';
export { EmbeddingExtractor } from './embeddings.js';
export { SimilarityCalculator, cosineSimilarity, dotProduct } from './similarity.js';
export { buildTfidfModel, transformQuery, tfidfCosineSim } from './tfidf.js';

// Default exported function untuk quick start
export async function createMatcher(qaData) {
  const matcher = new SemanticMatcher(qaData);
  await matcher.initialize();
  return matcher;
}
