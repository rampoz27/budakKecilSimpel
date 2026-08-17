/**
 * embeddings.js
 * Semantic embedding extractor menggunakan transformer models
 */

import { pipeline } from '@huggingface/transformers';

export class EmbeddingExtractor {
  constructor(modelName = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2') {
    this.modelName = modelName;
    this.extractor = null;
    this.initialized = false;
  }

  /**
   * Initialize transformer model (lazy load)
   */
  async initialize() {
    if (!this.initialized) {
      console.log(`🔄 Loading transformer model: ${this.modelName}...`);
      this.extractor = await pipeline('feature-extraction', this.modelName, { dtype: 'q8' });
      this.initialized = true;
      console.log(`✅ Model loaded successfully`);
    }
  }

  /**
   * Extract embedding untuk satu atau multiple teks
   * @param {string|string[]} texts - Text atau array of texts
   * @returns {Promise<number[][]>} Array of embeddings
   */
  async extract(texts) {
    await this.initialize();
    
    const input = Array.isArray(texts) ? texts : [texts];
    const output = await this.extractor(input, { pooling: 'mean', normalize: true });
    const embeddings = output.tolist();
    
    return Array.isArray(texts) ? embeddings : embeddings[0];
  }

  /**
   * Extract single text embedding
   */
  async extractSingle(text) {
    await this.initialize();
    const output = await this.extractor([text], { pooling: 'mean', normalize: true });
    return output.tolist()[0];
  }

  /**
   * Extract batch embeddings dengan caching
   */
  async extractBatch(texts, useCache = false) {
    if (!useCache) {
      return this.extract(texts);
    }

    const cache = new Map();
    const toFetch = [];
    const indices = [];

    for (let i = 0; i < texts.length; i++) {
      if (cache.has(texts[i])) {
        // Skip, sudah di cache
      } else {
        toFetch.push(texts[i]);
        indices.push(i);
      }
    }

    if (toFetch.length > 0) {
      const fetched = await this.extract(toFetch);
      toFetch.forEach((text, i) => {
        cache.set(text, fetched[i]);
      });
    }

    return texts.map(text => cache.get(text));
  }

  /**
   * Get model info
   */
  getModelInfo() {
    return {
      name: this.modelName,
      initialized: this.initialized,
      type: 'multilingual-embedding'
    };
  }
}
