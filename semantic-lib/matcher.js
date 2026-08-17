/**
 * matcher.js
 * Main semantic matcher class - combines embeddings + TF-IDF
 */

import { EmbeddingExtractor } from './embeddings.js';
import { SimilarityCalculator, dotProduct } from './similarity.js';
import { buildTfidfModel, transformQuery, tfidfCosineSim } from './tfidf.js';

export class SemanticMatcher {
  /**
   * @param {object[]} qaData - Array of {question, answer} objects
   * @param {object} options - {modelName, embeddingWeight, tfidfWeight, threshold}
   */
  constructor(qaData = [], options = {}) {
    this.qaData = qaData;
    this.questions = qaData.map((p) => p.question);
    this.answers = qaData.map((p) => p.answer);

    // Options dengan default values
    this.modelName = options.modelName || 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
    this.embeddingWeight = options.embeddingWeight ?? 0.8;
    this.tfidfWeight = options.tfidfWeight ?? 0.2;
    this.confidenceThreshold = options.threshold ?? 0.4;

    // Initialize components
    this.extractor = new EmbeddingExtractor(this.modelName);
    this.embeddings = [];
    this.tfidfModel = null;
    this.initialized = false;
  }

  /**
   * Initialize embeddings dan TF-IDF model
   */
  async initialize() {
    if (this.initialized) return;

    console.log(`🚀 Initializing SemanticMatcher...`);
    console.log(`   Questions: ${this.questions.length}`);
    console.log(`   Embedding weight: ${this.embeddingWeight}`);
    console.log(`   TF-IDF weight: ${this.tfidfWeight}`);

    // Build TF-IDF model
    this.tfidfModel = buildTfidfModel(this.questions);
    console.log(`✅ TF-IDF model built: ${this.tfidfModel.vocabulary.length} terms`);

    // Extract embeddings
    if (this.questions.length > 0) {
      console.log(`⏳ Extracting embeddings for ${this.questions.length} questions...`);
      this.embeddings = await this.extractor.extract(this.questions);
      console.log(`✅ Embeddings extracted: ${this.embeddings.length} vectors`);
    }

    this.initialized = true;
    console.log(`✅ SemanticMatcher ready!`);
  }

  /**
   * Add new Q&A pair
   */
  async addQAPair(question, answer) {
    this.questions.push(question);
    this.answers.push(answer);
    this.qaData.push({ question, answer });

    // Rebuild models
    this.tfidfModel = buildTfidfModel(this.questions);
    this.embeddings = await this.extractor.extract(this.questions);

    console.log(`✅ New Q&A pair added. Total: ${this.questions.length}`);
  }

  /**
   * Match question dan return best answer
   */
  async match(userQuestion, returnDetails = false) {
    if (!this.initialized) {
      throw new Error('SemanticMatcher not initialized. Call initialize() first.');
    }

    if (this.questions.length === 0) {
      return {
        answer: 'Maaf, belum ada data yang dikenali.',
        confidence: 0,
        matched_question: '',
        source: 'none',
        ...(returnDetails && {
          embedding_score: 0,
          tfidf_score: 0,
          details: null
        })
      };
    }

    // Extract user question embedding
    const userEmbedding = await this.extractor.extractSingle(userQuestion);

    // Find best match
    const result = this._findBestMatch(userEmbedding, userQuestion);

    if (!returnDetails) {
      return result;
    }

    return {
      ...result,
      details: {
        total_questions: this.questions.length,
        threshold: this.confidenceThreshold,
        weights: {
          embedding: this.embeddingWeight,
          tfidf: this.tfidfWeight
        }
      }
    };
  }

  /**
   * Find top-K similar questions
   */
  async findTopK(userQuestion, topK = 5) {
    if (!this.initialized) {
      throw new Error('SemanticMatcher not initialized. Call initialize() first.');
    }

    const userEmbedding = await this.extractor.extractSingle(userQuestion);
    const userTfidfVector = transformQuery(this.tfidfModel, userQuestion);

    const scores = this.questions.map((question, i) => {
      const embeddingScore = dotProduct(userEmbedding, this.embeddings[i]);
      const tfidfScore = tfidfCosineSim(userTfidfVector, this.tfidfModel.documentVectors[i]);
      const combined = this.embeddingWeight * embeddingScore + this.tfidfWeight * tfidfScore;

      return {
        index: i,
        question: question,
        answer: this.answers[i],
        score: combined,
        embedding_score: embeddingScore,
        tfidf_score: tfidfScore
      };
    });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Internal: find best match
   */
  _findBestMatch(userEmbedding, userQuestion) {
    const userTfidfVector = transformQuery(this.tfidfModel, userQuestion);

    let bestIdx = 0;
    let bestCombined = -Infinity;
    let bestEmbeddingScore = 0;
    let bestTfidfScore = 0;

    for (let i = 0; i < this.questions.length; i++) {
      const embeddingScore = dotProduct(userEmbedding, this.embeddings[i]);
      const tfidfScore = tfidfCosineSim(userTfidfVector, this.tfidfModel.documentVectors[i]);
      const combined = this.embeddingWeight * embeddingScore + this.tfidfWeight * tfidfScore;

      if (combined > bestCombined) {
        bestCombined = combined;
        bestIdx = i;
        bestEmbeddingScore = embeddingScore;
        bestTfidfScore = tfidfScore;
      }
    }

    const isConfident = bestCombined >= this.confidenceThreshold;

    return {
      answer: isConfident ? this.answers[bestIdx] : 'Maaf, aku nggak yakin dengan jawaban itu.',
      confidence: bestCombined,
      matched_question: isConfident ? this.questions[bestIdx] : '',
      embedding_score: bestEmbeddingScore,
      tfidf_score: bestTfidfScore,
      source: isConfident ? 'matched' : 'low_confidence'
    };
  }

  /**
   * Get matcher stats
   */
  getStats() {
    return {
      initialized: this.initialized,
      total_questions: this.questions.length,
      model: this.modelName,
      weights: {
        embedding: this.embeddingWeight,
        tfidf: this.tfidfWeight
      },
      threshold: this.confidenceThreshold,
      tfidf_vocab_size: this.tfidfModel ? this.tfidfModel.vocabulary.length : 0
    };
  }
}
