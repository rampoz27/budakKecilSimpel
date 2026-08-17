/**
 * examples.js
 * Example usage of semantic-matcher-lib
 */

import { createMatcher, SemanticMatcher } from './index.js';

// Example 1: Quick start dengan createMatcher
async function example1_quickStart() {
  console.log('\n=== Example 1: Quick Start ===\n');

  const qaData = [
    { question: 'siapa nama kamu', answer: 'Aku adalah AI Q&A sederhana!' },
    { question: 'apa itu python', answer: 'Python adalah bahasa pemrograman yang mudah dibaca...' },
    { question: 'apa itu javascript', answer: 'JavaScript adalah bahasa pemrograman utama untuk web...' },
    { question: 'apa itu machine learning', answer: 'Machine learning adalah cara komputer belajar pola dari data...' },
    { question: 'apa itu deep learning', answer: 'Deep learning adalah cabang ML yang pakai neural network berlapis-lapis...' }
  ];

  const matcher = await createMatcher(qaData);

  const questions = [
    'siapa kamu',
    'bagaimana cara belajar AI',
    'python buat apa',
    'apa bedanya ML sama DL'
  ];

  for (const q of questions) {
    const result = await matcher.match(q);
    console.log(`❓ User: ${q}`);
    console.log(`📝 Bot: ${result.answer}`);
    console.log(`🎯 Confidence: ${result.confidence.toFixed(3)} (matched: ${result.matched_question})\n`);
  }
}

// Example 2: Top-K matching
async function example2_topK() {
  console.log('\n=== Example 2: Top-K Matching ===\n');

  const qaData = [
    { question: 'apa itu AI', answer: 'AI adalah kecerdasan buatan...' },
    { question: 'apa itu machine learning', answer: 'ML adalah cara belajar dari data...' },
    { question: 'apa itu deep learning', answer: 'DL adalah ML dengan neural networks...' },
    { question: 'apa itu neural network', answer: 'Neural networks terinspirasi dari otak...' },
    { question: 'apa itu NLP', answer: 'NLP adalah processing bahasa natural...' }
  ];

  const matcher = new SemanticMatcher(qaData, {
    embeddingWeight: 0.8,
    tfidfWeight: 0.2,
    threshold: 0.3
  });

  await matcher.initialize();

  const question = 'bagaimana cara belajar machine learning';
  console.log(`❓ User: ${question}\n`);

  const topMatches = await matcher.findTopK(question, 3);

  topMatches.forEach((match, idx) => {
    console.log(`${idx + 1}. ${match.question}`);
    console.log(`   Score: ${match.score.toFixed(3)} (embedding: ${match.embedding_score.toFixed(3)}, tfidf: ${match.tfidf_score.toFixed(3)})`);
    console.log(`   Answer: ${match.answer.substring(0, 50)}...\n`);
  });
}

// Example 3: Dynamic learning
async function example3_dynamicLearning() {
  console.log('\n=== Example 3: Dynamic Learning (Add New Q&A) ===\n');

  const qaData = [
    { question: 'apa itu python', answer: 'Python adalah bahasa pemrograman...' },
    { question: 'apa itu javascript', answer: 'JavaScript adalah bahasa web...' }
  ];

  const matcher = new SemanticMatcher(qaData);
  await matcher.initialize();

  console.log(`📊 Stats before: ${matcher.getStats().total_questions} questions`);

  // Add new Q&A
  console.log('➕ Adding new Q&A pair...');
  await matcher.addQAPair('apa itu rust', 'Rust adalah bahasa sistem yang aman dan cepat...');
  await matcher.addQAPair('apa itu golang', 'Go (Golang) adalah bahasa pemrograman modern dari Google...');

  console.log(`📊 Stats after: ${matcher.getStats().total_questions} questions\n`);

  // Test dengan pertanyaan baru
  const result = await matcher.match('bagaimana dengan bahasa rust');
  console.log(`❓ User: bagaimana dengan bahasa rust`);
  console.log(`📝 Bot: ${result.answer}`);
  console.log(`🎯 Confidence: ${result.confidence.toFixed(3)}\n`);
}

// Example 4: Custom configuration
async function example4_customConfig() {
  console.log('\n=== Example 4: Custom Configuration ===\n');

  const qaData = [
    { question: 'apa itu AI', answer: 'AI adalah kecerdasan buatan...' },
    { question: 'machine learning itu apa', answer: 'ML adalah pembelajaran mesin...' },
    { question: 'deep learning itu apa', answer: 'DL adalah pembelajaran mendalam...' }
  ];

  // Config 1: Semantic-heavy (lebih paham makna)
  const matcher1 = new SemanticMatcher(qaData, {
    embeddingWeight: 0.95,
    tfidfWeight: 0.05,
    threshold: 0.5,
    modelName: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
  });

  // Config 2: Keyword-heavy (lebih paham kata kunci)
  const matcher2 = new SemanticMatcher(qaData, {
    embeddingWeight: 0.5,
    tfidfWeight: 0.5,
    threshold: 0.4
  });

  await matcher1.initialize();
  await matcher2.initialize();

  const question = 'apa itu pembelajaran dalam';

  console.log(`❓ Question: ${question}\n`);

  const result1 = await matcher1.match(question);
  console.log(`🧠 Semantic-heavy (0.95 embedding, 0.05 tfidf):`);
  console.log(`   Confidence: ${result1.confidence.toFixed(3)}`);
  console.log(`   Matched: ${result1.matched_question}\n`);

  const result2 = await matcher2.match(question);
  console.log(`📚 Keyword-heavy (0.5 embedding, 0.5 tfidf):`);
  console.log(`   Confidence: ${result2.confidence.toFixed(3)}`);
  console.log(`   Matched: ${result2.matched_question}\n`);
}

// Example 5: Get detailed stats
async function example5_stats() {
  console.log('\n=== Example 5: Detailed Statistics ===\n');

  const qaData = [
    { question: 'apa itu python', answer: 'Python...' },
    { question: 'apa itu javascript', answer: 'JavaScript...' },
    { question: 'apa itu rust', answer: 'Rust...' }
  ];

  const matcher = new SemanticMatcher(qaData);
  await matcher.initialize();

  const stats = matcher.getStats();
  console.log('📊 Matcher Statistics:');
  console.log(JSON.stringify(stats, null, 2));

  const result = await matcher.match('python untuk apa', true);
  console.log('\n📊 Match Result dengan Details:');
  console.log(JSON.stringify(result, null, 2));
}

// Run all examples
async function runAllExamples() {
  try {
    await example1_quickStart();
    await example2_topK();
    await example3_dynamicLearning();
    await example4_customConfig();
    await example5_stats();

    console.log('\n✅ All examples completed!');
  } catch (error) {
    console.error('❌ Error:', error);
  }
}

// Uncomment untuk jalanin semua examples
// runAllExamples();

export { example1_quickStart, example2_topK, example3_dynamicLearning, example4_customConfig, example5_stats };
