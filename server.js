/**
 * server.js — Simple Q&A (versi JS ensemble), berdiri sendiri, TERPISAH
 * dari CodeMind.
 *
 * Cara cari jawaban sekarang 2 TAHAP (bukan digabung jadi 1 pencarian):
 *   Tahap 1: cek dulu di data hasil Auto-Learning (Supabase) —
 *            ini yang paling "personal", hasil belajar kamu sendiri
 *   Tahap 2: kalau nggak ketemu yang cukup yakin di situ, BARU cek
 *            ke 54 data statis bawaan (qa_data.json)
 *
 * Jalankan lokal:
 *   npm install
 *   npm start
 */

import express from 'express';
import cors from 'cors';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { pipeline } from '@huggingface/transformers';
import { createClient } from '@supabase/supabase-js';
import { buildTfidfModel, transformQuery, cosineSim } from './tfidf.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const staticQaData = JSON.parse(readFileSync(join(__dirname, 'qa_data.json'), 'utf-8'));

const MODEL_NAME = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
const CONFIDENCE_THRESHOLD = 0.4;
const EMBEDDING_WEIGHT = 0.8;
const TFIDF_WEIGHT = 0.2;
const SUPABASE_REFRESH_MS = 5 * 60 * 1000; // 5 menit

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
const supabase =
  supabaseUrl && supabaseServiceKey ? createClient(supabaseUrl, supabaseServiceKey) : null;

if (!supabase) {
  console.warn(
    '⚠️  NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY belum di-set — cuma bakal pakai qa_data.json statis, hasil Auto-Learning nggak kepake.'
  );
}

let extractorPromise = null;

// ── TAHAP 1: data hasil belajar (Supabase) — direfresh berkala
let learnedQuestions = [];
let learnedAnswers = [];
let learnedEmbeddings = [];
let learnedTfidfModel = null;
let lastSupabaseFetch = 0;

// ── TAHAP 2: data statis (qa_data.json) — dihitung sekali, nggak pernah berubah
let staticQuestions = staticQaData.map((p) => p.question);
let staticAnswers = staticQaData.map((p) => p.answer);
let staticEmbeddings = [];
let staticTfidfModel = buildTfidfModel(staticQuestions);
let staticReady = false;

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', MODEL_NAME, { dtype: 'q8' });
  }
  return extractorPromise;
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

async function fetchLearnedKnowledge() {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('learned_knowledge')
    .select('question, answer, embedding');
  if (error) {
    console.error('[fetchLearnedKnowledge]', error);
    return [];
  }
  return (data ?? []).map((row) => ({
    question: row.question,
    answer: row.answer,
    embedding: Array.isArray(row.embedding) ? row.embedding : JSON.parse(row.embedding),
  }));
}

async function refreshLearnedDataset() {
  const learned = await fetchLearnedKnowledge();
  learnedQuestions = learned.map((p) => p.question);
  learnedAnswers = learned.map((p) => p.answer);
  learnedEmbeddings = learned.map((p) => p.embedding);
  learnedTfidfModel = learnedQuestions.length > 0 ? buildTfidfModel(learnedQuestions) : null;
  lastSupabaseFetch = Date.now();
  console.log(`📚 Data hasil belajar di-refresh: ${learnedQuestions.length} entri dari Supabase`);
}

async function ensureStaticEmbeddings() {
  if (staticReady) return;
  const extractor = await getExtractor();
  const output = await extractor(staticQuestions, { pooling: 'mean', normalize: true });
  staticEmbeddings = output.tolist();
  staticReady = true;
}

async function ensureFreshLearnedDataset() {
  if (Date.now() - lastSupabaseFetch > SUPABASE_REFRESH_MS) {
    await refreshLearnedDataset();
  }
}

// Cari kecocokan terbaik di 1 kumpulan data — dipakai buat tahap 1 dan
// tahap 2 secara terpisah, masing-masing dengan TF-IDF model-nya sendiri.
function findBestMatch(userEmbedding, userQuestion, tfidfModel, embeddings, questions, answers) {
  if (questions.length === 0) return null;

  const userTfidfVector = transformQuery(tfidfModel, userQuestion);

  let bestIdx = 0;
  let bestCombined = -Infinity;
  let bestEmbeddingScore = 0;
  let bestTfidfScore = 0;

  for (let i = 0; i < questions.length; i++) {
    const embeddingScore = dotProduct(userEmbedding, embeddings[i]);
    const tfidfScore = cosineSim(userTfidfVector, tfidfModel.documentVectors[i]);
    const combined = EMBEDDING_WEIGHT * embeddingScore + TFIDF_WEIGHT * tfidfScore;
    if (combined > bestCombined) {
      bestCombined = combined;
      bestIdx = i;
      bestEmbeddingScore = embeddingScore;
      bestTfidfScore = tfidfScore;
    }
  }

  return {
    answer: answers[bestIdx],
    confidence: bestCombined,
    matched_question: questions[bestIdx],
    embedding_score: bestEmbeddingScore,
    tfidf_score: bestTfidfScore,
  };
}

async function matchQuestion(question) {
  await ensureFreshLearnedDataset();
  await ensureStaticEmbeddings();

  const extractor = await getExtractor();
  const output = await extractor([question], { pooling: 'mean', normalize: true });
  const userEmbedding = output.tolist()[0];

  // TAHAP 1 — cek hasil belajar (Supabase) dulu
  if (learnedTfidfModel) {
    const learnedMatch = findBestMatch(
      userEmbedding,
      question,
      learnedTfidfModel,
      learnedEmbeddings,
      learnedQuestions,
      learnedAnswers
    );
    if (learnedMatch && learnedMatch.confidence >= CONFIDENCE_THRESHOLD) {
      return { ...learnedMatch, source: 'learned' };
    }
  }

  // TAHAP 2 — nggak ketemu (atau nggak cukup yakin) di hasil belajar,
  // coba cek ke data statis bawaan.
  const staticMatch = findBestMatch(
    userEmbedding,
    question,
    staticTfidfModel,
    staticEmbeddings,
    staticQuestions,
    staticAnswers
  );
  if (staticMatch && staticMatch.confidence >= CONFIDENCE_THRESHOLD) {
    return { ...staticMatch, source: 'static' };
  }

  // Nggak ketemu di dua-duanya.
  return {
    answer: 'Maaf, aku belum pernah belajar soal itu.',
    confidence: staticMatch ? staticMatch.confidence : 0,
    matched_question: '',
    embedding_score: staticMatch ? staticMatch.embedding_score : 0,
    tfidf_score: staticMatch ? staticMatch.tfidf_score : 0,
    source: 'none',
  };
}

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    staticEntries: staticQuestions.length,
    learnedEntries: learnedQuestions.length,
  });
});

app.post('/ask', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: 'question is required' });
    }
    const result = await matchQuestion(question);
    res.json(result);
  } catch (err) {
    console.error('[/ask]', err);
    res.status(500).json({ error: err.message || 'Unknown error' });
  }
});

app.post('/embed', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }
    const extractor = await getExtractor();
    const output = await extractor([text], { pooling: 'mean', normalize: true });
    const embedding = output.tolist()[0];
    res.json({ embedding });
  } catch (err) {
    console.error('[/embed]', err);
    res.status(500).json({ error: err.message || 'Unknown error' });
  }
});

app.post('/refresh', async (req, res) => {
  try {
    await refreshLearnedDataset();
    res.json({ status: 'ok', learnedEntries: learnedQuestions.length });
  } catch (err) {
    console.error('[/refresh]', err);
    res.status(500).json({ error: err.message || 'Unknown error' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  console.log(`✅ Simple Q&A JS API jalan di port ${PORT}`);
  try {
    await refreshLearnedDataset();
    await ensureStaticEmbeddings();
  } catch (err) {
    console.error('Gagal load dataset awal:', err);
  }
});
