/**
 * server.js — Simple Q&A (versi JS ensemble), berdiri sendiri, TERPISAH
 * dari CodeMind. Ini yang bikin CodeMind aman — kalau model ini butuh
 * banyak RAM, itu cuma nge-crash SERVICE INI doang, nggak ikut nyeret
 * CodeMind.
 *
 * Sekarang juga narik data dari Supabase (tabel learned_knowledge) —
 * hasil dari fitur Auto-Learning — digabung sama qa_data.json statis,
 * biar hal-hal baru yang "dipelajari" beneran kepake buat jawab
 * pertanyaan, bukan cuma nyimpen doang tanpa pernah dibaca lagi.
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

// Berapa lama data dari Supabase di-cache sebelum ditarik ulang — biar
// nggak nge-query Supabase di SETIAP request, tapi tetap "sadar" kalau
// ada hal baru yang dipelajari nggak lama sebelumnya.
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
let tfidfModel = null;

// Dataset gabungan yang beneran dipakai buat matching — combinedQuestions[i]
// selaras index-nya sama combinedAnswers[i] dan combinedEmbeddings[i].
let combinedQuestions = [];
let combinedAnswers = [];
let combinedEmbeddings = [];
let lastSupabaseFetch = 0;

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

// Tarik semua learned_knowledge dari Supabase — pgvector embedding-nya
// UDAH kehitung pas disimpen (lewat endpoint /embed yang sama), jadi
// nggak perlu dihitung ulang di sini, tinggal dipakai langsung.
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
    // Supabase balikin pgvector sebagai string "[0.1,0.2,...]" di
    // beberapa versi client — parse dulu biar aman dua-duanya.
    embedding: Array.isArray(row.embedding) ? row.embedding : JSON.parse(row.embedding),
  }));
}

// Bangun ulang dataset gabungan (statis + Supabase) dan TF-IDF model-nya.
// Dipanggil pas startup, dan otomatis diulang tiap SUPABASE_REFRESH_MS.
async function refreshDataset() {
  const learned = await fetchLearnedKnowledge();

  combinedQuestions = [...staticQaData.map((p) => p.question), ...learned.map((p) => p.question)];
  combinedAnswers = [...staticQaData.map((p) => p.answer), ...learned.map((p) => p.answer)];

  const extractor = await getExtractor();
  const staticOutput = await extractor(
    staticQaData.map((p) => p.question),
    { pooling: 'mean', normalize: true }
  );
  const staticEmbeddings = staticOutput.tolist();

  combinedEmbeddings = [...staticEmbeddings, ...learned.map((p) => p.embedding)];

  // TF-IDF butuh SEMUA dokumen buat ngitung IDF yang bener, jadi harus
  // dibangun ulang tiap kali dataset-nya berubah.
  tfidfModel = buildTfidfModel(combinedQuestions);

  lastSupabaseFetch = Date.now();
  console.log(
    `📚 Dataset di-refresh: ${staticQaData.length} statis + ${learned.length} dari Supabase = ${combinedQuestions.length} total`
  );
}

async function ensureFreshDataset() {
  if (Date.now() - lastSupabaseFetch > SUPABASE_REFRESH_MS) {
    await refreshDataset();
  }
}

async function matchQuestion(question) {
  await ensureFreshDataset();

  const extractor = await getExtractor();
  const output = await extractor([question], { pooling: 'mean', normalize: true });
  const userEmbedding = output.tolist()[0];

  const userTfidfVector = transformQuery(tfidfModel, question);

  let bestIdx = 0;
  let bestCombined = -Infinity;
  let bestEmbedding = 0;
  let bestTfidf = 0;

  for (let i = 0; i < combinedQuestions.length; i++) {
    const embeddingScore = dotProduct(userEmbedding, combinedEmbeddings[i]);
    const tfidfScore = cosineSim(userTfidfVector, tfidfModel.documentVectors[i]);
    const combined = EMBEDDING_WEIGHT * embeddingScore + TFIDF_WEIGHT * tfidfScore;
    if (combined > bestCombined) {
      bestCombined = combined;
      bestIdx = i;
      bestEmbedding = embeddingScore;
      bestTfidf = tfidfScore;
    }
  }

  if (bestCombined < CONFIDENCE_THRESHOLD) {
    return {
      answer: 'Maaf, aku belum pernah belajar soal itu.',
      confidence: bestCombined,
      matched_question: '',
      embedding_score: bestEmbedding,
      tfidf_score: bestTfidf,
    };
  }

  return {
    answer: combinedAnswers[bestIdx],
    confidence: bestCombined,
    matched_question: combinedQuestions[bestIdx],
    embedding_score: bestEmbedding,
    tfidf_score: bestTfidf,
  };
}

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', totalKnowledge: combinedQuestions.length });
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

// Dipakai fitur Auto-Learning di CodeMind — ngasih balik embedding
// mentah (array angka) buat teks yang dikasih.
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

// Endpoint manual buat maksa refresh dataset SEKARANG (nggak nunggu 5
// menit) — berguna abis selesai proses belajar baru.
app.post('/refresh', async (req, res) => {
  try {
    await refreshDataset();
    res.json({ status: 'ok', totalKnowledge: combinedQuestions.length });
  } catch (err) {
    console.error('[/refresh]', err);
    res.status(500).json({ error: err.message || 'Unknown error' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  console.log(`✅ Simple Q&A JS API jalan di port ${PORT}`);
  try {
    await refreshDataset();
  } catch (err) {
    console.error('Gagal load dataset awal:', err);
  }
});
