/**
 * server.js — Simple Q&A (versi JS ensemble), berdiri sendiri, TERPISAH
 * dari CodeMind. Ini yang bikin CodeMind aman — kalau model ini butuh
 * banyak RAM, itu cuma nge-crash SERVICE INI doang, nggak ikut nyeret
 * CodeMind (beda dari sebelumnya waktu nempel langsung, dan bikin
 * seluruh CodeMind kena OOM/status 137).
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
import { buildTfidfModel, transformQuery, cosineSim } from './tfidf.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const qaData = JSON.parse(readFileSync(join(__dirname, 'qa_data.json'), 'utf-8'));

const MODEL_NAME = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
const CONFIDENCE_THRESHOLD = 0.4;
const EMBEDDING_WEIGHT = 0.6;
const TFIDF_WEIGHT = 0.4;

const questions = qaData.map((p) => p.question);
const answers = qaData.map((p) => p.answer);

let extractorPromise = null;
let questionEmbeddingsPromise = null;
let tfidfModel = null;

function getExtractor() {
  if (!extractorPromise) {
    // dtype: 'q8' = versi model yang di-kompres jadi 8-bit (dari default
    // 32-bit). Ukurannya di memori bisa turun signifikan — ini upaya buat
    // muat di 512MB Render. Kalau model ini nggak punya varian q8 yang
    // tersedia di Hugging Face, baris ini bakal ngasih error yang beda
    // (bukan OOM lagi) — itu tandanya perlu coba dtype lain atau nyerah
    // ke opsi lokal-doang.
    extractorPromise = pipeline('feature-extraction', MODEL_NAME, { dtype: 'q8' });
  }
  return extractorPromise;
}

async function getQuestionEmbeddings() {
  if (!questionEmbeddingsPromise) {
    questionEmbeddingsPromise = (async () => {
      const extractor = await getExtractor();
      const output = await extractor(questions, { pooling: 'mean', normalize: true });
      return output.tolist();
    })();
  }
  return questionEmbeddingsPromise;
}

function getTfidfModel() {
  if (!tfidfModel) {
    tfidfModel = buildTfidfModel(questions);
  }
  return tfidfModel;
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

async function matchQuestion(question) {
  const extractor = await getExtractor();
  const questionEmbeddings = await getQuestionEmbeddings();
  const output = await extractor([question], { pooling: 'mean', normalize: true });
  const userEmbedding = output.tolist()[0];

  const tfidf = getTfidfModel();
  const userTfidfVector = transformQuery(tfidf, question);

  let bestIdx = 0;
  let bestCombined = -Infinity;
  let bestEmbedding = 0;
  let bestTfidf = 0;

  for (let i = 0; i < questions.length; i++) {
    const embeddingScore = dotProduct(userEmbedding, questionEmbeddings[i]);
    const tfidfScore = cosineSim(userTfidfVector, tfidf.documentVectors[i]);
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
    answer: answers[bestIdx],
    confidence: bestCombined,
    matched_question: questions[bestIdx],
    embedding_score: bestEmbedding,
    tfidf_score: bestTfidf,
  };
}

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Simple Q&A JS API jalan di port ${PORT}`);
});
