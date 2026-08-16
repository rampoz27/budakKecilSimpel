# Simple Q&A JS (Ensemble)

Model Q&A berbasis JavaScript murni — gabungan TF-IDF (port dari versi
Python) + Transformers.js (embedding semantik). Berdiri sendiri,
TERPISAH dari CodeMind — biar kalau butuh banyak RAM, itu nggak nyeret
CodeMind ikut down.

## Setup

```bash
npm install
```

## Jalanin lokal

```bash
npm start
```

Buka `http://localhost:3000/health` — harusnya balikin `{"status":"ok"}`.

## Endpoint

### `POST /ask`
```json
// Request
{ "question": "apa itu machine learning" }

// Response
{
  "answer": "Machine learning adalah cara komputer 'belajar' pola dari data...",
  "confidence": 0.87,
  "matched_question": "apa itu machine learning",
  "embedding_score": 0.92,
  "tfidf_score": 0.78
}
```

### `GET /health`
Cek servicenya nyala atau nggak.

## Deploy ke Render

- **Environment**: Node
- **Build Command**: `npm install`
- **Start Command**: `npm start`
- **Instance Type**: Free

Ini service **TERPISAH** dari CodeMind — bikin Web Service baru di
Render, jangan pakai service yang sama kayak CodeMind.

## Struktur file

```
package.json
server.js       ← Express server + logic matching
tfidf.js        ← TF-IDF murni JS, port dari Python
qa_data.json    ← 54 pasangan tanya-jawab
```
