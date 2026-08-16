/**
 * TF-IDF, sama persis logic-nya kayak versi Python (scikit-learn's
 * TfidfVectorizer) — smoothed IDF, L2 normalization.
 */

function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

function normalizeVector(v) {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  return norm === 0 ? v : v.map((x) => x / norm);
}

export function buildTfidfModel(documents) {
  const tokenizedDocs = documents.map(tokenize);

  const vocabularySet = new Set();
  tokenizedDocs.forEach((tokens) => tokens.forEach((t) => vocabularySet.add(t)));
  const vocabulary = Array.from(vocabularySet);

  const df = vocabulary.map(
    (term) => tokenizedDocs.filter((tokens) => tokens.includes(term)).length
  );

  const n = documents.length;
  const idf = df.map((count) => Math.log((n + 1) / (count + 1)) + 1);

  const documentVectors = tokenizedDocs.map((tokens) => {
    const tf = {};
    tokens.forEach((t) => {
      tf[t] = (tf[t] ?? 0) + 1;
    });
    const vector = vocabulary.map((term, i) => (tf[term] ?? 0) * idf[i]);
    return normalizeVector(vector);
  });

  return { vocabulary, idf, documentVectors };
}

export function transformQuery(model, query) {
  const tokens = tokenize(query);
  const tf = {};
  tokens.forEach((t) => {
    tf[t] = (tf[t] ?? 0) + 1;
  });
  const vector = model.vocabulary.map((term, i) => (tf[term] ?? 0) * model.idf[i]);
  return normalizeVector(vector);
}

export function cosineSim(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
