-- Gyan schema for MonkDB. {schema} and {dim} are substituted at startup.
-- Statements are separated by ";\n". Graph DDL lives in monkdb_graph.sql (enterprise feature).

CREATE TABLE IF NOT EXISTS {schema}.conversations (
  id TEXT PRIMARY KEY,
  hash TEXT,
  source TEXT,
  title TEXT,
  created_at TIMESTAMP WITH TIME ZONE,
  doc TEXT INDEX OFF
);

CREATE TABLE IF NOT EXISTS {schema}.chunks (
  id TEXT PRIMARY KEY,
  conv_id TEXT,
  idx INTEGER,
  text TEXT,
  ts TIMESTAMP WITH TIME ZONE,
  embedding FLOAT_VECTOR({dim}) WITH (similarity = 'cosine'),
  INDEX text_ft USING FULLTEXT (text) WITH (analyzer = 'english')
);

CREATE TABLE IF NOT EXISTS {schema}.atoms (
  id TEXT PRIMARY KEY,
  conv_id TEXT,
  type TEXT,
  title TEXT,
  summary TEXT,
  body TEXT,
  learned_at TIMESTAMP WITH TIME ZONE,
  doc TEXT INDEX OFF,
  embedding FLOAT_VECTOR({dim}) WITH (similarity = 'cosine'),
  INDEX atom_ft USING FULLTEXT (title, summary, body) WITH (analyzer = 'english')
);

CREATE TABLE IF NOT EXISTS {schema}.atom_concepts (
  atom_id TEXT,
  concept_id TEXT,
  PRIMARY KEY (atom_id, concept_id)
);

-- Graph vertices
CREATE TABLE IF NOT EXISTS {schema}.concepts (
  id TEXT PRIMARY KEY,
  name TEXT,
  aliases ARRAY(TEXT),
  mastery DOUBLE PRECISION,
  times_asked INTEGER,
  embedding FLOAT_VECTOR({dim}) WITH (similarity = 'cosine')
);

-- Graph edges
CREATE TABLE IF NOT EXISTS {schema}.relations (
  id TEXT PRIMARY KEY,           -- src|dst|type
  src TEXT,
  dst TEXT,
  type TEXT,
  weight DOUBLE PRECISION,
  atom_id TEXT
);

CREATE TABLE IF NOT EXISTS {schema}.cards (
  id TEXT PRIMARY KEY,
  atom_id TEXT,
  due TIMESTAMP WITH TIME ZONE,
  doc TEXT INDEX OFF
);

-- Time series of reviews -> streaks, forgetting curves, "Gyan Wrapped"
CREATE TABLE IF NOT EXISTS {schema}.reviews (
  card_id TEXT,
  ts TIMESTAMP WITH TIME ZONE,
  grade INTEGER
)
