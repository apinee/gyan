-- Native graph registration (MonkDB enterprise). Enabled with MONKDB_NATIVE_GRAPH=1.
CREATE GRAPH IF NOT EXISTS gyan_knowledge;
CREATE VERTEX TABLE IF NOT EXISTS {schema}.concepts FOR GRAPH gyan_knowledge KEY id;
CREATE EDGE TABLE IF NOT EXISTS {schema}.relations FOR GRAPH gyan_knowledge SOURCE KEY src TARGET KEY dst
