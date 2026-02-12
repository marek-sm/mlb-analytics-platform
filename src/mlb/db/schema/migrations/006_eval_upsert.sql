-- 006_eval_upsert.sql: Add unique constraint for per-metric upsert on eval_results
-- Implements FC-20: Prevent data loss on partial metric writes

-- Add unique constraint on (eval_date, market, metric) to enable safe upserts
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_unique ON eval_results (eval_date, market, metric);
