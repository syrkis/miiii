-- Drop tables if they exist (in correct order due to foreign key constraints)
DROP TABLE IF EXISTS metrics;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS tasks;

CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    task TEXT NOT NULL,
    prime INTEGER NOT NULL CHECK (prime > 0),
    latent_dim INTEGER NOT NULL CHECK (latent_dim > 0),
    depth INTEGER NOT NULL CHECK (depth > 0),
    heads INTEGER NOT NULL CHECK (heads > 0),
    lr REAL NOT NULL CHECK (lr > 0),
    l2 REAL NOT NULL CHECK (l2 >= 0),
    dropout REAL NOT NULL CHECK (dropout >= 0 AND dropout < 1),
    epochs INTEGER NOT NULL CHECK (epochs > 0)
);

CREATE TABLE tasks (
    task TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE metrics (
    run_id INTEGER,
    task TEXT NOT NULL,
    epoch INTEGER NOT NULL CHECK (epoch >= 0),
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    loss REAL NOT NULL,
    f1 REAL NOT NULL CHECK (f1 >= 0 AND f1 <= 1),
    acc REAL NOT NULL CHECK (acc >= 0 AND acc <= 1),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (task) REFERENCES tasks(task),
    PRIMARY KEY (run_id, task, epoch, split)
);

-- Optimized indexes for common queries
CREATE INDEX idx_metrics_task_perf ON metrics(task, split, f1 DESC);
CREATE INDEX idx_metrics_task_progress ON metrics(task, run_id, epoch);

-- For comparing hyperparameters
CREATE INDEX idx_runs_hyperparams ON runs(latent_dim, depth, heads, lr);
