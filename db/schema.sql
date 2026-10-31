-- Memory challenge database: what answering a question needs, nothing more.
--
-- A session's content is the same in every history it appears in, but its date belongs to the history:
-- turns hold content, history_sessions holds the date. Every search is scoped to one history.

CREATE EXTENSION IF NOT EXISTS vector;

-- A session in a history, with its date in that history
CREATE TABLE history_sessions (
    history_id   text NOT NULL,                          -- h_...
    session_id   text NOT NULL,                          -- s_...
    date_raw     text NOT NULL,                          -- '2023/05/20 (Sat) 02:21', sent to Reactor as written
    ts           timestamp NOT NULL,                     -- parsed, for date arithmetic
    PRIMARY KEY (history_id, session_id)
);
CREATE INDEX ON history_sessions (session_id);

-- Session content, stored once whatever the history
CREATE TABLE turns (
    turn_id      text PRIMARY KEY,                       -- t_...
    session_id   text NOT NULL,
    idx          int  NOT NULL,                          -- position in the session; neighbours are idx +- n
    role         text NOT NULL CHECK (role IN ('user', 'assistant')),
    content      text NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,  -- keyword search
    UNIQUE (session_id, idx)
);
CREATE INDEX ON turns USING gin (tsv);

-- Overlapping windows over a session's turns, in order (db/chunk.py). A chunk can span several turns.
-- Its span runs from (start_turn, start_char) to (end_turn, end_char): turns.idx and a character offset in that
-- turn's content, end exclusive. Neighbours overlap, so merging chunks means rendering the span from the first
-- one's start to the last one's end again (chunk.py: expand), never concatenating their texts.
CREATE TABLE chunks (
    chunk_id      text PRIMARY KEY,                      -- '<session_id>:<idx>'
    session_id    text NOT NULL,
    idx           int  NOT NULL,                         -- position in the session; neighbours are idx +- n
    prev_chunk_id text REFERENCES chunks DEFERRABLE INITIALLY DEFERRED,  -- NULL for a session's first chunk
    next_chunk_id text REFERENCES chunks DEFERRABLE INITIALLY DEFERRED,  -- NULL for its last
    start_turn    int  NOT NULL,
    start_char    int  NOT NULL,
    end_turn      int  NOT NULL,
    end_char      int  NOT NULL,
    n_tokens      int  NOT NULL,                         -- of text, by the embedding model's tokenizer
    has_question  boolean NOT NULL,                      -- holds a sentence with '?'
    text          text NOT NULL,                         -- '[USER] ...' / '[ASSISTANT] ...': what is embedded and shown
    schemed_text  text,                                  -- NULL for now: a structured version, maybe later
    embedded_text         vector(1024),                  -- set the dimension to your embedding model's
    embedded_schemed_text vector(1024),
    UNIQUE (session_id, idx)
);

CREATE TABLE questions (
    question_id  text PRIMARY KEY,                       -- q_...
    history_id   text NOT NULL,
    question     text NOT NULL,
    date_raw     text NOT NULL,                          -- pass to client.py --question-date as written
    ts           timestamp NOT NULL
);
