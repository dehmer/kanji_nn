--
--
--
CREATE TABLE glyph (
    id text NOT NULL,
    dataset text,
    literal text,
    unicode text,
    groups text,
    width int4,
    height int4,
    mode text,
    data bytea,
    PRIMARY KEY (id)
);

CREATE INDEX idx_glyph__literal ON glyph (literal);
CREATE INDEX idx_glyph__dataset ON glyph (dataset);
