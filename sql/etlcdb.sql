--
--
--
CREATE TABLE etlcdb (
    id text NOT NULL,
    entry text,
    literal: text,
    unicode text,
    groups text,
    width int4,
    height int4,
    mode text, -- PIL image mode
    data bytea,
    PRIMARY KEY (id)
);
