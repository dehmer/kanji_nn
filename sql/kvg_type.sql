
--
--
--
CREATE TABLE kvg_type (
  literal text,       -- character
  stroke_idx int,     -- 0-based stroke index
  type_literal text,  -- unicode type literal
  code_point text,    -- unicode code point for type literal
  cjk_name text       -- stroke name
);
