SELECT   id, dataset, literal, unicode, groups, data
FROM     glyph
WHERE    literal = 'ア'
AND      mode = 'L'
ORDER BY literal
