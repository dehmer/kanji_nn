# Prompt: Work-Style Assessment Request

Paste the following into any session (technical or otherwise) where you'd
like Claude to produce a comparable assessment for later compilation:

---

> Based on this conversation, please compile a high-level assessment of my
> working style, written for someone (e.g. a future employer or reader of
> a CV) who has no access to this conversation's actual content — so avoid
> referencing specific project details, filenames, code, or subject
> matter; keep it abstracted to *how I work*, not *what we worked on*.
>
> Please cover these topics, to the extent this conversation gives you
> evidence for them (skip or note as "insufficient evidence" any topic
> this session didn't touch):
>
> - General problem-solving methodology
> - Decision-making style (including how complexity/scope is handled)
> - Problem understanding and decomposition
> - Rigor / empirical discipline (measuring vs. assuming)
> - Intellectual honesty and willingness to update on evidence
> - Domain-learning velocity (picking up unfamiliar territory)
> - Communication and collaboration style
> - Conceptual synthesis (naming/structuring fuzzy problems)
> - Scope discipline (deferring vs. chasing tangents)
> - Assessment as an experienced software engineer generally, if
>   applicable
> - Assessment as a programmer in the specific language(s) used, if
>   applicable
> - Code quality, if code was shared or written
>
> Please be honest and specific rather than generically flattering — note
> real limitations or rough edges alongside strengths. State plainly that
> this is based on a single session and may not generalize. Format as a
> markdown document with clear headers per topic so multiple such
> assessments can later be compared and compiled.

---

## Notes on using this across sessions

- Run this at the *end* of a substantive working session, not a short or
  purely conversational one — thin sessions will mostly yield "insufficient
  evidence" and won't be useful to compile.
- Vary the domains/session types you sample from (technical depth, creative
  work, planning/strategy, debugging under pressure, etc.) if possible —
  a work style only observed in one kind of task is a weaker signal than
  one that holds across different kinds of work.
- When compiling multiple assessments, look for what's *consistent* across
  independently-generated write-ups — that's the trustworthy signal, more
  so than any single session's specific phrasing.