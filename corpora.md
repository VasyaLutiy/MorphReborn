# Introducing GPT Morph `.corpora`

`.corpora` is a folder with the project-specific context data, provided to extend the project text corpus, which is used for morphing.

The simplest way to include additional context data is in the form of Markdown files, nicely integrated to the text corpus of any project.

See [`documentation/modeling-approach.md`](./documentation/modeling-approach.md) for how `.corpora` fits into GPT Morph's whole-project context model — it needs no special handling, since it is just another folder of Markdown files picked up by the same whole-project walk as the rest of the source tree.
