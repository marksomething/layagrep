# layagrep

Semantic grep for files and stdin. Instead of matching literal words, `layagrep` asks [Laya](https://huggingface.co/convaiinnovations/laya) whether each line matches a natural-language description. It uses Laya's `noul` answer type and treats its returned value as the probability that the answer is **yes**.

## Requirements and setup

This project targets Python 3.13 and uses [uv](https://docs.astral.sh/uv/) for the virtual environment and dependency management. From the project directory:

```sh
uv sync
```

The first search downloads Laya's model checkpoint (the English model is about 800 MB). Model inference can require substantial memory; Laya automatically routes non-English input to its multilingual checkpoint.

## Usage

```sh
uv run layagrep DESCRIPTION [FILE ...]
```

For example:

```sh
uv run layagrep 'a user reports a billing problem or asks for a refund' support.log
cat support.log | uv run layagrep 'a customer threatens to cancel their account'
uv run layagrep 'mentions a failed deployment' src/*.log --scores --line-number
```

The default threshold is `0.5`. Lines are evaluated one at a time; only lines whose Laya `noul` yes-probability meets or exceeds the threshold are printed. A higher threshold is stricter:

```sh
uv run layagrep 'contains a security incident' app.log --threshold 0.8 --scores
```

Use `-` to read stdin explicitly. If no files are given, stdin is used.

Speed and ranking:

```sh
# score lines in batches of 32 (much faster on large files)
uv run layagrep 'a database connection error' app.log --batch-size 32 --scores

# show only the 10 best matches across all inputs, best first
uv run layagrep 'mentions memory pressure' *.log --top 10 --scores --line-number

# rank every match by score instead of input order
uv run layagrep 'mentions memory pressure' app.log --sort score --scores
```

### Options

- `--help`: show command help (`-h` follows grep convention and suppresses filenames).
- `-t, --threshold FLOAT`: match when noul probability is at least this value (0–1; default `0.5`).
- `-m, --mode {noul,choice}`: question type used for matching (default `noul`; see Notes on Laya).
- `-b, --batch-size N`: score N lines per batched call — faster on large inputs, but lines are scored in chunks rather than lazily one at a time.
- `--sort {line,score}`: output order — `line` (default) or `score` (best matches first).
- `--top N`: print only the N best matches across all inputs (implies `--sort score`; cannot be combined with `-c`, `-l`, or `-q`).
- `--scores`: prefix each result with its noul probability.
- `--json`: emit matching results as JSON Lines with `file`, `line`, `text`, `score`, `confidence`, and `routing` fields. `confidence` is Laya's calibrated answer confidence (the probability mass on the reported answer — the number Laya's model card recommends gating on). `routing` records which checkpoint answered and why.
- `-n, --line-number`: include line numbers.
- `-H, --with-filename` / `-h, --no-filename`: show or suppress filenames. Filenames are shown by default when searching multiple inputs.
- `-v, --invert-match`: select lines with a probability below the threshold.
- `-c, --count`: print the selected-line count for each input.
- `-l, --files-with-matches`: print an input's name when it has a match.
- `-q, --quiet`: stop at the first match without printing output.

Like grep, exit status is `0` when there is at least one selected line, `1` when there are none, and `2` when an input or inference error occurs. Piping into a command that exits early (e.g. `| head`) terminates silently with grep's SIGPIPE status.

## Notes on Laya

Laya is a non-autoregressive decision model. A `noul` question returns a yes/no probability; this tool passes each input line as the state and asks whether it matches the provided description. It does not generate a textual explanation for each match. Since this is a model decision rather than an exact keyword test, tune the threshold and validate results for your data.

Laya's model card notes that `noul` answers can sometimes follow the rendered `false:`/`true:` option labels rather than the content (issue #156). If `noul` results look stuck, run with `--mode choice`, which asks the same question as a two-option choice with neutral keys and the yes/no wording in the option descriptions — the mitigation the model card recommends.

For faster repeated searches, model lifecycle and device selection can be adjusted in `src/layagrep/engine.py` where the Laya `Router` is constructed.

## Layout

- `src/layagrep/engine.py` — the search engine: builds the Laya `noul` question, scores text, and lazily yields `Match` records. No CLI or output concerns, so it can back a library caller or server.
- `src/layagrep/cli.py` — command-line only: argument parsing, reading inputs, formatting matches, exit statuses.
- `tests/test_engine.py` / `tests/test_cli.py` — mirror that split.

Run the tests with:

```sh
uv run python -m unittest discover -s tests -v
```

## Development

Lint, type-check, and test (the same steps CI runs on every push):

```sh
uv run ruff check .
uv run mypy
uv run python -m unittest discover -s tests -v
```

The tests use fake predictors, so they never download the Laya checkpoint. CI is defined in `.github/workflows/ci.yml`.
