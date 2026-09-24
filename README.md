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

### Options

- `--help`: show command help (`-h` follows grep convention and suppresses filenames).
- `-t, --threshold FLOAT`: match when noul probability is at least this value (0–1; default `0.5`).
- `--scores`: prefix each result with its noul probability.
- `--json`: emit matching results as JSON Lines with `file`, `line`, `score`, and `text` fields.
- `-n, --line-number`: include line numbers.
- `-H, --with-filename` / `-h, --no-filename`: show or suppress filenames. Filenames are shown by default when searching multiple inputs.
- `-v, --invert-match`: select lines with a probability below the threshold.
- `-c, --count`: print the selected-line count for each input.
- `-l, --files-with-matches`: print an input's name when it has a match.
- `-q, --quiet`: stop at the first match without printing output.

Like grep, exit status is `0` when there is at least one selected line, `1` when there are none, and `2` when an input or inference error occurs.

## Notes on Laya

Laya is a non-autoregressive decision model. A `noul` question returns a yes/no probability; this tool passes each input line as the state and asks whether it matches the provided description. It does not generate a textual explanation for each match. Since this is a model decision rather than an exact keyword test, tune the threshold and validate results for your data. Laya's model card also notes that the base English checkpoint's `noul` answers can sometimes follow the `false`/`true` option labels rather than the content, so check its behavior on your use case.

For faster repeated searches, model lifecycle and device selection can be adjusted in `src/layagrep/cli.py` where the Laya `Router` is constructed.
