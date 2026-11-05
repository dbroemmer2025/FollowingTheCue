# E50 v5: output-decoding repair

E50 v5 fixes a model-specific output-decoding defect without changing any
frozen v3/v4 run. In the v4 DeepSeek-R1-Distill-Llama-8B movie pilot, vLLM
0.33.0 returned ByteLevel token surfaces (`Ġ` for spaces, `Ċ` for newlines)
in `CompletionOutput.text`. The final line `Answer:ĠC` was therefore not
recognized as `Answer: C`; zero natural answers were recorded and no cued
trials were assigned. The saved token IDs and raw output remain intact.

The v5 runner first asks the pinned tokenizer to decode the generated token
IDs. For this one model, if the result still contains ByteLevel surfaces, it
uses the standard `tokenizers.decoders.ByteLevel` decoder. The raw vLLM text,
canonical text, and decoder used are saved separately. Other profiles fail
before scoring/saving a batch if widespread ByteLevel surfaces appear; the
correct decoder for those models must be established explicitly. The `inspect`
command also checks an answer/newline round-trip for DeepSeek-Llama.

The original v4 pilot remains **NO-GO under its frozen code**. A CPU-only
diagnostic can show the recoverable answers without modifying that run:

```bash
python code/e50_reasoning_run_v5.py audit-decoding \
  --source-run e50/deepseek_r1_llama8b_movie_v4
```

Expected on the saved pilot: 80 surface rows, zero stored natural answers,
76 recoverable natural answers, two capped responses, and two genuine
non-option answers. This diagnostic does not create cued trials or retroactively
turn the frozen v4 NO-GO into a GO.

## Fresh DeepSeek-Llama pilot

Use a **new directory** and a new seed so the interim sample is not the same
40 items already inspected. From the repository root on Colab (L4 or A100):

```bash
python code/e50_reasoning_run_v5.py freeze \
  --out e50/deepseek_r1_llama8b_movie_v5 \
  --source-run e47/qwen2_5_7b_movie_recommendations_thinking_v10 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --revision auto --seed 51
python code/e50_reasoning_run_v5.py inspect \
  --out e50/deepseek_r1_llama8b_movie_v5
python code/e50_reasoning_run_v5.py run \
  --out e50/deepseek_r1_llama8b_movie_v5 --stop-after-interim
python code/e50_reasoning_run_v5.py report \
  --out e50/deepseek_r1_llama8b_movie_v5
```

Inspect `interim_think.json` and several `progress.jsonl` rows. Check that
`text` and `final` have ordinary spaces/newlines, `decoder_used` is recorded,
natural answers are parsed, and the frozen switch/resist occupancy gate passes
for **both** cue families. If the interim is GO, continue the same directory:

```bash
python code/e50_reasoning_run_v5.py run \
  --out e50/deepseek_r1_llama8b_movie_v5
python code/e50_reasoning_run_v5.py report \
  --out e50/deepseek_r1_llama8b_movie_v5
```

If it is NO-GO, stop and report the reason. Correct decoding may still reveal
high sampling instability or too few authority-cue resisters. Do not change
the gate after seeing the pilot.

## Other v4 models already running

Let them finish with the frozen v4 code. Then run the read-only diagnostic:

```bash
python code/e50_reasoning_run_v5.py audit-decoding \
  --source-run e50/<existing_v4_run>
```

For non-Llama models, `surface_rows` should be zero. If it is nonzero, do not
use that run's parsed answers until its tokenizer-specific decoding is fixed
under a new protocol. If it is zero, this particular defect is absent; the
normal interim gates still decide whether to continue.

## Review saved traces

The E50 `export-review` command writes channel-separated CSV rows, but does not
show an item together with its two uncued samples and all cue conditions. The
read-only viewer does; it checks saved record hashes and never reparses answers:

```bash
code/show_e50_trace.sh --run e50/magistral_small_2509_movie_v4 --list \
  --condition disagree_authority --switch yes --think-mention yes
code/show_e50_trace.sh --run e50/magistral_small_2509_movie_v4 95 \
  --condition disagree_authority --output-text reports/e50_item_95.txt
```

Use `--full-prompts` for every complete cue prompt, `--rendered-prompt` for the
exact chat-template input, `--raw-generation` for unsplit backend text, and
`--token-ids` for stored IDs. The default shows the uncued prompt once and a
diff for each cued prompt. `--list --forced yes --cap no` finds uncapped parser
failures. The lexical mention flags are screening aids only. Review cue presence,
cue consideration, and causal attribution separately in both the thinking and
final channels. Do not treat a final answer's silence as evidence that the
thinking channel was silent.
