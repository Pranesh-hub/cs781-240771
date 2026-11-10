# bilstm_all_phases.ipynb: one BiLSTM prediction for each of Phases 2-7

**Use:** upload the notebook to Kaggle, attach the dataset `cs781-phase1-v3-bilstm-recurrent` (it is in the `puciitk` account), add the secret `HF_TOKEN`
(and `KAGGLE_USERNAME` / `KAGGLE_KEY` if a phase's credentials check asks), turn Internet ON and pick GPU T4 x1, then Save & Run All.
Edit only the CONFIG cell: `RUN_PHASES` (which phases), `CAP_MIN` (training minutes per phase), `CHILD_TIMEOUT_MIN` (hard stop per phase).
Output: `/kaggle/working/predictions/phase_N/prediction.json` (+ `log.txt` per phase), `summary.json`, and `bilstm_predictions.zip`.
Finished phases are kept: a second session skips them, so the run can be split over two sessions.

| Phase | What runs | Source | Tested |
|---|---|---|---|
| 2 NER | BiLSTM tagger (embeddings + lowest LSTM layer frozen), then your counsel/JUDGE rule layer | new code + rule functions extracted from `cs781_champion_plus_rules_run_cells.ipynb` | CPU, tiny model, synthetic data: runs end to end |
| 3 Rhetorical roles | one full fine-tune of the BiLSTM (3 epochs) through the Phase 3 pipeline, BiLSTM registered via `arch_loaders` | `phase3_rr_pipeline.ipynb` cells + a registration cell | syntax only |
| 4 CJPE | `P5_bilstm_partial` | `phase4_cjpe_v4_final.ipynb` cells | syntax only |
| 5 Statutes | `E4_bilstm_partial` (pilot skipped) | `phase5_lsi_part2_E4_E5_final.ipynb` cells | syntax only |
| 6 Summaries | `S7_bilstm_partial` (BiLSTM seq2seq; the encoder + GPT-2 decoder variant is NOT built) | `00_BEST__p6__gpt2+bilstm__part4_kd_bilstm_s6_s7.ipynb` cells | syntax only |
| 7 Retrieval | `BL1` partial fine-tune, no continued pre-training, no dataset upload | `00_BEST__p7__bilstm__bilstm_retriever.ipynb` cells | syntax only |

Each phase is a separate Python process, so a crash in one phase cannot touch the others, and every cell is run inside try/except (like nbconvert --allow-errors).
Times are the source notebooks' own estimates plus the caps in CONFIG, not measurements. Phase 3's output uses the pipeline's "keyed" schema: check it against the official spec.
`build_sources/` has the scripts that generate the notebook from your existing notebooks (`python build_orch.py`).
