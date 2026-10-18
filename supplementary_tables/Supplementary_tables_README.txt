SUPPLEMENTARY TABLES
Career Recommendation Beyond Overall Accuracy: Model Comparisons and Broader Occupational Retrieval

Read with Supplementary_Information_1_condensed_2026_10_18.docx. These files are submission attachments, independent of the planned GitHub repository.
CSV encoding: UTF-8 with BOM. First row contains column headings. Values and precision are transcribed exactly from the previous Word tables; no new estimates have been calculated. Unicode minus signs and confidence intervals are retained as displayed text. Blank cells, if any, remain blank.

FILE INDEX
Supplementary_Table_S7.csv — Table S7 Sensitivity of fusion performance to ranking length, input depth and neural weight on JobHop
Supplementary_Table_S8.csv — Table S8 Sensitivity of fusion performance to ranking length, input depth and neural weight on Decorte
Supplementary_Table_S9.csv — Table S9 Training duration in epochs and checkpoint selection for the self-attention model
Supplementary_Table_S13.csv — Table S13 History composition and agreement between recency and BiLSTM rankings
Supplementary_Table_S14.csv — Table S14 Fusion NDCG differences within outcome-defined evaluation groups
Supplementary_Table_S15.csv — Table S15 Observed and composition-standardised fusion NDCG differences
Supplementary_Table_S20.csv — Table S20 Fusion NDCG differences for the two existing BiLSTM training seeds
Supplementary_Table_S21.csv — Table S21 Chronological BiLSTM training and checkpoint selection
Supplementary_Table_S26.csv — Table S26 Chronological confirmation contrasts by training seed
Supplementary_Table_S27.csv — Table S27 Chronological fusion contrasts by training seed

INTERPRETATION
S7/S8: k is output length, L input depth and w neural weight. NDCG contrasts compare fusion with recency/BiLSTM; repeated recall compares fusion with recency, broader recall with BiLSTM at the same k. Simultaneous 95% intervals are estimate ± 0.01049774 (JobHop) or ± 0.11971212 (Decorte), one 96-contrast family per dataset; displayed rounding applies. See Section S9.
S9/S21: epochs and tuning scores describe checkpoint selection; neither table is a new performance comparison. See Sections S10/S16.
S13: KW denotes KARRIEREWEGE; KW unique is its unique nonmatching sample. Agreement is mean component top-ten overlap, not accuracy. See Section S13.
S14: outcome groups retain complete targets; values are descriptive conditional fusion NDCG differences. Decorte has only one mixed-target history. See Section S13.
S15: R is recency; N is BiLSTM. Standardisation uses pooled repeat-only/unseen-positive weights 0.292802/0.707198. This does not control all cross-source differences. See Section S14.
S20: two existing BiLSTM seeds; fusion NDCG contrasts are descriptive. See Section S15.
S26/S27: five chronological training seeds; neural means BiLSTM. Change in model gap is the unseen-positive neural-minus-recency contrast minus the overall contrast. Repeated/broader recall contrasts use their eligible subsets. See Section S16 for the paired inference, sample sizes and uncertainty estimates retained in Word.

All files contain aggregate study results; no individual career histories are included.