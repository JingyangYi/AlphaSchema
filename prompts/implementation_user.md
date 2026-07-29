Implement the stock factor plan below.

Semantic order: event -> context -> qualities -> direction -> output.

Implementation procedure:
1. Implement the event as a measurable market event. For a time-series event,
   first construct a causal score independently for every symbol. For a
   cross-sectional event, compare symbols only at the same timestamp.
2. Use context to construct the event's reference environment. A time-series
   context is calculated independently by symbol; a cross-sectional context
   may describe same-date groups, market state, or peer-relative position.
3. Use qualities for filtering, confirmation, normalization, liquidity
   control, cleanliness constraints, or cross-sectional rank/z-score. A
   quality must not independently determine the trading direction.
4. Use direction to convert the semantic score into the final alpha sign and
   strength, such as relative continuation, relative reversal, or ranking
   stronger stocks above weaker stocks.
5. Use output to determine the final matrix form, including normalization,
   clipping, smoothing, persistence, or event decay.
6. Propose two test periods in `FACTOR_META`: one fast and one slow. The slow
   period should normally be three to five times the fast period.

Input-field rules:
- Prefer adjusted open, high, low, close, volume, amount, and vwap.
- Liquidity or capacity context may use amount, volume, turnover_rate,
  market_value, or neg_market_value.
- Trading-state context may use is_open, is_st, limit_pct, limit_up, or
  limit_down.
- Unless the schema explicitly requires otherwise, do not use raw_* or twap_*
  fields, and do not assign stock-market meaning to open_interest.

Cross-sectional safety rules:
- At time t, output may use contemporaneous information across all symbols but
  no observation from any symbol after t.
- Do not use full-sample global means, standard deviations, ranks, or future
  windows.
- `rank(axis=1)`, `mean(axis=1)`, and `std(axis=1)` are allowed same-date
  operations.
- Time smoothing must be causal: trailing `rolling`, `ewm(adjust=False)`,
  positive `shift`, `pct_change`, or `diff`.
- Call `pct_change(..., fill_method=None)` and handle missing values explicitly.
- Do not use positional `.iloc` slicing for adjacent rows. Use `shift(1)` and
  wide-frame vectorized operations.
- Return a finite time-by-symbol matrix with symbols as columns. Handle warmup,
  infinities, missing optional inputs, and extreme values explicitly.
- Do not change the panel function into a single-symbol function even if the
  selected schemas are mostly time-series based.

If `extra_implementation_prompt` is present in the task JSON, follow it as
implementation guidance without changing the semantic plan.

Factor plan:
```json
{{PLAN_JSON}}
```
