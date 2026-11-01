# The column you did not read

*Why a 98.7% accurate PII scanner is the wrong contract, and what a recall guarantee actually costs.*

Every data platform eventually meets the same column. It is called `notes`, or `comments`, or `attr_7`. It holds free text typed by support agents and, sometimes, by customers. Most rows say things like "refund approved by finance". A few say "please call me at (11) 98877-1234". The scanner that tags columns for the catalog reads twenty rows, sees no phone number, and marks the column as safe. Three weeks later the column is in a shared analytics schema, a vendor extract and a model training set.

Nobody did anything wrong in that story. The scanner was accurate. That is the problem.

## Accuracy is not the contract

I built a small PII tagging pipeline to make this concrete. It reads sampled values from a column, never the column name (real schemas are full of `col_17`), and assigns one of nineteen classes: nine PII types such as email, phone, card number, national id, person name, birth date and "free text with leaked contact data", and ten non-PII types such as order ids, SKUs, hostnames, cities and plain free text. The data is synthetic but adversarial on purpose. Surnames double as city names. Birth dates overlap with business dates. Raw eleven-digit national ids look like long order ids. Leaking comment columns leak in only 8% to 35% of rows.

The classifier is plain multinomial logistic regression over hashed character n-grams of each value and of its "shape" (`Ana-42` becomes `Aaa-99`), plus a handful of distribution statistics. On 1900 held-out columns it is 98.7% accurate.

It also misses 15% of leaking free-text columns. Almost all of its errors live in one cell of the confusion matrix, and it is the cell a privacy officer cares about most.

A risk owner does not want an accuracy number. They want a dial: miss at most 5% of personal-data columns, and tell me what that costs. That is a statement about a conditional error rate on a decision, and it turns out there is a clean, distribution-free way to make it.

## From scores to sets

The first step is to stop treating the classifier's argmax as the answer and ask how unsure it is. Split conformal prediction does this without trusting the model's probabilities. Take a calibration set, compute the score 1 minus the probability of the true class for each column, and take its ceil((n+1)(1-alpha))/n quantile q. For a new column, return every class whose score is at most q. Under exchangeability, the true class is in the set with probability at least 1 minus alpha.

That guarantee is marginal, and marginal is the trap again. At 95%, marginal sets covered 95.7% of all test columns while covering only 79% of leaking free text. The easy classes paid for the hard one.

Mondrian conformal fixes this by computing one quantile per class, which makes the guarantee hold for every class separately. Leaking free text jumped to 98% coverage. The price is that clean free-text columns now often get a set containing both `free_text` and `free_text_pii`. That ambiguity is not noise. It is the model telling you, with a guarantee attached, which columns it cannot decide from the rows it was given.

There is a sharp edge here worth knowing before you deploy it. With 40 calibration columns per class, a 98% class-conditional set is impossible: the required rank ceil(41 times 0.98) is 41, larger than n, so every set becomes the full label list. Class-conditional guarantees at level alpha need at least 1/alpha minus 1 calibration examples of every class. For rare column types, that is a data collection problem, not a modeling one.

## Read more rows, not a bigger model

What should happen to an ambiguous column? The fashionable answer is to send it to an LLM. I did that as a control: a Claude model labelled 100 test columns zero-shot, from values only, blind to labels and to the classifier. On the rows it saw, it was the better classifier: 97% accuracy against 94% for the logistic regression on the same columns, and not a single false PII call.

On the 50 escalated columns it missed 3 of 5 leaking free-text columns. All three had no leak in the sample. The information was not in the input.

So the pipeline does something duller. When a column's set mixes PII and non-PII labels, it re-reads the same table with four times the rows, at least sixty. That affected 4.2% of test columns, every one of them free text. In the original samples, 89% of leaking columns showed a leak; after escalation, 100% did. On the same 50 escalated columns, the guard missed one leaking column where the LLM missed three. The numbers are small and I would not call it a benchmark, but the mechanism is general: a model cannot classify evidence it was never shown, and conformal sets tell you exactly where to spend extra reads.

## Turning the dial

The last step is the decision itself. The pipeline sums the probability on PII classes and quarantines the column when that sum crosses a threshold lambda. Conformal risk control picks lambda from calibration data. On the PII calibration columns, define the loss as 1 when the column would not be quarantined. It is bounded and grows with lambda, so the largest lambda with n/(n+1) times the empirical miss rate plus 1/(n+1) at most alpha guarantees an expected miss rate of at most alpha on new columns.

One subtlety cost me a design change. The escalation rule depends on the Mondrian thresholds. If the same calibration data chose the rule and set lambda, calibration columns would no longer be exchangeable with future ones. So the two calibrations use disjoint data.

At a 5% target, the two-stage guard missed 4.3% of PII columns on the held-out set and wrongly quarantined 0.2% of the rest. Escalation did not reduce the miss rate, because risk control spends the whole budget by design. It reduced the price: at a strict 2% target, the one-stage pipeline had to lower lambda to 0.32 and quarantine 3% of clean free text, while the two-stage pipeline kept lambda at 0.83 and quarantined 1%.

## What the guarantee does not say

Two findings deserve as much space as the good ones.

First, "in expectation" means what it says. Over 300 random re-splits, the average miss rate at a 5% target was 4.97%, right on target. But 44% of individual runs landed above 5%. If the contract with legal is "at most 5% with 90% confidence", you need a high-probability variant. Replacing the CRC inequality with an exact Clopper-Pearson upper bound, scanned as a fixed sequence so no multiplicity correction is needed, cut the share of runs above target to 11%, at the cost of a slightly more aggressive threshold.

Second, the guarantee is about columns that look like the calibration columns. I built a shifted test set where leaking comments leak in 2% to 8% of rows instead of 8% to 35%. With the old lambda, the miss rate went from 4.3% to 11.8%, and recall on leaking free text fell to 18%. Two things broke together: exchangeability, and the features, because a mean-pooled "share of rows with an email" is diluted by rare leaks. The fixes are boring and necessary: max-pooled "any row" indicators, and calibration data refreshed from recent production samples rather than a historical labelled corpus.

## The takeaway

Governance teams do not need a smarter scanner as much as they need an honest one. A cheap classifier, a conformal layer that knows when it is unsure, a rule that reads more rows exactly there, and a threshold set by the business's risk appetite rather than by intuition, gets you a number you can put in a policy: the expected share of personal-data columns that slip through. It also tells you, with measurements, where that number stops being true.

The code, the synthetic generator, the cached LLM labels and every number in this article are reproducible in about thirty seconds on a laptop: [github.com/julianodutraa/pii-recall-guard](https://github.com/julianodutraa/pii-recall-guard).
