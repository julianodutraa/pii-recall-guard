# Prompt used for the LLM second opinion

The model receives only the sampled values of each column (no column name,
no ground truth, no classifier output) and this instruction:

> You are a data governance reviewer. Each item below is a sample of rows
> from one database column whose name is unknown. Assign exactly one label
> per column from this list.
>
> PII labels: `email`, `phone`, `card_number`, `national_id` (CPF style tax
> id, formatted or 11 raw digits), `person_name`, `street_address`,
> `ip_address`, `birth_date`, `free_text_pii` (free text comments where at
> least some rows contain personal contact data such as an email or phone).
>
> Non-PII labels: `order_id`, `sku`, `event_timestamp`, `amount`, `status`,
> `hostname`, `uuid`, `city`, `calendar_date` (generic business date),
> `free_text` (free text comments with no personal data).
>
> Empty strings and tokens like N/A, null, unknown, 0 or a lone dash are
> missing values and carry no signal. Answer with a JSON object mapping each
> column id to its label.

The answers are cached in `llm_labels.json` together with provenance, so the
evaluation is reproducible without API access.
