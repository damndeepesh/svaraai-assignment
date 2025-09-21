## Q1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?
Use a pretrained model and freeze most layers, fine-tune lightly with strong regularization and stratified CV. Add smart data augmentation (paraphrases) and pseudo‑label unlabeled emails; use active learning to label the most uncertain samples first.

## Q2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

Evaluate per segment (industry, role, region) and monitor fairness metrics; retrain if gaps appear. Add confidence thresholds + human review for low‑confidence cases, and run toxicity/safety filters and ongoing audit logs.

## Q3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

Provide clear context (prospect role, company, pain point) and constraints (1‑2 sentences, concrete value, no fluff). Use few‑shot examples, require quoting a verifiable detail, and forbid generic phrases and hallucinations.
