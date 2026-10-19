# Shared Data Schemas

All pipeline stages exchange data using the structures defined in `src/data_schemas.py`.

## Core Types

### Document
- `doc_id`: str
- `text`: str
- `domain`: optional (news / finance / medicine / law / ...)
- `source`: optional
- `metadata`: dict

### EDU
- `edu_id`: stable string (e.g. `doc1_edu_000`)
- `doc_id`: str
- `text`: original sentence text
- `index`: 0-based order
- `start_char`, `end_char`: optional character offsets
- `metadata`: dict

### EDUPair
- `pair_id`, `doc_id`
- `edu_i_id`, `edu_j_id`, indices, `distance`
- `similarity`: float
- `selected`: bool
- `metadata`

### QUDQuestion
- `question_id`, `pair_id`, `doc_id`
- `question_text`
- `category`, `specificity`
- `generation_config`
- `is_valid`, `rejection_reason`
- `metadata`

### SalienceScore
- `question_id`, `pair_id`
- `centrality`, `answerability`, `final_score`
- `metadata`

### RelationPrediction
- `prediction_id`, `pair_id` / `span_id`
- `label`, `confidence`
- `features`, `metadata`

### AnswerFitScore
- `question_id`, `answer_edu_id`, `score`
- `metadata`

### DiscourseNode / DiscourseTree
- Recursive tree structure with span indices, relation labels, confidences, and child nodes.

## Serialization

Use `to_dict()` / `from_dict()` methods and the helpers in `src/data_schemas.py` (`save_json`, `load_json`).
