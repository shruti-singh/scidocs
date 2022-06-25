from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths

# point to the data, which should be in scidocs/data by default
data_paths = DataPaths()

# specter_pretrained_pt, specter_pretrained_hf, oag_init_silm, oagbert_indexfixedfrom_oag, oag_fromnewpushonthudmnlprepo
# model_name could be [specter, shf, oag, oagbert, newoag]
model_name = 'oagbert'

# point to the included embeddings jsonl
classification_embeddings_path = 'data/trained_embs/{}-embeddings/cls.jsonl'.format(model_name)
user_activity_and_citations_embeddings_path = 'data/trained_embs/{}-embeddings/user-citation.jsonl'.format(model_name)
recomm_embeddings_path = 'data/trained_embs/{}-embeddings/recomm.jsonl'.format(model_name)

# now run the evaluation
scidocs_metrics = get_scidocs_metrics(
    data_paths,
    classification_embeddings_path,
    user_activity_and_citations_embeddings_path,
    recomm_embeddings_path,
    val_or_test='val',  # set to 'val' if tuning hyperparams
    n_jobs=12,  # the classification tasks can be parallelized
    cuda_device=-1  # the recomm task can use a GPU if this is set to 0, 1, etc
)

print(scidocs_metrics)
