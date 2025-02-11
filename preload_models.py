# from sentence_transformers import SentenceTransformer
# import os
#
# # Load model names from environment variables (fallback to default models)
# MODEL_CONFIG = {
#     "mpnet": os.getenv("MODEL_MPNET", "sentence-transformers/all-mpnet-base-v2"),
#     "minilm": os.getenv("MODEL_MINILM", "sentence-transformers/all-MiniLM-L6-v2"),
#     "mxbai": os.getenv("MODEL_MXBAI", "mixedbread-ai/mxbai-embed-large-v1"),
#     "sentence_t5": os.getenv("MODEL_SENTENCE_T5", "sentence-transformers/sentence-t5-base"),
#     "multilingual_e5": os.getenv("MODEL_MULTILINGUAL_E5", "intfloat/multilingual-e5-large"),
# }
#
# # Download & Cache Models
# for model_name, model_path in MODEL_CONFIG.items():
#     print(f"Downloading {model_name} ({model_path})...")
#     SentenceTransformer(model_path)
#
# print("All models downloaded and cached successfully.")
