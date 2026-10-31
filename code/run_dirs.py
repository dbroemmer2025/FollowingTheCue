from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIRS = [
    ROOT / "e47/qwen2_5_7b_movie_recommendations_thinking_v10",
    ROOT / "e47/qwen2_7b_it_movie_recommendations_thinking_v10",
    ROOT / "e47/gemma_2_9b_it_movie_recommendations_thinking_v10",
    ROOT / "e47/mistral_nemo_instruct_2407_movie_recommendations_thinking_v10",
    ROOT / "e47/olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10",
    ROOT / "e47/qwen2_5_7b_temporal_sequences_thinking_v10",
    ROOT / "e47/qwen2_7b_it_temporal_sequences_thinking_v10",
    ROOT / "e47/gemma_2_9b_it_temporal_sequences_thinking_v10",
    ROOT / "e47/mistral_nemo_instruct_2407_temporal_sequences_thinking_v10",
    ROOT / "e47/olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10",
]
