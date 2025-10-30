import json
import pandas as pd


model_path = "ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd-blk0"
save_scores = json.load(open(f"{model_path}/scores_l2-category.json"))

res = {}
for category in [
            "overall",
            "visual_recognition",
            "localization",
            "ocr",
            "counting",
            "hallucination",
            "image_retrieval",
            "threed",
            "visual_captioning",
            "visual_grounding",
            "doc_understanding",
            "action_recognition",
            "pixel_level_perception",
            "image-to-image_translation",
            "relation_reasoning",
            "intelligence_quotient_test",
            "emotion",
            "visual_illusion",
            "meme_understanding",
            "visual_prompt_understanding",
            "anomaly_detection",
            "keypoint_detection",
            "visual_commonsense_reasoning",
            "image_evaluation_judgement",
            "multiple_image_analysis",
            "cross_image_matching",
            "temporal_understanding",
            "visual_code",
            "medical_understanding",
            "autonomous_driving",
            "discipline_knowledge_reasoning",
            "embodied_ai",
            "gui_navigation",
]:
    acc = save_scores[category]['acc'] * 100
    mean_score = save_scores[category]['mean_score'] * 100
    res[category] = f"{acc:.2f}/{mean_score:.2f}"
res = pd.DataFrame([res]).T
res.to_csv(f"{model_path}/scores.csv")
print(res)