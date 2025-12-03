import json
import os
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    model_path = "ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd-sft737k-ftclip-ftsd-blk5"
    # model_path = "ross-siglip-qwen2-7b-flux-kl8-dit3x-pt558k-sft737k"
    result_path = f"../VLMEvalKit/outputs/{model_path}/{model_path}_MMT-Bench_VAL.xlsx"
    data = pd.read_excel(result_path, sheet_name="Sheet1").to_dict("records")
    os.makedirs(f"./{model_path}", exist_ok=True)
    results = []
    for idx, item in enumerate(tqdm(data)):
        img_path = f"/root/LMUData/images/MMT-Bench_VAL/{item['index']}.jpg"

        info = {
            "index": item["index"],
            "category": item["category"],
            "l2-category": item["l2-category"],
            "answer": item["answer"],
            "prediction": item["prediction"],
            "image_path": img_path,
            "correct": int(item["answer"].lower() == item["prediction"][0].lower()),
        }
        results.append(info)

    for k in ["l2-category"]:
        
        save_scores = {}
        all_category = set([x[k] for x in results])
        for category in all_category:
            correct = [x["correct"] for x in results if x[k] == category]
            acc = sum(correct) / len(correct)
            save_scores[category] = {"acc": acc}

        correct = [x["correct"] for x in results]
        acc = sum(correct) / len(correct)
        save_scores["overall"] = {"acc": acc}

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
        res[category] = f"{acc:.2f}"
    res = pd.DataFrame([res]).T
    res.to_csv(f"{model_path}/scores.csv")
    print(res)