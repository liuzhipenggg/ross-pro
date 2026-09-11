#!/usr/bin/env python3
"""One-off: migrate PROBE_EDIT to compare3 + add SD15/SD35 probe questions."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDIT = ROOT / "outputs/recon_pairs_random/PROBE_EDIT.json"

# Per-image patches: notes + new questions
PATCHES: dict[int, dict] = {
    0: {
        "看SD15": "环/绿球大致在；字母图例糊成乱线；+/- 符号消失",
        "看SD35": "核更像 3D 渲染球；环仍在；字母仍不可读；+/- 仍丢",
        "questions_没重建出来": [
            {"id": "00i", "question": "Are the plus (+) and minus (-) charge symbols inside the spheres readable?", "answer": "no in both SD15 and SD35"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "00x1", "question": "Which recon shows a more 3D glossy nucleus cluster, SD15 or SD35?", "answer": "SD35"},
            {"id": "00x2", "question": "Can the legend letters A/C/E be read in either reconstruction?", "answer": "no, both illegible"},
        ],
    },
    1: {
        "看SD15": "笔形在；握把圆孔→锯齿乱纹；透明段环糊",
        "看SD35": "笔尖更准；握把变光滑无圆孔；金属高光略糊",
        "questions_sd15_vs_sd35": [
            {"id": "01x1", "question": "Which recon turns the grip dimples into jagged zig-zag artifacts, SD15 or SD35?", "answer": "SD15"},
            {"id": "01x2", "question": "Which recon preserves a sharper pen tip shape?", "answer": "SD35"},
        ],
    },
    2: {
        "看SD15": "两人仍在；瓶→卡片/盒；裙子→迷彩纹；脸严重畸变；倒酒动作弱",
        "看SD35": "整体更糊、偏棕褐；瓶→白色光斑；八字胡消失；倒酒动作几乎没了",
        "questions_没重建出来": [
            {"id": "02h", "question": "Does the man have a prominent handlebar mustache?", "answer": "yes in GT; lost or blurred in both recons"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "02x1", "question": "Which recon has a sepia/brown tint instead of true black-and-white?", "answer": "SD35"},
            {"id": "02x2", "question": "Which recon preserves the pouring-into-glass interaction slightly better?", "answer": "SD15"},
            {"id": "02x3", "question": "In which recon does the bottle become a bright white glow between the figures?", "answer": "SD35"},
        ],
    },
    3: {
        "看SD15": "夜景骨架在但极糊、几乎无细节；地面/license plate 全丢",
        "看SD35": "细节更多但有绿色颗粒噪点；路灯上下光晕；地面像草地而非水泥",
        "questions_没重建出来": [
            {"id": "03f", "question": "Is a white license plate readable on a left parked car?", "answer": "yes in GT only"},
            {"id": "03g", "question": "Does the pavement look like smooth concrete rather than grainy/grass-like?", "answer": "yes in GT; SD35 looks grainy"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "03x1", "question": "Which recon shows a vertical light flare/streak from the street lamp?", "answer": "SD35"},
            {"id": "03x2", "question": "Which recon is sharper overall despite artifacts?", "answer": "SD35"},
        ],
    },
    4: {
        "看SD15": "黄色蛋块在；贝果→焦黑长条；种子消失",
        "看SD35": "构图类似；贝果仍像深色肉块；背景更糊",
        "questions_sd15_vs_sd35": [
            {"id": "04x1", "question": "Does either recon preserve the round bagel shape with a hole?", "answer": "no, both turn it into a dark block"},
            {"id": "04x2", "question": "Which recon keeps the yellow omelet mass clearer?", "answer": "roughly equal; SD35 slightly sharper"},
        ],
    },
    5: {
        "看SD15": "三只船+白鸟轮廓在；人/遮阳棚细节丢",
        "看SD35": "岩石纹理更好；鸟形更清；左船上人仍不可数",
        "questions_sd15_vs_sd35": [
            {"id": "05x1", "question": "Which recon shows sharper foreground rock texture?", "answer": "SD35"},
            {"id": "05x2", "question": "Can people on the left boat be counted in SD35?", "answer": "no, still too vague"},
        ],
    },
    6: {
        "看SD15": "女人+熊在；蓝圈文字→白色乱 scribble；脸塑料感",
        "看SD35": "熊更锐；脸 identity 崩（眼变黑洞）；文字→外星符号体",
        "questions_sd15_vs_sd35": [
            {"id": "06x1", "question": "Which recon keeps the woman's eye color closer to blue/hazel?", "answer": "SD15"},
            {"id": "06x2", "question": "Which recon renders the teddy bear fur sharper?", "answer": "SD35"},
            {"id": "06x3", "question": "Are magazine names (Vogue/Cosmo/Instyle) readable in either recon?", "answer": "no"},
        ],
    },
    7: {
        "看SD15": "布局在；日历 07→08；笔像钳子；硬币糊",
        "看SD35": "日历变文档乱文；中间 tray 出现红白小旗；钥匙→黑色遥控器",
        "questions_没重建出来": [
            {"id": "07h", "question": "Does a red-and-white flag-like object appear in the coin tray?", "answer": "yes in SD35 only (hallucination)"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "07x1", "question": "Which recon changes the calendar top-right number from 07 to 08?", "answer": "SD15"},
            {"id": "07x2", "question": "Which recon turns the keys into a black rectangular remote/device?", "answer": "SD35"},
        ],
    },
    8: {
        "看SD15": "停车场+购物车棚骨架在；红色横幅文字死",
        "看SD35": "结构更清；文字/药店 A 仍不可读",
        "questions_sd15_vs_sd35": [
            {"id": "08x1", "question": "Which recon preserves wet-ground reflection slightly better?", "answer": "SD35"},
        ],
    },
    9: {
        "看SD15": "北美东海岸轮廓在；高亮条变棕褐",
        "看SD35": "轮廓仍在；绿色高亮略恢复但仍不准",
        "questions_sd15_vs_sd35": [
            {"id": "09x1", "question": "Which recon keeps the coastal highlight closer to lime green?", "answer": "SD35 (still wrong vs GT)"},
        ],
    },
    10: {
        "看SD15": "竖条重复/碎片化；骑手红夹克色块在；人脸不可辨",
        "看SD35": "主场景更连贯；右下白色对角 smear 伪影",
        "questions_没重建出来": [
            {"id": "10h", "question": "Is there a large white diagonal streak across the bottom-right?", "answer": "yes in SD35 only (artifact)"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "10x1", "question": "Which recon shows vertical striping/fragmentation of the scene?", "answer": "SD15"},
            {"id": "10x2", "question": "Which recon makes the horse-jumping action easier to recognize?", "answer": "SD35"},
        ],
    },
    11: {
        "看SD15": "夜景+灯柱在；极糊",
        "看SD35": "细节更多；颜色偏绿黄噪点",
        "questions_sd15_vs_sd35": [
            {"id": "11x1", "question": "Which recon is less blurry overall?", "answer": "SD35"},
        ],
    },
    12: {
        "看SD15": "下半幅全黑；竹干在但无竹叶；雾过白",
        "看SD35": "全幅在；竹干向右倾斜；重度颗粒噪点；叶形仍丢",
        "questions_重建出来了": [
            {"id": "12d", "question": "Are distinct bamboo leaves visible?", "answer": "yes in GT; no in SD15; blurry patches in SD35"},
        ],
        "questions_没重建出来": [
            {"id": "12e", "question": "Is the bottom half of the frame missing/black?", "answer": "yes in SD15 only"},
            {"id": "12f", "question": "Are the bamboo stalks mostly vertical?", "answer": "yes in GT; slanted in SD35"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "12x1", "question": "Which recon crops out the bottom half as solid black?", "answer": "SD15"},
            {"id": "12x2", "question": "Which recon adds heavy grain/noise across the image?", "answer": "SD35"},
        ],
    },
    13: {
        "看SD15": "人物轮廓在；文字/条纹细节丢",
        "看SD35": "稍锐；条纹仍不可读",
        "questions_sd15_vs_sd35": [
            {"id": "13x1", "question": "Which recon preserves jersey stripe pattern slightly better?", "answer": "SD35"},
        ],
    },
    14: {
        "看SD15": "室内布局在；小物体糊",
        "看SD35": "更锐；颜色略偏",
        "questions_sd15_vs_sd35": [
            {"id": "14x1", "question": "Which recon keeps wall/furniture edges sharper?", "answer": "SD35"},
        ],
    },
    15: {
        "看SD15": "主体色块在",
        "看SD35": "细节稍多",
        "questions_sd15_vs_sd35": [],
    },
    16: {
        "看SD15": "猫/沙发在；纹理糊",
        "看SD35": "猫形更清；背景噪点",
        "questions_sd15_vs_sd35": [
            {"id": "16x1", "question": "Which recon preserves the cat silhouette more clearly?", "answer": "SD35"},
        ],
    },
    17: {
        "看SD15": "两人+球在；号码/文字丢",
        "看SD35": "动作更清；号码仍不可读",
        "questions_sd15_vs_sd35": [
            {"id": "17x1", "question": "Which recon makes the players' poses easier to recognize?", "answer": "SD35"},
        ],
    },
    18: {
        "看SD15": "四步结构在；橙框变褐紫；箭头→锯齿；文字全糊",
        "看SD35": "橙框颜色恢复；连接符变菱形非箭头；文字像清晰假字",
        "questions_没重建出来": [
            {"id": "18g", "question": "What shape connects the four boxes in SD35?", "answer": "diamonds (not arrows like GT)"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "18x1", "question": "Which recon preserves the orange box color better?", "answer": "SD35"},
            {"id": "18x2", "question": "Can the name 'John Henry' be read in either recon?", "answer": "no"},
            {"id": "18x3", "question": "Which recon replaces downward arrows with diamond connectors?", "answer": "SD35"},
        ],
    },
    19: {
        "看SD15": "五人+粉衣在；脸糊",
        "看SD35": "人数色块在；脸仍软",
        "questions_sd15_vs_sd35": [
            {"id": "19x1", "question": "Which recon keeps the pink-shirt child color clearer?", "answer": "SD35"},
        ],
    },
    20: {
        "看SD15": "棒球姿态在；下半幅黑边；队名/号码糊；手糊",
        "看SD35": "更锐但手/头盔解剖畸变；球棒木纹变波浪线",
        "questions_没重建出来": [
            {"id": "20h", "question": "Is the bottom of the frame cropped to solid black?", "answer": "yes in SD15 only"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "20x1", "question": "Which recon distorts the batter's hands into blob-like shapes?", "answer": "SD35"},
            {"id": "20x2", "question": "Which recon adds a black bar at the bottom?", "answer": "SD15"},
        ],
    },
    21: {
        "看SD15": "雏菊结构在；背景变 khaki；stay/positive 仍大致可读但变粗",
        "看SD35": "背景绿更接近 GT；文字完全乱码",
        "questions_sd15_vs_sd35": [
            {"id": "21x1", "question": "Which recon keeps the bright green background?", "answer": "SD35"},
            {"id": "21x2", "question": "Which recon keeps the words 'stay' and 'positive' more readable?", "answer": "SD15"},
            {"id": "21x3", "question": "In SD35, are the smiley eyes centered in the yellow circle?", "answer": "no, pushed to the top"},
        ],
    },
    22: {
        "看SD15": "国画风在；人物变矮胖；文字→伪汉字",
        "看SD35": "人物后仰模糊；文字列→墨渍；线条更糊",
        "questions_没重建出来": [
            {"id": "22h", "question": "Is the figure standing upright as in GT?", "answer": "yes in GT; squat/wide in SD15; leaning back in SD35"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "22x1", "question": "Which recon keeps sharper clothing fold outlines?", "answer": "SD15"},
            {"id": "22x2", "question": "Which recon degrades the vertical calligraphy into smudges more severely?", "answer": "SD35"},
        ],
    },
    23: {
        "看SD15": "战车+双马极糊；拖尸不可辨",
        "看SD35": "细节显著更多；城墙上空出现亮爆炸/光斑伪影；拖尸略可见",
        "questions_没重建出来": [
            {"id": "23h", "question": "Is there a bright explosion or light burst in the sky above the wall?", "answer": "yes in SD35 only (artifact)"},
        ],
        "questions_sd15_vs_sd35": [
            {"id": "23x1", "question": "Which recon shows the dragged body behind the chariot more clearly?", "answer": "SD35"},
            {"id": "23x2", "question": "Which recon is overall blurrier?", "answer": "SD15"},
        ],
    },
}


def merge_questions(existing: list, new: list) -> list:
    by_id = {q["id"]: q for q in existing}
    for q in new:
        by_id[q["id"]] = q
    return list(by_id.values())


def main():
    doc = json.load(open(EDIT, encoding="utf-8"))
    doc["_说明"] = [
        "【你改这里】每张图自己改 questions_* 和看重建图/SD15/SD35 备注。",
        "流程：1) 打开 compare3（GT|SD15|SD35）；2) 判断各重建保住了/丢了什么；3) 设计代表性问题；4) 跑脚本时只用【原图 gt_image】问 xomni 模型。",
        "questions_重建出来了 / questions_没重建出来：相对「重建可见性」设计的 decodability probe。",
        "questions_sd15_vs_sd35：专门区分 SD15 与 SD35 重建差异（看 compare3 中间/右栏）。",
        "不要改 i / gt_image / compare3 路径，除非你换图。",
    ]
    for item in doc["items"]:
        i = item["i"]
        stem = Path(item["gt_image"]).stem
        item["compare3"] = str(
            ROOT / f"outputs/recon_pairs_random/compare3/{i:02d}_{stem}_compare3.png"
        )
        item.pop("compare2", None)
        patch = PATCHES.get(i, {})
        for k, v in patch.items():
            if k.startswith("questions_") and isinstance(v, list):
                item[k] = merge_questions(item.get(k, []), v)
            else:
                item[k] = v
        if "看SD15" not in item:
            item.setdefault("看SD15", item.get("看重建图_保住了", ""))
        if "看SD35" not in item:
            item.setdefault("看SD35", "(待补)")
        item.setdefault("questions_sd15_vs_sd35", [])

    json.dump(doc, open(EDIT, "w"), ensure_ascii=False, indent=2)
    print(f"=> patched {EDIT}")


if __name__ == "__main__":
    main()
