"""Generate publication-ready figures and tables for the course report."""
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
CN = {"write":"书写", "read":"阅读", "lookup":"抬头听讲", "turn_head":"转头",
      "raise_hand":"举手", "stand":"站立", "discuss":"讨论"}


def setup():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.dpi": 160,
                         "savefig.dpi": 300, "axes.grid": True, "grid.alpha": .22})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def overall(df):
    labels = df.experiment.tolist()
    metrics = [("precision", "精确率 P"), ("recall", "召回率 R"),
               ("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95")]
    x = np.arange(len(labels)); width = .19
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for i, (col, label) in enumerate(metrics):
        bars = ax.bar(x + (i-1.5)*width, df[col], width, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.02); ax.set_ylabel("指标值")
    ax.set_title("不同模型在独立测试集上的检测性能对比", pad=38)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.005))
    save(fig, "01_overall_metrics.png")


def per_class(pc):
    piv = pc.pivot(index="name", columns="experiment", values="ap50").loc[list(CN)]
    piv.index = [CN[x] for x in piv.index]
    fig, ax = plt.subplots(figsize=(10, 5.5)); piv.plot.bar(ax=ax, width=.78)
    ax.set_ylim(0, 1.05); ax.set_xlabel("行为类别"); ax.set_ylabel("AP@0.5")
    ax.set_title("各行为类别AP@0.5对比"); ax.legend(title="模型", ncol=3)
    ax.tick_params(axis="x", rotation=0)
    save(fig, "02_per_class_ap50.png")

    base = pc[pc.experiment == "YOLO26n@640"].set_index("name")
    final = pc[pc.experiment == "YOLO26s@640"].set_index("name")
    gain = (final.ap50 - base.ap50).loc[list(CN)] * 100
    colors = ["#2f7ed8" if x >= 0 else "#d9534f" for x in gain]
    fig, ax = plt.subplots(figsize=(9, 4.8)); bars=ax.bar([CN[x] for x in gain.index], gain, color=colors)
    ax.axhline(0, color="black", lw=.8); ax.bar_label(bars, fmt="%+.2f", fontsize=9)
    ax.set_ylabel("AP@0.5变化（百分点）"); ax.set_title("最终模型相对基线模型的逐类别性能变化")
    save(fig, "03_final_vs_baseline_gain.png")


def efficiency(df):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sizes = 120 + df.parameters / 50000
    ax.scatter(df.model_inference_ms, df.map50_95, s=sizes, c=["#888", "#2f7ed8", "#f28e2b"], alpha=.85)
    for _, r in df.iterrows(): ax.annotate(r.experiment, (r.model_inference_ms, r.map50_95), xytext=(7,5), textcoords="offset points")
    ax.set_xlabel("单张图像模型推理时间（ms）"); ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("模型精度—速度权衡（气泡大小表示参数量）")
    save(fig, "04_accuracy_speed_tradeoff.png")


def training_curves():
    specs=[("baseline_yolo26n","YOLO26n@640"),("experiment_yolo26s_640","YOLO26s@640"),("experiment_yolo26n_960","YOLO26n@960")]
    fig, axes=plt.subplots(1,2,figsize=(11,4.4))
    for folder,label in specs:
        d=pd.read_csv(ROOT/"experiments"/folder/"results.csv"); d.columns=[x.strip() for x in d.columns]
        epoch=d["epoch"]+1
        axes[0].plot(epoch,d["metrics/mAP50(B)"],label=label,lw=1.6)
        axes[1].plot(epoch,d["metrics/mAP50-95(B)"],label=label,lw=1.6)
    for ax,title in zip(axes,["验证集 mAP@0.5","验证集 mAP@0.5:0.95"]):
        ax.set_xlabel("训练轮次"); ax.set_ylabel("指标值"); ax.set_title(title); ax.legend()
    save(fig,"05_training_curves.png")


def flow_diagram():
    fig, ax=plt.subplots(figsize=(12,3.2)); ax.set_xlim(0,12); ax.set_ylim(0,3); ax.axis("off")
    nodes=[("课堂图像/视频",.2,"#d9edf7"),("抽帧与缩放",2.2,"#dff0d8"),("YOLO26s\n行为检测",4.2,"#ffe0b2"),("置信度筛选\n与NMS",6.2,"#f3e5f5"),("中文标注与\n行为统计",8.2,"#dcedc8"),("图片/视频\nCSV/JSON",10.2,"#ffcdd2")]
    for text,x,color in nodes:
        ax.add_patch(FancyBboxPatch((x,.9),1.55,1.15,boxstyle="round,pad=.08",fc=color,ec="#555",lw=1.2))
        ax.text(x+.775,1.475,text,ha="center",va="center",fontsize=10)
    for (_,x,_),(_,nx,_) in zip(nodes,nodes[1:]):
        ax.add_patch(FancyArrowPatch((x+1.57,1.475),(nx-.03,1.475),arrowstyle="-|>",mutation_scale=14,lw=1.3,color="#444"))
    ax.set_title("课堂学生行为识别系统总体流程",fontsize=14,pad=12)
    save(fig,"06_system_flow.png")


def collect_evidence():
    copies={
      ROOT/"experiments/experiment_yolo26s_640_test/confusion_matrix_normalized.png":"07_final_confusion_matrix_normalized.png",
      ROOT/"experiments/experiment_yolo26s_640_test/val_batch0_pred.jpg":"08_final_prediction_batch.jpg",
      ROOT/"outputs/predictions/stage9/image_0100_annotated.jpg":"09_system_demo.jpg",
      ROOT/"outputs/figures/dataset_analysis/class_distribution.png":"10_dataset_class_distribution.png",
      ROOT/"docs/statistics/split_class_distribution.png":"11_split_class_distribution.png"}
    for src,name in copies.items(): shutil.copy2(src,FIG/name)


def tables(df,pc):
    out=df.copy()
    out["parameters_M"]=out.parameters/1e6
    out["weight_MB"]=out.weight_mb
    cols=["experiment","imgsz","parameters_M","gflops","weight_MB","precision","recall","map50","map50_95","model_inference_ms"]
    out[cols].round(4).to_csv(TAB/"model_comparison.csv",index=False,encoding="utf-8-sig")
    p=pc.copy(); p["类别中文"]=p.name.map(CN)
    p[["experiment","class_id","类别中文","precision","recall","ap50","ap50_95"]].round(4).to_csv(TAB/"per_class_metrics.csv",index=False,encoding="utf-8-sig")
    split=json.loads((ROOT/"docs/statistics/scbehavior_split_summary.json").read_text(encoding="utf-8"))
    (TAB/"dataset_split_summary.json").write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    setup()
    df=pd.read_csv(ROOT/"outputs/reports/experiment_comparison/experiment_comparison.csv")
    pc=pd.read_csv(ROOT/"outputs/reports/experiment_comparison/per_class_comparison.csv")
    overall(df); per_class(pc); efficiency(df); training_curves(); flow_diagram(); collect_evidence(); tables(df,pc)
    manifest={"figures":sorted(x.name for x in FIG.iterdir()),"tables":sorted(x.name for x in TAB.iterdir())}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
