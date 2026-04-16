import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os

def create_dashboard(json_path, output_image):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    log_history = data['log_history']
    report = data['classification_report']
    cm = np.array(data['confusion_matrix'])
    metrics = data['metrics']

    # Set style
    sns.set_theme(style="white")
    plt.rcParams['font.family'] = 'Malgun Gothic' # Standard Korean font on Windows
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # 1. Confusion Matrix
    ax_cm = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=True)
    ax_cm.set_title("Confusion Matrix", fontsize=15, pad=15)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    # 2. Accuracy & Loss Curves
    # Extract logs
    steps = []
    loss = []
    eval_loss = []
    eval_acc = []
    
    for log in log_history:
        if 'loss' in log and 'step' in log:
            steps.append(log['step'])
            loss.append(log['loss'])
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
            eval_acc.append(log['eval_accuracy'])

    ax_curves = fig.add_subplot(gs[0, 1])
    ax_curves_loss = ax_curves.twinx()
    
    # Plot Accuracy
    p1, = ax_curves.plot(np.linspace(0, steps[-1], len(eval_acc)), eval_acc, label="Val Accuracy", color="orange", linewidth=2)
    ax_curves.set_ylabel("Accuracy")
    ax_curves.set_ylim(0.5, 1.05)
    
    # Plot Loss
    p2, = ax_curves_loss.plot(steps, loss, label="Train Loss", color="cornflowerblue", alpha=0.6)
    ax_curves_loss.set_ylabel("Loss")
    
    ax_curves.set_title("Accuracy & Loss Curves", fontsize=15, pad=15)
    ax_curves.legend(handles=[p1, p2], loc='lower right')

    # 3. Classification Report Summary
    ax_report = fig.add_subplot(gs[0, 2])
    ax_report.axis('off')
    report_text = "Classification Report:\n\n"
    for label, metrics_dict in report.items():
        if isinstance(metrics_dict, dict):
            report_text += f"Class {label}:\n"
            report_text += f"  Precision: {metrics_dict['precision']:.4f}\n"
            report_text += f"  Recall:    {metrics_dict['recall']:.4f}\n"
            report_text += f"  F1-Score:  {metrics_dict['f1-score']:.4f}\n\n"
    ax_report.text(0.1, 0.5, report_text, fontsize=12, verticalalignment='center', family='monospace')

    # 4. Final Dashboard (Bottom Row)
    # Background for summary
    ax_summary = fig.add_subplot(gs[1:, :])
    ax_summary.axis('off')
    
    # Draw a colored background box for the summary area
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_summary.transAxes, color='#f4f8f4', zorder=-1)
    ax_summary.add_patch(rect)
    
    # Summary texts
    test_acc = metrics.get('test_accuracy', metrics.get('eval_accuracy', 0))
    test_f1 = metrics.get('test_f1', metrics.get('eval_f1', 0))
    
    ax_summary.text(0.05, 0.7, "테스트 정확도", fontsize=20, weight='bold', color='#2e7d32')
    ax_summary.text(0.05, 0.4, f"{test_acc*100:.2f}%", fontsize=35, weight='bold', color='#2e7d32')
    ax_summary.text(0.05, 0.2, f"{len(cm.flatten())}개 테스트 샘플 기준", fontsize=12, color='gray')
    
    ax_summary.text(0.35, 0.7, "평균 F1-스코어", fontsize=20, weight='bold', color='#1565c0')
    ax_summary.text(0.35, 0.4, f"{test_f1:.2f}", fontsize=35, weight='bold', color='#1565c0')
    ax_summary.text(0.35, 0.2, "정밀도와 재현율의 조화평균", fontsize=12, color='gray')
    
    ax_summary.text(0.65, 0.7, "클래스별 성능 (F1)", fontsize=20, weight='bold', color='#37474f')
    f1_0 = report.get('0', {}).get('f1-score', 0)
    f1_1 = report.get('1', {}).get('f1-score', 0)
    ax_summary.text(0.65, 0.5, f"● 낚시성 (0): {f1_0:.2f}", fontsize=15)
    ax_summary.text(0.65, 0.3, f"● 일반   (1): {f1_1:.2f}", fontsize=15)

    # Vertical separators
    ax_summary.axvline(0.32, 0.1, 0.9, color='#cfd8dc', linewidth=1)
    ax_summary.axvline(0.62, 0.1, 0.9, color='#cfd8dc', linewidth=1)

    plt.savefig(output_image, dpi=150)
    print(f"Dashboard saved to {output_image}")

if __name__ == "__main__":
    create_dashboard("c:/LLM/training_results.json", "c:/LLM/training_dashboard.png")
