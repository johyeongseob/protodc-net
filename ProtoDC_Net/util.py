import numpy as np

def calculate_accuracies(confusion_matrix):
    num_classes = len(confusion_matrix)
    class_accuracies = []
    class_f1s = []

    # 클래스별 accuracy 및 F1-score 계산
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = np.sum(confusion_matrix) - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total = np.sum(confusion_matrix[i, :])
        acc = TP / total if total > 0 else 0.0

        class_accuracies.append((TP, total, acc))
        class_f1s.append(f1)

    # 전체 Accuracy
    total_correct = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # macro F1-score
    macro_f1 = np.mean(class_f1s)

    # 출력
    accuracies_summary = ", ".join(
        [f"Class {i}: {acc[2]:.2%} ({int(acc[0])}/{int(acc[1])})" for i, acc in enumerate(class_accuracies)]
    )

    f1_summary = ", ".join(
        [f"Class {i}: {f1:.3f}" for i, f1 in enumerate(class_f1s)]
    )

    print(f"Accuracy: {accuracies_summary}, Total Accuracy: {total_accuracy:.2%}")
    # print(f"F1-score: {f1_summary}, Macro F1-score: {macro_f1:.3f}")
