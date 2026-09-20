import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, dataloader, device, criterion=None):
    """Runs the model over dataloader and returns a metrics dict.

    If `criterion` is provided, also computes the average loss (used for
    per-epoch validation-loss tracking during training).
    """
    model.eval()
    preds = []
    true = []
    total_loss = 0.0

    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels_device = labels.float().to(device)
            outputs = model(texts)

            if criterion is not None:
                total_loss += criterion(outputs, labels_device).item()

            predictions = (outputs > 0.5).long().cpu().numpy()
            preds.extend(predictions)
            true.extend(labels.numpy())

    metrics = {
        "accuracy": accuracy_score(true, preds),
        "precision": precision_score(true, preds, zero_division=0),
        "recall": recall_score(true, preds, zero_division=0),
        "f1": f1_score(true, preds, zero_division=0),
    }
    if criterion is not None:
        metrics["loss"] = total_loss / len(dataloader)

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} "
        f"| Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )
    return metrics
