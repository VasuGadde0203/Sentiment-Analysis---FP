import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis: Custom LSTM vs. AWD-LSTM (ULMFiT) vs. BERT"
    )
    parser.add_argument(
        "model",
        choices=["lstm", "ulmfit", "bert", "compare", "all"],
        help="Which model to train, or 'compare' to build the comparison report, or 'all' to run everything in sequence.",
    )
    args = parser.parse_args()

    if args.model in ("lstm", "all"):
        from src.train_custom import train

        train()

    if args.model in ("ulmfit", "all"):
        from src.ulmfit_train import train_ulmfit

        train_ulmfit()

    if args.model in ("bert", "all"):
        from src.bert_train import train_bert

        train_bert()

    if args.model in ("compare", "all"):
        from src.compare import compare

        compare()


if __name__ == "__main__":
    main()
