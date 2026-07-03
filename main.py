"""
Entry point for the Mood Machine rule based mood analyzer.
"""

from typing import List

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS


def evaluate_rule_based(posts: List[str], labels: List[str]) -> float:
    """
    Evaluate the rule based MoodAnalyzer on a labeled dataset.

    Prints each text with its predicted label and the true label,
    then returns the overall accuracy as a float between 0 and 1.
    """
    analyzer = MoodAnalyzer()
    correct = 0
    total = len(posts)

    print("=== Rule Based Evaluation on SAMPLE_POSTS ===")
    for text, true_label in zip(posts, labels):
        predicted_label = analyzer.predict_label(text)
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        # If you implement explain(), you can uncomment these lines:
        # reason = analyzer.explain(text)
        # print(f'"{text}" -> predicted={predicted_label}, true={true_label} ({reason})')

        reason = analyzer.explain(text)
        print(f'"{text}" -> predicted={predicted_label}, true={true_label} ({reason})')

    if total == 0:
        print("\nNo labeled examples to evaluate.")
        return 0.0

    accuracy = correct / total
    print(f"\nRule based accuracy on SAMPLE_POSTS: {accuracy:.2f}")
    return accuracy


def run_batch_demo() -> None:
    """
    Run the MoodAnalyzer on the sample posts and print predictions only.

    This is a quick way to see how your rules behave without comparing
    to the true labels.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Batch Demo on SAMPLE_POSTS (rule based) ===")
    for text in SAMPLE_POSTS:
        label = analyzer.predict_label(text)
        # If explain() is implemented, show a short explanation.
        # reason = analyzer.explain(text)
        # print(f'"{text}" -> {label} ({reason})')
        reason = analyzer.explain(text)
        print(f'"{text}" -> {label} ({reason})')


def run_interactive_loop() -> None:
    """
    Let the user type their own sentences and see the predicted mood.

    Type 'quit' or press Enter on an empty line to exit.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Interactive Mood Machine (rule based) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the Mood Machine.")
            break

        label = analyzer.predict_label(user_input)
        # If explain() is implemented, you can include an explanation:
        # reason = analyzer.explain(user_input)
        # print(f"Model: {label} ({reason})")
        reason = analyzer.explain(user_input)
        print(f"Model: {label} ({reason})")


from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")

    # 1. Split the data! (80% for training, 20% for testing)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        SAMPLE_POSTS, TRUE_LABELS, test_size=0.2, random_state=42
    )

    # 2. Train the model ONLY on the training data.
    vectorizer, model = train_ml_model(train_texts, train_labels)

    # 3. Evaluate the model ONLY on the unseen testing data.
    print("=== Evaluating on UNSEEN Test Data ===")
    evaluate_on_dataset(test_texts, test_labels, vectorizer, model)

    # Let the user try their own examples.
    run_interactive_loop(vectorizer, model)