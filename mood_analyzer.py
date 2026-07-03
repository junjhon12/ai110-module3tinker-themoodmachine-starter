import string
from typing import List, Dict, Tuple, Optional
from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

class MoodAnalyzer:
    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        cleaned = text.strip().lower()
        for p in string.punctuation:
            cleaned = cleaned.replace(p, "")

        return cleaned.split()

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        tokens = self.preprocess(text)
        score = 0

        # Reset hits for each new text
        self.current_positive_hits = []
        self.current_negative_hits = []

        negation_words = {"not", "never", "no", "dont", "cant", "isnt", "didnt"}
        is_negated = False

        for token in tokens:
            if token in negation_words:
                is_negated = True
                continue

            multiplier = -1 if is_negated else 1

            if token in self.positive_words:
                score += (1 * multiplier)
                # Track the hit, including the negation context if applicable
                self.current_positive_hits.append(f"not {token}" if is_negated else token)
                is_negated = False
            elif token in self.negative_words:
                score -= (1 * multiplier)
                # Track the hit, including the negation context if applicable
                self.current_negative_hits.append(f"not {token}" if is_negated else token)
                is_negated = False

        # Store the final score so explain() can use it
        self.current_score = score
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        score = self.score_text(text)
        
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        # Calculate scores and populate the instance variables
        self.score_text(text)
        
        return (
            f"Score = {self.current_score} "
            f"(positive context: {self.current_positive_hits or '[]'}, "
            f"negative context: {self.current_negative_hits or '[]'})"
        )