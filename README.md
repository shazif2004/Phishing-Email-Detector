# Phishing Email Detector

A tool that detects phishing emails using Markov chains. It learns patterns from normal and scam emails, then classifies new emails based on which patterns they match.

## How It Works

The detector trains two separate models - one on legitimate emails and one on phishing emails. Each model learns common word patterns like "click here to" or "verify your account". 

When you give it a new email, both models score how similar the email is to their training data. Whichever model scores higher wins. If the phishing model scores higher, the email is flagged as a scam.

It also checks for obvious red flags like urgency words, suspicious domains, and requests for passwords.

## Usage
```python
detector = MarkovEmailDetector(order=2)
detector.train_legitimate(normal_emails)
detector.train_phishing(scam_emails)

result = detector.detect(test_email)
print(f"{result['verdict']} - {result['confidence']:.1f}% confidence")
```

## What You Get

The detector tells you:
- Is this phishing or legitimate
- How confident it is (percentage)
- Which suspicious features it found
- Scores from both models for comparison

## Why This Project

This shows how simple probability can solve real problems. Phishing costs billions annually and this demonstrates a practical ML solution using only basic statistics. No complex libraries or neural networks needed.

The code is clean, well-documented, and easy to understand. Good for learning how text classification works or as a portfolio piece.

## Limitations

Needs training data to work. Cannot detect completely new phishing tactics. Does not check email headers or sender reputation. Works best with 100+ examples of each type.

## Requirements

Python 3.6+ only. No external dependencies.
