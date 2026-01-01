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

## Limitations

Needs training data to work. Cannot detect completely new phishing tactics. Does not check email headers or sender reputation. Works best with 100+ examples of each type.


