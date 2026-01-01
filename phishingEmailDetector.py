import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple
import json


class MarkovEmailDetector:
    """
    A Markov Chain-based email classifier that detects phishing attempts
    by learning language patterns from legitimate and phishing emails.
    """
    
    def __init__(self, order: int = 2):
        """
        Initialize the email detector.
        
        Args:
            order: The order of the Markov chain (n-gram size)
        """
        self.order = order
        
        # TWO SEPARATE BRAINS: one learns "normal emails", one learns "scam emails"
        # Each brain remembers: "after seeing words X and Y, what word usually comes next?"
        self.legitimate_chain = defaultdict(lambda: defaultdict(int))
        self.phishing_chain = defaultdict(lambda: defaultdict(int))
        
        # Keep count of how many emails we've seen
        self.legitimate_count = 0
        self.phishing_count = 0
        self.trained = False
        
    def preprocess_email(self, text: str) -> str:
        """
        Clean up the email so we focus on patterns, not specific details.
        Example: "Visit http://evil.com" becomes "Visit <URL>"
        """
        text = text.lower()
        
        # Replace all URLs with <URL> tag (so we look for "click <URL>" pattern, not the specific URL)
        text = re.sub(r'http[s]?://\S+', '<URL>', text)
        
        # Replace all emails with <EMAIL> tag
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
        
        # Replace money amounts like "$500" with <MONEY>
        text = re.sub(r'\$\d+(?:\.\d+)?', '<MONEY>', text)
        
        # Replace all numbers with <NUMBER>
        text = re.sub(r'\b\d+\b', '<NUMBER>', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Break text into pieces (words and punctuation).
        "Hello, world!" becomes ["hello", ",", "world", "!"]
        """
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def build_chain(self, tokens: List[str], chain: Dict):
        """
        Build the Markov chain by counting word patterns.
        
        How it works:
        - Look at every pair of words (if order=2)
        - Count what word comes after that pair
        
        Example with "I am happy. I am sad":
        State ("i", "am") → "happy" appears 1 time
        State ("i", "am") → "sad" appears 1 time
        """
        for i in range(len(tokens) - self.order):
            # Get the current "state" (pair of words if order=2)
            state = tuple(tokens[i:i + self.order])
            
            # Get the next word after this state
            next_token = tokens[i + self.order]
            
            # Count it! (this is the "learning" part)
            chain[state][next_token] += 1
    
    def train_legitimate(self, emails: List[str]):
        """
        Teach the detector what NORMAL emails look like.
        """
        for email in emails:
            processed = self.preprocess_email(email)
            tokens = self.tokenize(processed)
            # Feed patterns into the "good email" brain
            self.build_chain(tokens, self.legitimate_chain)
            self.legitimate_count += 1
        
        print(f"Trained on {self.legitimate_count} legitimate emails")
        print(f"  Unique patterns: {len(self.legitimate_chain)}")
    
    def train_phishing(self, emails: List[str]):
        """
        Teach the detector what SCAM emails look like.
        """
        for email in emails:
            processed = self.preprocess_email(email)
            tokens = self.tokenize(processed)
            # Feed patterns into the "scam email" brain
            self.build_chain(tokens, self.phishing_chain)
            self.phishing_count += 1
        
        print(f"Trained on {self.phishing_count} phishing emails")
        print(f"  Unique patterns: {len(self.phishing_chain)}")
        self.trained = True
    
    def calculate_log_probability(self, tokens: List[str], chain: Dict) -> float:
        """
        Calculate how "likely" this email is according to one of our brains.
        
        Think of it like: "Does this email sound like the emails you trained me on?"
        Higher score = more similar
        Lower score = less similar
        """
        if len(tokens) < self.order + 1:
            return float('-inf')  # Not enough data
        
        log_prob = 0.0  # Start with 0 score
        
        # Look at every word pattern in the email
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i:i + self.order])
            next_token = tokens[i + self.order]
            
            # Check: "Have I seen this pattern before?"
            if state in chain and next_token in chain[state]:
                # YES! Calculate how common this pattern is
                total = sum(chain[state].values())  # How many times did we see this state?
                prob = chain[state][next_token] / total  # What % of time did this word follow?
                log_prob += math.log(prob)  # Add to score (using log for math reasons)
            else:
                # NEVER seen this pattern = very suspicious
                log_prob += math.log(1e-10)  # Tiny penalty
        
        return log_prob
    
    def detect(self, email: str) -> Dict:
        """
        THE MAIN DETECTION LOGIC - This is where the magic happens!
        
        We ask both brains: "Does this email look like YOUR training data?"
        - Normal brain says: "This looks 30% like normal emails"
        - Scam brain says: "This looks 70% like scam emails"
        Result: PROBABLY A SCAM!
        """
        if not self.trained:
            return {"error": "Model not trained"}
        
        # Clean up the email
        processed = self.preprocess_email(email)
        tokens = self.tokenize(processed)
        
        # STEP 1: Ask the "normal email" brain for a score
        legit_score = self.calculate_log_probability(tokens, self.legitimate_chain)
        
        # STEP 2: Ask the "scam email" brain for a score
        phishing_score = self.calculate_log_probability(tokens, self.phishing_chain)
        
        # STEP 3: Compare the scores and decide
        if legit_score == float('-inf') and phishing_score == float('-inf'):
            # Email is totally unlike anything we've seen
            confidence = 0.5
            is_phishing = False
        else:
            # Convert scores to percentages using softmax math
            # This turns two scores into "X% chance of phishing"
            max_score = max(legit_score, phishing_score)
            legit_exp = math.exp(legit_score - max_score)
            phishing_exp = math.exp(phishing_score - max_score)
            total = legit_exp + phishing_exp
            
            phishing_prob = phishing_exp / total  # % chance of being phishing
            confidence = phishing_prob
            is_phishing = phishing_score > legit_score  # Whichever score is higher wins
        
        # STEP 4: Also check for obvious red flags (bonus detection)
        features = self.extract_features(email)
        
        return {
            "is_phishing": is_phishing,
            "confidence": confidence * 100,
            "legitimate_score": legit_score,
            "phishing_score": phishing_score,
            "suspicious_features": features,
            "verdict": "PHISHING" if is_phishing else "LEGITIMATE"
        }
    
    def extract_features(self, email: str) -> List[str]:
        """
        Look for OBVIOUS scam indicators (backup detection method).
        
        Like a checklist:
        ☑ Says "URGENT"? → Scam tactic
        ☑ Has sketchy .tk domain? → Common scam domain
        ☑ Asks for password? → RED FLAG
        """
        features = []
        text_lower = email.lower()
        
        # TACTIC 1: Urgency words (scammers want you to panic and click fast)
        urgency_words = ['urgent', 'immediate', 'act now', 'expire', 'suspended', 
                        'verify', 'confirm', 'update', 'security alert']
        
        for word in urgency_words:
            if word in text_lower:
                features.append(f"Urgency keyword: '{word}'")
        
        # TACTIC 2: Too many links (suspicious)
        urls = re.findall(r'http[s]?://\S+', email)
        if len(urls) > 2:
            features.append(f"Multiple URLs ({len(urls)})")
        
        # TACTIC 3: Sketchy domains (free domains scammers love)
        suspicious_domains = ['.tk', '.ml', '.ga', 'bit.ly', 'tinyurl']
        for url in urls:
            for domain in suspicious_domains:
                if domain in url:
                    features.append(f"Suspicious domain: {domain}")
        
        # TACTIC 4: Asking for sensitive info (HUGE red flag)
        personal_info = ['password', 'ssn', 'social security', 'credit card', 
                        'account number', 'pin', 'banking']
        for term in personal_info:
            if term in text_lower:
                features.append(f"Requests sensitive info: '{term}'")
        
        # TACTIC 5: Generic greeting (real companies use your name)
        if re.search(r'dear (customer|user|member)', text_lower):
            features.append("Generic greeting (no name)")
        
        # TACTIC 6: Common typos (scammers are sloppy)
        misspellings = ['recieve', 'seperate', 'occured', 'privilage']
        for word in misspellings:
            if word in text_lower:
                features.append(f"Possible typo: '{word}'")
        
        return features
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            "legitimate_emails": self.legitimate_count,
            "phishing_emails": self.phishing_count,
            "legitimate_patterns": len(self.legitimate_chain),
            "phishing_patterns": len(self.phishing_chain),
            "markov_order": self.order
        }


# Sample training data
LEGITIMATE_EMAILS = [
    """
    Hi John,
    
    Thanks for your email yesterday. I've reviewed the quarterly report and 
    everything looks good. The team did an excellent job meeting our targets.
    
    Let's schedule a meeting next Tuesday to discuss the upcoming project timeline.
    I'm available after 2 PM.
    
    Best regards,
    Sarah
    """,
    
    """
    Dear Customer,
    
    Your order #12345 has been shipped and should arrive within 3-5 business days.
    You can track your package using the tracking number: 1Z999AA10123456784
    
    If you have any questions, please contact our customer service team.
    
    Thank you for your purchase!
    Amazon Customer Service
    """,
    
    """
    Hello Team,
    
    Please remember that our monthly all-hands meeting is scheduled for 
    Friday at 10 AM in Conference Room B. We'll be discussing Q3 goals 
    and celebrating recent wins.
    
    See you there!
    Mike
    """,
    
    """
    Hi there,
    
    Your GitHub repository has received a new pull request from contributor
    jane_doe. Please review the changes at your convenience.
    
    View Pull Request: https://github.com/yourrepo/pull/42
    
    GitHub
    """
]

PHISHING_EMAILS = [
    """
    URGENT: Your Account Has Been Suspended!
    
    Dear Customer,
    
    We have detected unusual activity on your account. Your account will be 
    permanently closed within 24 hours unless you verify your identity immediately.
    
    Click here to verify now: http://secure-banking.tk/verify
    
    You must update your password, social security number, and account number 
    to restore access.
    
    Act now or lose access forever!
    
    Security Team
    """,
    
    """
    Congratulations! You've Won $1,000,000!
    
    Dear Lucky Winner,
    
    You have been selected to recieve one million dollars in our lottery.
    To claim your prize, you must act immediately and provide your banking 
    details.
    
    Click here: http://bit.ly/claim-prize-now
    
    Provide your account number, PIN, and SSN to verify your identity.
    
    This offer expires in 24 hours!
    """,
    
    """
    Security Alert: Verify Your Account Now
    
    Dear User,
    
    Your PayPal account has been limited due to suspicious activity. 
    Confirm your identity immediately or your account will be closed.
    
    Update your credit card information here: http://paypal-secure.ml/update
    
    Failure to comply will result in permanent suspension.
    
    PayPal Security
    """,
    
    """
    URGENT: IRS Tax Refund
    
    Dear Taxpayer,
    
    You are eligible for a tax refund of $2,847. To process your refund,
    we need you to verify your social security number and banking details.
    
    Click here to claim: http://irs-refund.ga/claim
    
    This link expires in 48 hours. Act now to recieve your refund.
    
    Internal Revenue Service
    """
]

TEST_EMAILS = [
    """
    Hi Sarah,
    
    Thanks for the meeting yesterday. I've attached the document you requested.
    Let me know if you need anything else.
    
    Best,
    Tom
    """,
    
    """
    URGENT ACTION REQUIRED!
    
    Your Netflix account will be suspended unless you update your payment 
    information immediately. Click here to verify: http://netflix-billing.tk
    
    Update your credit card now or lose access!
    """
]


def main():
    """Main demonstration of phishing detection."""
    print("=" * 70)
    print("Markov Chain Phishing Email Detector")
    print("=" * 70)
    print()
    
    # Initialize and train detector
    print("Training detector...\n")
    detector = MarkovEmailDetector(order=2)
    
    # TRAINING PHASE: Teach it what normal vs scam emails look like
    detector.train_legitimate(LEGITIMATE_EMAILS)
    detector.train_phishing(PHISHING_EMAILS)
    
    print()
    stats = detector.get_statistics()
    print("Training Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Testing Detection")
    print("=" * 70)
    
    # TESTING PHASE: See if it can correctly identify test emails
    for i, email in enumerate(TEST_EMAILS, 1):
        print(f"\n--- Test Email #{i} ---")
        print(email[:100] + "..." if len(email) > 100 else email)
        print("\n" + "-" * 70)
        
        result = detector.detect(email)
        
        print(f"VERDICT: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Legitimate Score: {result['legitimate_score']:.2f}")
        print(f"Phishing Score: {result['phishing_score']:.2f}")
        
        if result['suspicious_features']:
            print("\nSuspicious Features Detected:")
            for feature in result['suspicious_features']:
                print(f"  ⚠️  {feature}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 70)
    
    while True:
        try:
            print("\nPaste email text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    if lines:
                        break
            
            if not lines:
                continue
            
            email = '\n'.join(lines)
            
            if email.lower() == 'quit':
                break
            
            result = detector.detect(email)
            
            print("\n" + "=" * 70)
            print(f"🔍 ANALYSIS RESULTS")
            print("=" * 70)
            print(f"\n{'🚨 PHISHING DETECTED' if result['is_phishing'] else '✅ LEGITIMATE EMAIL'}")
            print(f"Confidence: {result['confidence']:.1f}%")
            
            if result['suspicious_features']:
                print(f"\nSuspicious Features ({len(result['suspicious_features'])}):")
                for feature in result['suspicious_features']:
                    print(f"  ⚠️  {feature}")
            else:
                print("\n✓ No suspicious features detected")
            
            print("\n" + "=" * 70)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()