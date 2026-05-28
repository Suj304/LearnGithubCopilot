import PyPDF2
import re
import sys
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        with open(pdf_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        sys.exit(1)

def extract_key_terms(text, top_n=15):
    """Extract key terms from text using frequency analysis."""
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    # Add policy-specific common words to ignore
    additional_stop = {'policy', 'coverage', 'insured', 'shall', 'may', 'will', 'also', 'must'}
    stop_words.update(additional_stop)
    
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 3]
    
    # Get most common terms
    term_freq = Counter(tokens)
    key_terms = term_freq.most_common(top_n)
    
    return [term for term, _ in key_terms]

def extract_exclusions(text):
    """Extract exclusion clauses from text."""
    exclusions = []
    
    # Common patterns for exclusions
    patterns = [
        r'(?:exclusion|excluded|does not cover|not covered)s?:?\s+([^\n.]+[.:])',
        r'(?:this policy does not cover|coverage does not include)s?:?\s+([^\n.]+[.:])',
        r'(?:excluded are|exclusions are)s?:?\s+([^\n.]+[.:])',
    ]
    
    # Look for explicit exclusion sections
    exclusion_section = re.search(
        r'exclusions?.*?(?=\n(?:coverage|benefits|conditions|general|additional)|\Z)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if exclusion_section:
        section_text = exclusion_section.group()
        # Extract bullet points or numbered items
        items = re.findall(r'(?:^|\n)\s*(?:[-•*]|\d+\.)\s+([^\n]+)', section_text, re.MULTILINE)
        exclusions.extend([item.strip() for item in items if item.strip()])
    
    # Also apply specific patterns
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        exclusions.extend([match.strip() for match in matches if match.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_exclusions = []
    for excl in exclusions:
        if excl.lower() not in seen:
            seen.add(excl.lower())
            unique_exclusions.append(excl)
    
    return unique_exclusions[:10]  # Return top 10

def extract_coverage_limits(text):
    """Extract coverage limits and amounts from text."""
    limits = []
    
    # Patterns to match coverage limits (e.g., $1,000, $100,000, USD 50,000)
    amount_patterns = [
        r'(?:limit|maximum|up to|capped at|not exceed)s?.*?\$[\d,]+(?:\.\d{2})?',
        r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per|per each|for each|annually|per year))?',
        r'(?:coverage limit|benefit limit|maximum benefit)s?.*?\$[\d,]+(?:\.\d{2})?',
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        limits.extend(matches)
    
    # Look for coverage limit sections
    limit_section = re.search(
        r'(?:coverage limits?|benefit limits?|policy limits?).*?(?=\n(?:exclusions|conditions|deductible)|\Z)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if limit_section:
        section_text = limit_section.group()
        items = re.findall(r'(?:^|\n)\s*(?:[-•*]|\d+\.)\s+([^\n]+)', section_text, re.MULTILINE)
        limits.extend([item.strip() for item in items if item.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_limits = []
    for limit in limits:
        if limit.lower() not in seen:
            seen.add(limit.lower())
            unique_limits.append(limit)
    
    return unique_limits[:10]  # Return top 10

def generate_summary(text):
    """Generate a bullet-point summary of the PDF content."""
    summary = []
    summary.append("=" * 70)
    summary.append("POLICY SUMMARY REPORT")
    summary.append("=" * 70)
    summary.append("")
    
    # Key Terms
    summary.append("KEY TERMS")
    summary.append("-" * 70)
    key_terms = extract_key_terms(text)
    for term in key_terms:
        summary.append(f"  • {term.capitalize()}")
    summary.append("")
    
    # Exclusions
    summary.append("EXCLUSIONS")
    summary.append("-" * 70)
    exclusions = extract_exclusions(text)
    if exclusions:
        for excl in exclusions:
            # Truncate long exclusions
            if len(excl) > 80:
                excl = excl[:77] + "..."
            summary.append(f"  • {excl}")
    else:
        summary.append("  • No specific exclusions found")
    summary.append("")
    
    # Coverage Limits
    summary.append("COVERAGE LIMITS")
    summary.append("-" * 70)
    limits = extract_coverage_limits(text)
    if limits:
        for limit in limits:
            # Truncate long limits
            if len(limit) > 80:
                limit = limit[:77] + "..."
            summary.append(f"  • {limit}")
    else:
        summary.append("  • No specific coverage limits found")
    summary.append("")
    
    summary.append("=" * 70)
    summary.append(f"Total text extracted: {len(text)} characters")
    summary.append("=" * 70)
    
    return "\n".join(summary)

def save_summary(summary, output_file=None):
    """Save summary to file or print to console."""
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(summary)
            print(f"Summary saved to: {output_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    else:
        print(summary)

def main():
    """Main function to orchestrate PDF processing."""
    if len(sys.argv) < 2:
        print("Usage: python pdf_policy_summary.py <pdf_file> [output_file]")
        print("Example: python pdf_policy_summary.py policy.pdf summary.txt")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Extract text from PDF
    print(f"Extracting text from {pdf_file}...")
    text = extract_text_from_pdf(pdf_file)
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(text)
    
    # Save or display summary
    save_summary(summary, output_file)

if __name__ == "__main__":
    main()
