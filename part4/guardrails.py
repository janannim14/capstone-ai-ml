import re

def check_input_guardrails(user_input: str) -> tuple[bool, str]:
    """
    Scans incoming user queries for credential leak hazards or system prompt injection attempts.
    Returns: (is_safe, error_message)
    """
    # 🚨 1. Regex to check for potential API Keys or Secret Tokens
    # Detects common patterns like sk-..., AIzaSy..., or standard generic tokens
    api_key_pattern = re.compile(
        r'(?:sk-[a-zA-Z0-9]{32,48})|(?:AIzaSy[a-zA-Z0-9_-]{33})|(?:[a-zA-Z0-9_-]{40,})', 
        re.IGNORECASE
    )
    
    # 🚨 2. Regex to identify aggressive prompt injection attacks
    injection_pattern = re.compile(
        r'(ignore previous instructions|system prompt|reveal your system|override rules|act as a sudo)', 
        re.IGNORECASE
    )

    # Validate patterns
    if api_key_pattern.search(user_input) and any(keyword in user_input.lower() for keyword in ['key', 'token', 'secret', 'pass']):
        return False, "⚠️ Security Guardrail Triggered: Potential API key or credential leak detected in your input!"
        
    if injection_pattern.search(user_input):
        return False, "🚨 Safety Guardrail Triggered: Prompt injection or system override attempt blocked."

    return True, ""