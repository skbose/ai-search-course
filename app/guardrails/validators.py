from guardrails.validator_base import Validator, PassResult, FailResult, register_validator
import re

# 1. Abusive/offensive language filter
@register_validator(name="no_offensive_language", data_type="string")
class NoOffensiveLanguage(Validator):
    offensive_words = ["idiots", "stupid", "dumb", "fool", "moron"]

    def validate(self, value, metadata={}):
        if not isinstance(value, str) or not value.strip():
            return PassResult()  # ignore empty or non-string
        if any(word in value.lower() for word in self.offensive_words):
            return FailResult(error_message="Response contains offensive/abusive language.")
        return PassResult()


# 2. Personal info blocker
@register_validator(name="no_personal_info", data_type="string")
class NoPersonalInfo(Validator):
    patterns = [
        r"\b\d{10}\b",                     # phone numbers
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",  # emails
        r"\b\d{16}\b"                      # credit card numbers
    ]

    def validate(self, value, metadata={}):
        if not isinstance(value, str) or not value.strip():
            return PassResult()
        for pattern in self.patterns:
            if re.search(pattern, value):
                return FailResult(error_message="Response contains personal information.")
        return PassResult()


# 3. Verbosity control
@register_validator(name="no_verbosity_spam", data_type="string")
class NoVerbositySpam(Validator):
    def validate(self, value, metadata={}):
        if not isinstance(value, str):
            return PassResult()
        if len(value.split()) > 80:
            return FailResult(error_message="Response is too verbose or rambling.")
        return PassResult()


# 4. Must end with polite phrase
@register_validator(name="ends_with_thanks", data_type="string")
class EndsWithThanks(Validator):
    def validate(self, value, metadata={}):
        if not isinstance(value, str):
            return PassResult()
        if not value.strip().endswith("thanks for asking!"):
            return FailResult(error_message="Response must end with 'thanks for asking!'")
        return PassResult()


# 5. Prevent false claims (hard-coded check for example)
@register_validator(name="no_false_claims", data_type="string")
class NoFalseClaims(Validator):
    forbidden_claims = [
        "prime minister of india is",
    ]

    def validate(self, value, metadata={}):
        if not isinstance(value, str):
            return PassResult()
        if any(claim in value.lower() for claim in self.forbidden_claims):
            return FailResult(error_message="Response contains a potential false claim.")
        return PassResult()


# 6. Repetition control
@register_validator(name="no_repetition", data_type="string")
class NoRepetition(Validator):
    def __init__(self, max_repeats: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_repeats = max_repeats

    def validate(self, value, metadata={}):
        if not isinstance(value, str):
            return PassResult()
        words = value.split()
        for i in range(len(words) - self.max_repeats + 1):
            if all(w.lower() == words[i].lower() for w in words[i:i+self.max_repeats]):
                return FailResult(
                    error_message=f"Word '{words[i]}' is repeated more than {self.max_repeats} times consecutively."
                )
        return PassResult()
