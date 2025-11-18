"""Dynamic YAML-based PII redaction system."""

import re
import yaml
import hashlib
from typing import Dict, List, Any

FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "VERBOSE": re.VERBOSE,
    "DOTALL": re.DOTALL,
}


class Redactor:
    def __init__(
        self,
        rules_file: str,
        strategy: str = "pseudonymize",
        mask_keep: int = 4,
    ):
        self.strategy = strategy
        self.mask_keep = mask_keep
        self.rules_file = rules_file
        self.rules = self._load_rules()
        self._pseudomap: Dict[str, str] = {}

    def _load_rules(self) -> List[Dict[str, Any]]:
        with open(self.rules_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rules = []

        for entry in cfg.get("rules", []):
            name = entry["name"]
            kind = entry.get("kind", name)
            pattern = entry["pattern"]
            flag_names = entry.get("flags", [])

            if isinstance(flag_names, str):
                flag_names = [flag_names]

            flags = 0

            for fl in flag_names:
                flags |= FLAG_MAP.get(fl.upper(), 0)
            rules.append(
                {
                    "name": name,
                    "kind": kind,
                    "pattern": re.compile(pattern, flags),
                }
            )

        return rules

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def _pseudonymize(self, kind: str, value: str) -> str:
        key = f"{kind}|{value}"

        if key not in self._pseudomap:
            self._pseudomap[key] = f"<{kind.upper()}:{self._hash(value)}>"

        return self._pseudomap[key]

    def _mask(self, value: str) -> str:
        k = self.mask_keep

        if len(value) <= k:
            return "*" * len(value)

        return "*" * (len(value) - k) + value[-k:]

    def _replace(self, kind: str, value: str) -> str:
        if self.strategy == "pseudonymize":
            return self._pseudonymize(kind, value)
        elif self.strategy == "mask":
            return self._mask(value)
        else:
            return f"<REDACTED:{kind}>"

    def redact(self, text: str) -> str:
        for rule in self.rules:
            kind = rule["kind"]
            pattern = rule["pattern"]
            text = pattern.sub(
                lambda m: self._replace(kind, m.group(0)), text
            )
        return text