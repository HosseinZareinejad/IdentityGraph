from hazm import Normalizer

class TextNormalizer:
    def __init__(self):
        self.normalizer = Normalizer()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return self.normalizer.normalize(str(text).strip())

    def build_composite_text(self, name: str, city: str, bio: str) -> str:
        n_name = self.normalize(name) or "نامشخص"
        n_city = self.normalize(city) or "نامشخص"
        n_bio = self.normalize(bio) or "بدون بیوگرافی"
        return f"[نام: {n_name}] [شهر: {n_city}] [بیوگرافی: {n_bio}]"

# Global instance for easy import
normalizer = TextNormalizer()
