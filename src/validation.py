
class ProductValidator:

    @staticmethod
    def validate(actual:str, expected:str) -> bool:
        return actual.lower() == expected.lower()