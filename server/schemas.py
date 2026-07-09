from pydantic import BaseModel, Field, field_validator


class HousePricePredictionRequest(BaseModel):
    location: str = Field(..., description="Location of the property in Bengaluru")
    sqft: float = Field(..., description="Total square feet area")
    bhk: int = Field(..., description="Number of bedrooms (BHK)")
    bath: int = Field(..., description="Number of bathrooms")

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str) -> str:
        val = value.strip()
        if not val:
            raise ValueError("Location cannot be empty or only whitespace")
        return val

    @field_validator("sqft")
    @classmethod
    def validate_sqft(cls, value: float) -> float:
        if value < 100.0 or value > 25000.0:
            raise ValueError("Square footage must be between 100 and 25000")
        return value

    @field_validator("bhk")
    @classmethod
    def validate_bhk(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("BHK must be between 1 and 20")
        return value

    @field_validator("bath")
    @classmethod
    def validate_bath(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("Bathrooms must be between 1 and 20")
        return value


class SHAPFeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP value contribution to prediction")


class HousePricePredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price in Lakhs INR")
    shap_contributions: list[SHAPFeatureContribution] = Field(
        ..., description="Top contributing features explaining prediction"
    )
