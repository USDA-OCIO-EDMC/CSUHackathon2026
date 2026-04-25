# HackathonSp26

## Run
python main.py

## Formula for prediction of corn yield

The multi-linear regression equation for grain yield during the silking stage (SS: 15 days pre- to 15 days post-silking) is:

    y = −2.256x₁ + 10.351x₂ − 76.091x₃ + 2.785x₄ + 2940.349

    R² = 0.474, p < 0.05

Where:

    y = Grain yield (kg ha⁻¹)
    x₁ = Precipitation (mm)
    x₂ = Solar radiation / SR (MJ m⁻²)
    x₃ = Diurnal temperature range / DTR (°C)
    x₄ = Extreme degree days / EDD > 32°C

Grain Yield Formula (SH Stage)

For the silking-to-harvest (SH) stage, a slightly simpler equation is provided with a better fit:

    y = 0.084x₁ + 12.559x₂ − 112.215x₃ + 1933.795

    R² = 0.619, p < 0.05

Where:

    y = Grain yield (kg ha⁻¹)
    x₁ = Precipitation (mm)
    x₂ = Solar radiation / SR (MJ m⁻²)
    x₃ = Diurnal temperature range / DTR (°C)

Key Takeaways

    The SH stage formula is the simpler and better-fitting of the two (R² = 0.619 vs. 0.474), requiring only 3 variables: precipitation, solar radiation, and diurnal temperature range.
    Solar radiation is the dominant factor, explaining 63.1% (SS stage) and 86.4% (SH stage) of yield variability.
    These formulas are based on summer maize in the Anhui province of China, so they are most applicable to similar humid, transitional climate zones.