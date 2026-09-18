"""
Model registry — shared dictionary holding loaded models across requests.
Populated during application lifespan startup in app.main.
"""

models: dict = {}
