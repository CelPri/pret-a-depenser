from src.model.model import load_model

m = load_model()
print(type(m))

for attr in ["feature_names_in_", "n_features_in_"]:
    print(attr, getattr(m, attr, None))
