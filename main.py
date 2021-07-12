from trainmodel import *

colors = ["RGB", "YUV"]

for color in colors:
    trained_model = load_model(color)
    pred_test(21, color, trained_model)

