from imageai.Prediction.Custom import CustomImagePrediction
import os

predictor = CustomImagePrediction()
predictor.setModelPath(model_path="model_ex-031_acc-0.999333.h5")
predictor.setJsonPath(model_json="model_class.json")
predictor.loadFullModel(num_objects=5)

directory = "C:/Programs/Java/Computer-Vision-Team19/testing/"

ints = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        ints.append(int(filename.split(".", 1)[0]))
    else:
        print("Ignoring file named " + filename)

ints.sort()
print()

log = open("run3.txt", "w")
for i in ints:
    filename = str(i) + ".jpg"
    path = os.path.join(directory, filename)
    prediction, probability = predictor.predictImage(image_input=path, result_count=1)
    line = filename + " " + str(prediction[0]) + " (" + str(probability[0]) + ")"
    print(line)
    log.write(line + "\n")
log.close()
