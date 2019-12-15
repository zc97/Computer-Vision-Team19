from imageai.Prediction.Custom import CustomImagePrediction
import os

# Load the model to use for prediction
predictor = CustomImagePrediction()
predictor.setModelPath(model_path="model_ex-031_acc-0.999333.h5")
predictor.setJsonPath(model_json="model_class.json")
predictor.loadFullModel(num_objects=5)

directory = "C:/Programs/Java/Computer-Vision-Team19/testing/"

ints = []

# Put the names of the image files into the array of ints
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        ints.append(int(filename.split(".", 1)[0]))
    else:
        print("Ignoring file named " + filename)

# Sort the filenames and print a newline
ints.sort()
print()

# Log prediction results to a file
log = open("../run3.txt", "w")
for i in ints:
    filename = str(i) + ".jpg"
    path = os.path.join(directory, filename)
    prediction, probability = predictor.predictImage(image_input=path, result_count=1)
    line = filename + " " + str(prediction[0])
    # line += " (" + str(probability[0]) + ")"
    print(line)
    log.write(line + "\n")
log.close()
