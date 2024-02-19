from imageai.Classification import ImageClassification
import os 

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2() #This sets the model we want to use a few are already preloaded 
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2-b0353104.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "house.jpg"), result_count=3 )  #Result count is how many predictions we want the model to give us
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(f'{eachPrediction} , : , {eachProbability}')
