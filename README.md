** Hand Gesture Recognition using Machine learning **

This project aims to harness the power of computer vision and machine learning
to recognize and translate hand gestures in real-time. By creating an
intuitive and accessible platform, we seek to empower disabled individuals,
enabling seamless interaction in everyday situations—whether in educational
settings, public services, or personal communication.

**About the repositary files and order of execution of files:-**

**1)Handlandmarker.py**
This program is coded to access the camera and display the skeletal coordinates of the detected hands.
This is run to check if the camera is accessed and configured properly.

**2)Handlandmarker_model.py**
This program is run to custom train various hand gestures that are shown to the camera with custom names as inputted by the user.
The gestures that are inputted are processed and made into a CSV file ('Handmarks.csv')and is stored locally for training a model.

**3)Handlandmarktrain.py**
This program is run to develop a local trained model.

pipelines = {
    'lr':make_pipeline(StandardScaler() , LogisticRegression()),
    'rc':make_pipeline(StandardScaler() , RidgeClassifier()),
    'rf':make_pipeline(StandardScaler() , RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler() , GradientBoostingClassifier())
}

This program develops four model based on the above mentioned algorithm.

The user can then check the accuracy score of each trained algorithm then choose the best fit algorithm.
for algo , model in fit_models.items():
    pred = model.predict(X_test)
    print(algo , accuracy_score(y_test , pred))
with open('Hand_test_new.pkl' , 'wb') as f:
    pickle.dump(fit_models['rf'], f)

The program then creates a pickle file(trained model) which can used to check of gestures in real time.

**4)handmarkeroutput.py**
This program runs to detect and recognize various gestures in real-time that are pre-trained in the model as done earlier.

**I Hope this code is useful to developers out there and as well as students drop a like as an appreciation. Thanks in Advance ;)**




