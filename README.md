## Development Environment

### Required software:
* `pip install virtualenvwrapper`
* `mkvirtualenv motion`
* `workon motion`
* `brew install opencv3 --with-contrib --with-python3`
  * `ln -s /usr/local/opt/opencv/lib/python3.6/site-packages/cv2.cpython-36m-darwin.so /usr/local/opt/opencv/lib/python3.6/site-packages/cv2.so`

### Post Environment Creation:
```
ln -s /usr/local/opt/opencv/lib/python3.6/site-packages/cv2.so $VIRTUAL_ENV/lib/python3.6/site-packages/cv2.so
```

## Steps For Running Facial Recognition

### 1. Collect Training Images

###### Option 1 (Simple Version)
* Run `python capture.py`
* Enter a number for the command line prompt 'Enter user id: '
* 40 pictures will be taken of the user's face
* Pictures will be saved to the `images` folder

###### Option 2 (Manually Add Pictures)
* Add face pictures to the `images` folder
* For each person, save images in the format `UserX.Y.jpg`
  * Where X is the User ID, and Y is the image number for that user
  * The 20th picture for a User with an ID of 1 would be `User1.20.jpg`

### 2. Train Model
* Run `python train.py`
  * Will save a file `trainingdata.yml` to the `train` folder

### 3. Classify Faces
* Run `python classify.py`
  * Takes in a video stream and identifies faces in each frame
* Optional: Update empty dict in `classify.py` to map User ID to a name
  * Where User ID is the key and the person's name is the value
