# Employee-Attendance-System--Face-Recognition-And-Identification
Employee Attendance System
Having employee attendance management system is to keep proper track of the working hours of the employees. Taking and keeping employee attendance information in excel sheet can be wasting of time for HR employee so that smart attendance system using face recognition and identification can help saving time, cost, fake attendance and accuracy.

Face recognition is the task of making a positive identification of a face in a photo or video image against a pre-existing database of faces. Artificial Intelligence (AI) is describing different types of technologies .

Among them, computer vision is a kind of advanced technology that implements deep learning and pattern identification to interpret the content of an image, other text and video.

Computer vision is an integral field of AI , enabling computers to identify , process and interpret visual data.

Face recognition (FR) system identifies a face by matching it with the facial database. It has gained great progress in the recent years due to improvement in design and learning of features and face recognition models.

The techniques used in best facial recognition systems may depend on the
application of system.

This system may be divided into two categories
•Capture an image from a webcam camera , Flask API sends its image to the firebase database and then find a person from his/her image in that database. These systems returns the details of the person being searched for. Often only one image is available per person. It is usually not necessary for recognition to be done in real time.
•Identify a person in real time. These are used in systems which allow access to a certain group of people and deny access to others. Multiple images per person are often available for training and real time recognition is required.

First of all , I have collected employee images by using mobile IP Webcam connecting with PC . Using IP Webcam in mobile device :

Get started:

Here , collect employee images using IP Webcam :

url = ‘Fill in your device IP address'
cap = cv2.VideoCapture(url)
while(a<2000):
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canvas = detect(gray, frame)
if frame is not None:
cv2.imshow(‘frame’,frame)
print(“Number of image : “,a)
q = cv2.waitKey(1)
if q == ord(“q”):
break
cv2.destroyAllWindows()

Face detection is a kinds o f computer vision problem that can be helped humans to solve and has been solved reasonably well by classical feature-based techniques using the cascade classifier.

Most recently deep learning methods have achieved state-of-the-art results on standard benchmark face detection datasets.

After collecting the images cropped irrelevant parts of the face and cut save in the created folder.

Then , Loading the cascades
face_cascade = cv2.CascadeClassifier(‘haarcascade_frontalface_default.xml’)
a=0
# Defining a function that will do the detections
def detect(gray, frame):
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
global a
print (faces.shape)
print (“Number of faces detected: “ + str(faces.shape[0]))
print(“Data Found”)
cv2.putText(frame, “Number of faces detected: “ + str(faces.shape[0]), (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.rectangle(frame, (x+10, y+10), (x+w+10, y+h+10), (255, 0, 0), 2)
roi_gray = gray[y:y+h, x:x+w]
roi_color = frame[y+20:y+h+20, x+20:x+w+20]
cv2.imwrite(“/save/image_file_path/emp-”+str(a)+”.jpg”, roi_color)
a+=1
return frame

After that , an image becomes a detect image in a saving folder.

#### Figure 1:  Capture Face Detection in dlib

The proposed idea is for the second type of systems with varying facial details, expressions, and angles.

dlib is a toolkit for making real world machine learning and data analysis applications in C++. While the library is originally written in C++ , it has good, easy to use Python binding.

I have used dlib for face detection and the frontal face detector in dlib works really well.I have used face detector in dlib for HOG face recognition and linear SVM.

Histogram of Oriented Gradients (HOG) face recognition pipeline consists of four stages: face detection, face identification, face representation (or feature extraction), and classification.

The proposed method extracts facial features from input images and feeds them to support vector classifier for training and classification.

Data Preprocessing and Training an image:

Load images in saved folder :

def adjust_gamma(input_image, gamma=1.0):
table = np.array([((iteration / 255.0) ** (1.0 / gamma)) * 255
for iteration in np.arange(0, 256)]).astype(“uint8”)
return cv2.LUT(input_image, table)

def read_image(path, gamma=0.75):
output = cv2.imread(path)
return adjust_gamma(output, gamma=gamma)

def face_vector(input_image):
faces = facevec.detector(input_image, 1)
if not faces:
return None
f = faces[0]
shape = facevec.predictor(input_image, f)
face_descriptor = facevec.face_model.compute_face_descriptor(input_image, shape)
return face_descriptor

print(“Retrieving a saved folder images …”)
sub1=glob.glob(“./save/image_file_path/emp-/*.jpg”)
print(“Retrieved {} faces !”.format(len(sub1)))

Reading images :

print(“Reading saved folder images …”)
for i, sub in enumerate(sub2):
print(“Reading {} of {}\r”.format(i, len(sub2)))
face_vectors = face_vector(read_image(sub))
if face_vectors is None:
continue
vectors.append(dlib.vector(face_vectors))
labels.append(s2)

Building Model:

onehotencoder = OneHotEncoder()

lab = lab.reshape(-1,1)
lab.shape

lab = np.maximum(lab,0)

y = onehotencoder.fit_transform(lab).toarray()

x_train, x_test, y_train, y_test = train_test_split(X,lab,test_size = 0.2, random_state = 0)

classifier = SVC(kernel=’rbf’,C=10,probability= True)

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

### Here is a figure 2 result.
### Figure 2 : Capture Face Recognition And Identification

The system captures the images , then Flask API sends that image to the database and matches it with images inside from the database .

If an image is found in the database , then it will show the employee’s name with accuracy. If an image is not in the database,then it will show unknown as an employee’s name.

# * — — — — — SEND data to API — — — — — *
# Make a POST request to the API
if capture_count==0:
r = requests.post(url=’http://127.0.0.1:5000/receive_data’,json=front_data)
# Print to status of the request:
print(“Status: “, r.status_code)
capture_count+=1

else:break

After receiving data via API , then configured in firebase database:

firebase = pyrebase.initialize_app(firebaseConfig)
database = firebase.database()

#Initialize Flask Application
app=Flask(__name__)

#Create a employee
@app.route(‘/receive_data’,methods=[‘POST’])
def add_employee():

Name=request.json[‘Name’]
Time=request.json[‘Time’]
Date=request.json[‘Date’]
employeeid=request.json[‘employeeid’]

print(“Name : “,Name,”Time :” ,Time,”Date: “,Date,”Employee Id :”,employeeid)
#Store In firebase

designation=database.child(“EmployeeAttendance”).child(employeeid).child(“Details”).child(“Designation”).get()
emp_design=designation.val()
database.child(“EmployeeAttendance”).child(employeeid).child(“Details”).child(“Designation”).set(emp_design)
database.child(“EmployeeAttendance”).child(employeeid).child(“Details”).child(“Name”).set(Name)

db_get_info=database.child(“EmployeeAttendance”).child(employeeid).child(“Attendance”).child(Date).get()
time_value=db_get_info.val()
time_str=”Time-1"
num_count=0
if not time_value:
database.child(“EmployeeAttendance”).child(employeeid).child(“Attendance”).child(Date).child(time_str).set(Time)
else:
count_int=len(time_value)
count_attr=int(len(time_value))+1
con_cat_time=”Time-”+str(count_attr)
database.child(“EmployeeAttendance”).child(employeeid).child(“Attendance”).child(Date).child(con_cat_time).set(Time)

firebase = pyrebase.initialize_app(firebaseConfig)
database = firebase.database()

The system stores and synchronizes the facial fine points of the employees in the firebase database.

Once the process is done, the employee only needs to look at the camera and the attendance is automatically marked in the face recognition attendance system.

The admin can easily review the employee attendance records information by using google sheet.

date_check=sheet.findall(Date)
print(“Date Check : “,date_check)
result=0

if len(sheet.findall(Name)) <1:
# Insert the list as a row
insertRow = [Date,employeeid,Name,Time]
sheet.insert_row(insertRow,(index_data+1))
else:
### check employee name
if len(date_check) <1:
insertRow = [Date,employeeid,Name,Time]
sheet.insert_row(insertRow,(index_data+1))
else:
for k in range(sheet.row_count):
if k <=len(data):
j=k+1
row=sheet.row_values(j)[0:3]
if row[0]==Date and row[2]==Name:
result +=j ###update
sheet.update_cell(result,5,Time)
pass
else:
result =0 ####insert new record
else:break

if result == 0:
insertRow = [Date,employeeid,Name,Time]
sheet.append_row(insertRow)

The purpose of this system is used for the employee’s check-in,check-out per day and can easily know who is early in,early out , check-in late and check-out late during theirs working time.

In addition , the system is used for not only employee but also considered a person who is not included in the employee database ,then will show us ‘unknown’ employee name.
