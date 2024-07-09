import cv2
import numpy as np
import face_recognition
import os 

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])#عشان اطلع الاسم من غير الامتداد بتاعه
print(classNames)#هطبع ال اسماء كلها في ال ترمينال من غير الامتداد

def findEncodeings(image):#هنا هبتدي اتعرف ع الصور 
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#بحول صيغة الصورة
        encode = face_recognition.face_encodings(img)[0]#هنا المكتبة هتتعرف علي كل النقط الي ف الصورة
        encodeList.append(encode)#هحط البيانات الي اتاخدت ف الخطوة الي فاتت ع الصور
    return encodeList

encodeListKnown = findEncodeings(images)#هربط الاسماء بالصور بالبيانات و بعدين احطهم ف ليست و اطبع
print('Encoding Complete.')#لما اعمل رن هيطلع الرسالة دي

cap = cv2.VideoCapture(0)#كود استخدام الكاميرا 

while True:
    _, img = cap.read() #هو هنا بيقرا الفيديو علي هيئة صور
    img = cv2.flip(img , 1)  #عشان اقلب الصورة

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # type: ignore #هنا هصغر الصورة للربع عشان اسرع العملية
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)# بحول عشان اعمل انكووود

    faceCurentFrame = face_recognition.face_locations(imgS) #بحدد مكان الوش
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)#بعمل انكوود للوش

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):#هبدا اقارن بين الصور الي ف الفولدر عن طريق اني احطهم ف مصفوفة
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)#بتقارن
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)#بتكمل المقارنه بحساب المسافات بين النقط ف كل وش
        matchIndex = np.argmin(faceDis)#ف الخطوات الي فاتت حسب قيم و الصورة الصح ف الفولدر خدت اقل قيمة

        if matches[matchIndex]:#لو الوش الي قدامي محطوط ف الداتا هعمل التالي
            name = classNames[matchIndex].upper()
            print(name)#هاخد الاسم و اطبعه
            y1, x2, y2, x1 = faceLoc#هعرف اربع متغيرات و هما نقط المربع الي حوالين الوش
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 #هضرب في اربعة عشان فوق قسمت عليها
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2) #رسم المربع حوالين الصورة
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED) #المربع الي تحت الاسم الي هحط فيه الوش
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2) #كتابة الاسم الي هيتحط

    cv2.imshow('Face Recogntion', img) #عشان اعرض الفيديو
    cv2.waitKey(1)