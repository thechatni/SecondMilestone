from django.shortcuts import render
from django.contrib.sessions.models import Session
import speech_recognition as sr
from django.http import JsonResponse, HttpResponse
import json
import spacy
import en_core_web_sm


def home(request):

    request.session['wrong'] = 0
    request.session['flag1'] = 0
    request.session['flag2'] = 0
    request.session['flag3'] = 0
    request.session['stop'] = 0
    request.session['one'] = 1
    request.session['too'] = 0
    request.session['tree'] = 0
    request.session['check'] = 0.4
    return render(request, 'home.html')


def narration(request):
    from keras.models import load_model
    from time import sleep
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing import image
    import cv2
    import numpy as np
    face_classifier = cv2.CascadeClassifier(
        'main/static/main/haarcascade_frontalface_default.xml')
    classifier = load_model('main/static/main/Emotion_Detection.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                print("\nprediction = ", preds)
                label = class_labels[preds.argmax()]
                print("\nprediction max = ", preds.argmax())
                print("\nlabel = ", label)
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            print("\n\n")
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'narration.html')


def speech_to_text(request):

    return render(request, 'speech_to_text.html')


def writeToExcel(request):
    import smtplib as sm
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import pandas as p

    nCount = 0
    yCount = 0
    ans = 'ttt'
    data = p.read_excel('main/static/main/test_data.xls')
    data_col = data.get('Verdict')

    print(data_col)
    for x in data_col:
        if (x == 'Yes'):
            yCount += 1
        else:
            nCount += 1

    if (yCount > nCount):
        print('YES')
    else:
        print('NO')

    return HttpResponse(json.dumps(ans))


def g1(request):
    import speech_recognition as sr
    wrong = request.session.get('wrong')
    flag1 = request.session.get('flag1')
    check = request.session.get('check')
    word1 = 'it was a Honda'
    judge = 'OK'
    context = {}
    if(request.POST):
        # get audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak:")
            audio = r.listen(source)

        try:
            text1 = r.recognize_google(audio)

            nlp = en_core_web_sm.load()

            doc1 = nlp(word1)
            doc2 = nlp(text1)
            res = doc1.similarity(doc2)
            print(res)
            if res > check:
                judge = "Hmm, I see"
            else:
                judge = "Yeah, not sure about that"
                wrong = wrong+1
            output = " " + text1
        except sr.UnknownValueError:
            output = "Please press record when you are ready to speak"
        except sr.RequestError as e:
            output = "Could not request results; {0}".format(e)

    context['data'] = output
    context['judge'] = judge
    request.session['wrong'] = wrong
    request.session['flag1'] = flag1
    context['flag'] = flag1
    ##################################################################

    return HttpResponse(json.dumps(context))


def g2(request):
    import speech_recognition as sr
    wrong = request.session.get('wrong')
    flag2 = request.session.get('flag2')
    check = request.session.get('check')
    word1 = 'it was a blue bandana'
    judge = 'OK'
    context = {}
    if(request.POST):
        # get audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak:")
            audio = r.listen(source)

        try:
            text1 = r.recognize_google(audio)
            nlp = en_core_web_sm.load()

            doc1 = nlp(word1)
            doc2 = nlp(text1)
            res = doc1.similarity(doc2)
            print(res)
            if res > check:
                judge = "Hmm, I see"
            else:
                judge = "Yeah, not sure about that"
                wrong = wrong+1
            output = " " + text1
        except sr.UnknownValueError:
            output = "Please press record when you are ready to speak"
        except sr.RequestError as e:
            output = "Could not request results; {0}".format(e)

    context['data'] = output
    context['judge'] = judge
    request.session['wrong'] = wrong
    request.session['flag2'] = flag2
    context['flag'] = flag2
    ##################################################################

    return HttpResponse(json.dumps(context))


def g3(request):
    import speech_recognition as sr
    wrong = request.session.get('wrong')
    flag3 = request.session.get('flag3')
    check = request.session.get('check')
    word1 = 'it was 1962'
    judge = 'OK'
    context = {}
    if(request.POST):
        # get audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak:")
            audio = r.listen(source)

        try:
            text1 = r.recognize_google(audio)
            nlp = en_core_web_sm.load()

            doc1 = nlp(word1)
            doc2 = nlp(text1)
            res = doc1.similarity(doc2)
            print(res)
            if res > check:
                judge = "Hmm, I see"
            else:
                judge = "Yeah, not sure about that"
                wrong = wrong+1
            output = " " + text1
        except sr.UnknownValueError:
            output = "Please press record when you are ready to speak"
        except sr.RequestError as e:
            output = "Could not request results; {0}".format(e)

    context['data'] = output
    context['judge'] = judge
    request.session['wrong'] = wrong
    request.session['flag3'] = flag3
    context['flag'] = flag3
    ##################################################################

    return HttpResponse(json.dumps(context))


def facialexp(request):
    from keras.models import load_model
    from time import sleep
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing import image
    import cv2
    import numpy as np
    face_classifier = cv2.CascadeClassifier(
        'main/static/main/haarcascade_frontalface_default.xml')
    classifier = load_model('main/static/main/Emotion_Detection.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count1 = 0
    count2 = 0
    count3 = 0
    noMore1 = 0
    noMore2 = 0
    noMore3 = 0
    hap_count = 0
    ang_count = 0
    neut_count = 0
    sad_count = 0
    surp_count = 0

    while True:
        # Grab a single frame of video
        one = request.session.get('one')
        too = request.session.get('too')
        tree = request.session.get('tree')
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        label = ''
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
                # print("\nprediction = ", preds)
                label = class_labels[preds.argmax()]
                flag = 'no'
                if (one == 1):
                    if (label == 'Neutral'):
                        neut_count += 1
                    # if (label == 'Angry'):
                    #     ang_count+=1
                    # if (label == 'Sad'):
                    #     sad_count+=1
                    # if (label == 'Surprise'):
                    #     surp_count+=1
                    count1 = count1 + 1
                    print(count1)
                    if (count1 >= 50):

                        if(neut_count < 35):
                            request.session['flag1'] = 1
                            print('flag1')

                        resp = {}
                        # resp['label'] = label
                        return HttpResponse(json.dumps(resp))

                elif (too == 1):
                    if (label == 'Neutral'):
                        neut_count += 1
                    # if (label == 'Angry'):
                    #     ang_count+=1
                    # if (label == 'Sad'):
                    #     sad_count+=1
                    # if (label == 'Surprise'):
                    #     surp_count+=1
                    count1 = count1 + 1
                    print(count1)
                    if (count1 >= 50):

                        if(neut_count < 35):
                            request.session['flag2'] = 1
                            print('flag2')

                        resp = {}
                        # resp['label'] = label
                        return HttpResponse(json.dumps(resp))

                elif (tree == 1):
                    if (label == 'Neutral'):
                        neut_count += 1
                    # if (label == 'Angry'):
                    #     ang_count+=1
                    # if (label == 'Sad'):
                    #     sad_count+=1
                    # if (label == 'Surprise'):
                    #     surp_count+=1
                    count1 = count1 + 1
                    print(count1)
                    if (count1 >= 50):

                        if(neut_count < 35):
                            request.session['flag3'] = 1
                            print('flag3')

                        resp = {}
                        # resp['label'] = label
                        return HttpResponse(json.dumps(resp))

                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            print("\n\n")
            if request.session.get('stop') == 1:
                break

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    resp = {}
    resp['label'] = label
    return HttpResponse(json.dumps(resp))

    # cap.release()
    # cv2.destroyAllWindows()

    # resp['artistName'] = names[0].text
    # resp['artistImage'] = images[7].get('content')

    # return HttpResponse(json.dumps(resp))


# def changeFlag(request):
#     request.session['flag1'] = 1


def two(request):
    request.session['one'] = 0
    request.session['too'] = 1
    request.session['tree'] = 0
    return render(request, 'two.html')


def three(request):
    request.session['one'] = 0
    request.session['too'] = 0
    request.session['tree'] = 1
    return render(request, 'three.html')


def decision(request):

    flag1 = request.session.get('flag1')
    flag2 = request.session.get('flag2')
    flag3 = request.session.get('flag3')

    data = request.POST.get('record')
    wrong = request.session.get('wrong')
    request.session['stop'] = 1
    sum = flag1+flag2+flag3
    import speech_recognition as sr
    if(wrong >= 2 or sum >= 2):
        final = "I don't believe you are telling the truth. You are now a prime suspect!"
    else:
        final = "Yep, your answers check out. Sorry for disturbing you. You're free to go"
    context = {}
    context['final'] = final

    return render(request, 'decision.html', context)
