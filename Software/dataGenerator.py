from bs4 import BeautifulSoup
import glob
import shutil
import re

print("welcome to the data generator script")
img_dir = "C:\\Users\\Bogdan\\Repos\\HIT-UAV-Infrared-Thermal-Dataset\\normal_xml\\JPEGImages"
anotation_dir = "C:\\Users\\Bogdan\Repos\\HIT-UAV-Infrared-Thermal-Dataset\\normal_xml\\Annotations"

for annotation_path in glob.iglob(f'{anotation_dir}\\*.xml'):
    with open(annotation_path, 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "xml")
    b_name = Bs_data.find_all("name")
    fileName = re.search("\d_.*xml$",annotation_path)
    print(annotation_path)
    print(annotation_path[fileName.start():fileName.end()-4])
    fileName = annotation_path[fileName.start():fileName.end()-4]

    personPresent = False
    for tag in b_name:

        if tag.getText() == "Person":
            personPresent = True
            src = img_dir+"\\"+fileName+".jpg"
            print(src)
            des = "C:\\Users\\Bogdan\\Repos\\AERO-Connect2023\\AERO-Connect2023\\Software\\DataRaw\\test\\people"
            shutil.copy(src, des)
            break
        else:
            src = img_dir + "\\" + fileName + ".jpg"
            des = "C:\\Users\\Bogdan\\Repos\\AERO-Connect2023\\AERO-Connect2023\\Software\\DataRaw\\test\\noPerson"
            shutil.copy(src, des)








# "C:\Users\Bogdan\Repos\HIT-UAV-Infrared-Thermal-Dataset\normal_xml\Annotations\0_60_30_0_01609.xml"


'''

with open(anotation_dir+"\\0_60_30_0_01609.xml", 'r') as f:
    data = f.read()

Bs_data = BeautifulSoup(data, "xml")

b_name = Bs_data.find_all("name")

print(b_name[0].getText())

'''
