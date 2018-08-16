import xml.etree.ElementTree as ET
import re
import json
import os

# field to parse if we decide to sepreate into columns
fieldsToParse = set(['Report Status', 'ADMISSION DATE', 'DISCHARGE DATE', 'ADDENDUM TO DISCHARGE SUMMARY',
'ADMISSION DIAGNOSIS', 'DISCHARGE DIAGNOSIS', 'DISCHARGE MEDICATIONS' 'MEDICATIONS ON DISCHARGE', 'OTHER DIAGNOSIS', 'DIET',
'RETURN TO WORK', 'PRINCIPAL PROCEDURE', 'HISTORY OF PRESENT ILLNESS','HISTORY OF PRESENT ILLNESS / REASON FOR HOSPITALIZATION' ,
'PAST MEDICAL HISTORY', 'BRIEF RESUME OF HOSPITAL COURSE', 'HOSPITAL COURSE AND TREATMENT', 'HOSPITAL COURSE',
'PAST SURGICAL HISTORY', 'ALLERGIES', 'MEDICATIONS', 'MEDICATIONS ON ADMISSION', 'ADMIT DIAGNOSIS', 'PRINCIPAL DISCHARGE DIAGNOSIS',
'FAMILY HISTORY', 'SOCIAL HISTORY', 'HABITS', 'VITAL SIGNS', 'DIAGNOSTIC STUDIES', 
'DISCHARGE ORDERS', 'DISCHARGE INSTRUCTIONS', 'DISPOSITION', 'DISCHARGE DISPOSITION', 'DISPOSITION / PLAN ON DISCHARGE',
'PHYSICAL EXAMINATION','LABORATORY DATA', 'ADMISSION LABORATORY DATA', 'TR', 'DD', 'TD', 'FOLLOW UP']) 

data = {}
fieldPostRegex = '\s+:([\S+\n\r\s]*?).*[A-Z].*[A-Z].*\s:'

def xmlToJson():
    #code for converting each field as a column 
    tree = ET.parse('new_smoker_train.xml')
    root = tree.getroot()
    total = 0
    for record in root:
        recordId = record.attrib['ID']
        data[recordId] = {}
        total += 1
        # if (total == 10):
        #     break;
        print ('converting record no:...' + str(total) + '...' + recordId)
        for textTag in record:
            # print str(textTag.text)
            for tagToSearch in fieldsToParse:
                expr = tagToSearch + fieldPostRegex
                # print ('Tag is :' + tagToSearch + "..searching...")
                matched = re.search(expr, str(textTag.text), re.MULTILINE)
                # print matches
                if matched:
                    # print(matched.group(1))
                    data[recordId][tagToSearch] = matched.group(1).strip()
                else:
                    # print ("no result")
                    data[recordId][tagToSearch] = None

    with open("new_smoker_train.json","w") as f:
        json.dump(dict(data),f)

        
       
def xmlToJsonHighLevel(fileName):
    # code for converting the whle text tag as one
    tree = ET.parse(fileName)
    root = tree.getroot()
    total = 0
    for record in root:
        recordId = record.attrib['ID']
        data[recordId] = {}
        total += 1
        # if (total == 10):
        #     break;
        #print ('converting record no:...' + str(total) + '...' + recordId)
        #print ('id is:' + str(recordId))

        # if (total == 105):
        #     print('issue')
        children = record.getchildren()
        if len(children) == 2:
            data[recordId][children[0].tag] = children[0].attrib['STATUS']
        data[recordId][children[len(children) - 1].tag] = children[len(children) - 1].text.replace('\n','') 

    nameOfFileWithoutExt = os.path.splitext(fileName)[0]
    jsonFileName = nameOfFileWithoutExt + ".json"
    with open(jsonFileName,"w") as f:
        json.dump(dict(data),f)

    #print ('done: ' + jsonFileName)
    return jsonFileName


#  Uncomment this line if you want to run the high level code
# xmlToJsonHighLevel("smokers_surrogate_train_all_version2.xml")
#xmlToJsonHighLevel("unannotated_test.xml")  # testing
