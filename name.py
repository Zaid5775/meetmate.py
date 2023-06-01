import spacy

nlp = spacy.load("en_core_web_sm")

text = "Ms. Johnson is a great teacher. Zaid Umer SAhil always salim khan makes class interesting and engaging."

doc = nlp(text)

teacher_names = []
for ent in doc.ents:
    if ent.label_ == "PERSON":
        teacher_names.append(ent.text)

print(teacher_names)
