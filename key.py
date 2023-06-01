from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# summary = "The most important details in this text are that the recording and transcription is now started, the Sumi libraries are being evaluated, the timelines and action points of the other team members are being reviewed, and the next week is being reviewed to determine who is expected to do what. The most important details in this text are that the project is about team meeting and that the team should be having more meetings to understand how team meetings work and how they can pick up action points and transcribe them and then summarize them into the suitable action. Additionally, the team should identify what keywords they are trying to pick and use them in their transcription and summarization. Finally, the transcription will be using a machine learning algorithm and the siomi."
summary = 'The recording and transcription is now started. Umar will discuss some of the points on Omars project with Zafar Abbasi today. Omar is going to be checking out with the Sumi technology next week.Zafar Abbasi and Umer khan are discussing the work distribution of their team members. They agree that it is important to set some timelines and the distribution of work has to be clearly known.Zafar Abbasi is glad that one person shows the best initiative in the project. Umer khan wants Zafar to come back with a visible timeline and plan for the actions that he has to complete.Zafar Abbasi and Umer Khan are talking about how to use the technology in the transcription and summarization of a document.Zafar Abbasi and Umer khan are talking about the process of summarizing a speech. The summary picks the word and puts it in the summary. The words are relevant throughout. The clarity of speech is importantZafar Abbasi and Umer khan are talking about summarizing. Zafar asks Umer about how summarization works. Umer explains that summarization is about picking up the keywords and the associated actions andFor the unsupervised machine learning algorithm, will see the relevancy of the world or it will not see that who is the speaker who is speaking that particular line or particular phrase, so that may be an issue which we havent Umer khan and Zafar Abbasi will try to make sure the transcription and summarization tool is able to consider as much as possible in the meetings.Zafar Abbasi explains to Umer Khan that it is always somebodys task to do this and if your project guess the nature of success that you are aiming for it to do, then it could be the next bigUmer khan explains to Zafar Abbasi that the Assembly AI needs a token which they havent used, so they are only using the Google speech recognition. The issue is when the video length is too long orUmer khan, Zafar Abbasi and Umer Khan are talking about the level of accuracy of the video. The video has to be recorded with a transcript. The summarization part has not been done yet.Zafar Abbasi and Umer khan have just been working on the conversion. Umer is working with the logic from MP4 to MP3 conversion, then MP3 to WAV conversion dot WAV, thenUmer khan and Zafar Abbasi talk about the project. The project manager is going to develop the interface in a way that is user-friendly.Zafar Abbasi, Umer khan and Umer Khan are going to use the voice sample next week to work on the project. The project will be completed in 6-6 to 8 weeks.Zafar Abbasi and Umer khan are talking about how to measure whether they are able to reach where they want to reach week on week and whether they have met the objective of making sure they can convert a meetingZafar Abbasi sends the recording to Umer Khan so that he can work with it forward. Umer has a good weekend.'
# Tokenize the summary into words
words = summary.split()

# Join the words back into a string
text = " ".join(words)

# Create a TF-IDF vectorizer
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Compute the TF-IDF scores
scores = vectorizer.fit_transform([text])

# Get the feature names (i.e., the words)
feature_names = vectorizer.get_feature_names()

# Get the TF-IDF scores for each word
tfidf_scores = scores.toarray()[0]

# Create a list of (word, score) pairs
word_scores = [(word, score) for word, score in zip(feature_names, tfidf_scores)]

# Sort the (word, score) pairs by score in descending order
word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

# Get the top 10 words
top_words = [word for word, score in word_scores[:10]]

print(top_words)